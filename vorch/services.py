"""The service manifest — what vorch depends on, what it costs, and who may start it.

vorch drives other people's servers over HTTP and contains no model of its own, so before it can
do anything a handful of processes must be running: PostgreSQL, Gemma (llama-server), ComfyUI,
sometimes OmniVoice, the API and the UI. This module is the *declaration* of that list. It reads
YAML and answers questions about it. It starts nothing, stops nothing, and speaks to no service —
that is ``vorch.supervisor``, which imports this. Keeping the split means the description of the
world stays testable without a GPU, a network or a subprocess.

Built-in defaults ship in ``services_builtin/services.yaml``; a user file
(``config.SERVICES_FILE``, ``~/vorch/services.yaml``) is merged **over** them per service, key by
key, so changing one path does not mean restating an entry. Two conveniences in values:

``${VORCH_X}``
    the environment variable, falling back to ``config.X`` — so the manifest follows whatever you
    already set in ``~/vorch/vorch.env`` and needs no second copy of a URL. A non-``VORCH_``
    name resolves from the environment only.

``@file:PATH``
    the file's stripped contents, resolved relative to the service's ``cwd``. Used for
    OmniVoice's reference transcript, which must be passed verbatim. Read late (when the
    environment is actually built), so a missing file is a start-time error with a clear message
    rather than an import-time crash.

Fields that carry real weight, rather than documentation:

``manage``  ``never`` probe and print ``hint`` · ``adopt`` report, never spawn · ``full`` may spawn
``vram``    ``class`` ``none`` / ``resident`` (fixed tax) / ``elastic`` (freeable without stopping)
            / ``exclusive`` (its presence forbids any other heavy step), with a **measured**
            ``holds_mb``. The manifest is where measurements live so docstrings stop carrying
            stale numbers.
``lifetime`` ``long`` · ``warm`` (kept for ``idle_timeout``, then reaped) · ``phase``
``adopt``   ``verify`` ``port`` / ``argv`` / ``env`` and ``on_unverified``. Verifying by ``env``
            is how a narration server reading the wrong cloned voice becomes visible: the voice is
            passed only as an environment variable and ``/health`` never reports it back.
``match``   how to recognise the process, when it is not the one we would spawn — ``comfy launch``
            forks ``python main.py`` and the *grandchild* is what holds the VRAM.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from . import config

log = logging.getLogger("vorch.services")

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MANAGE = ("never", "adopt", "full")
LIFETIMES = ("long", "warm", "phase")
VRAM_CLASSES = ("none", "resident", "elastic", "exclusive")
PROBES = ("http", "dsn", "none")
VERIFY = ("port", "argv", "env")
ON_UNVERIFIED = ("warn", "refuse", "adopt")

BUILTIN_FILE = Path(__file__).resolve().parent / "services_builtin" / "services.yaml"

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_FILE_PREFIX = "@file:"


class ServiceError(ValueError):
    """The manifest is invalid, or a value in it cannot be resolved."""


# --- value resolution ---------------------------------------------------------------


def resolve_vars(value: str, *, where: str) -> str:
    """Expand ``${NAME}``. ``VORCH_X`` falls back to ``config.X`` when unset in the environment."""

    def one(m: re.Match) -> str:
        name = m.group(1)
        got = os.environ.get(name)
        if got is None and name.startswith("VORCH_"):
            attr = getattr(config, name[len("VORCH_"):], None)
            got = None if attr is None else str(attr)
        if got is None:
            raise ServiceError(f"{where}: ${{{name}}} is not set and has no config default")
        return got

    return _VAR_RE.sub(one, value)


def _expand(value: str, *, where: str) -> str:
    return str(Path(v).expanduser()) if (v := resolve_vars(value, where=where)).startswith("~") else v


# --- the pieces of a service --------------------------------------------------------


@dataclass(frozen=True)
class HealthProbe:
    """How to ask a service whether it is up. ``none`` means we cannot, only look for the process."""

    probe: str = "none"
    url: str = ""
    target: str = ""            # dsn probes
    timeout: float = 3.0
    expect_json: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"probe": self.probe, "timeout": self.timeout}
        if self.url:
            d["url"] = self.url
        if self.target:
            d["target"] = self.target
        if self.expect_json:
            d["expect_json"] = dict(self.expect_json)
        return d


@dataclass(frozen=True)
class VramSpec:
    """What a service costs the card, and whether it can be made to give it back."""

    kind: str = "none"          # one of VRAM_CLASSES ("class" is a keyword)
    needs_mb: int = 0           # what it wants when fully loaded
    holds_mb: int = 0           # MEASURED resting footprint
    free_endpoint: str | None = None   # elastic only: freeable without stopping

    @property
    def elastic(self) -> bool:
        return self.kind == "elastic"

    @property
    def exclusive(self) -> bool:
        return self.kind == "exclusive"

    def to_dict(self) -> dict:
        d = {"class": self.kind, "needs_mb": self.needs_mb, "holds_mb": self.holds_mb}
        if self.free_endpoint:
            d["free_endpoint"] = self.free_endpoint
        return d


@dataclass(frozen=True)
class AdoptSpec:
    """Whether an already-running instance may be used, and how hard we check it is the right one."""

    allowed: bool = True
    verify: str = "port"
    on_unverified: str = "warn"

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "verify": self.verify, "on_unverified": self.on_unverified}


@dataclass(frozen=True)
class MatchSpec:
    """How to recognise the running process among all of /proc."""

    cmdline_contains: tuple[str, ...] = ()
    cwd: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"cmdline_contains": list(self.cmdline_contains)}
        if self.cwd:
            d["cwd"] = self.cwd
        return d


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    description: str = ""
    manage: str = "adopt"
    cwd: str = "."
    argv: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    health: HealthProbe = field(default_factory=HealthProbe)
    ready_timeout: float = 120.0
    depends_on: tuple[str, ...] = ()
    vram: VramSpec = field(default_factory=VramSpec)
    lifetime: str = "long"
    idle_timeout: float | None = None
    adopt: AdoptSpec = field(default_factory=AdoptSpec)
    evict_adopted: bool = False
    match: MatchSpec = field(default_factory=MatchSpec)
    log: str = ""
    hint: str = ""
    builtin: bool = True
    source: str = ""

    # -- derived -----------------------------------------------------------------

    @property
    def managed(self) -> bool:
        """May vorch spawn and stop this itself?"""
        return self.manage == "full"

    @property
    def cwd_path(self) -> Path:
        """``cwd`` as an absolute path; a relative one is relative to the repo root."""
        p = Path(self.cwd).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def log_path(self) -> Path:
        return config.LOG_DIR / (self.log or f"{self.name}.log")

    @property
    def pidfile(self) -> Path:
        return config.RUN_DIR / f"{self.name}.json"

    def argv_expanded(self) -> list[str]:
        """The command, with ``~`` and ``${VAR}`` resolved. What you would type yourself."""
        return [_expand(a, where=f"service {self.name!r} argv") for a in self.argv]

    def env_resolved(self) -> dict[str, str]:
        """``env`` with ``@file:`` contents inlined. Raises ServiceError on a missing file."""
        out: dict[str, str] = {}
        for key, raw in self.env.items():
            where = f"service {self.name!r} env {key}"
            value = resolve_vars(str(raw), where=where)
            if value.startswith(_FILE_PREFIX):
                ref = Path(value[len(_FILE_PREFIX):]).expanduser()
                path = ref if ref.is_absolute() else self.cwd_path / ref
                try:
                    value = path.read_text(encoding="utf-8").strip()
                except OSError as e:
                    raise ServiceError(f"{where}: cannot read {path} ({e})") from None
            out[key] = value
        return out

    def health_url(self) -> str:
        """The probe URL with ``${VAR}`` resolved (empty for a non-http probe)."""
        return (resolve_vars(self.health.url, where=f"service {self.name!r} health.url")
                if self.health.probe == "http" and self.health.url else "")

    def health_target(self) -> str:
        """The dsn a ``dsn`` probe connects to, resolved (empty for other probes)."""
        return (resolve_vars(self.health.target, where=f"service {self.name!r} health.target")
                if self.health.probe == "dsn" and self.health.target else "")

    def identity(self) -> dict[str, str]:
        """The env entries that say *which* instance this is — the cloned voice, above all.

        Long values (a whole transcript) are recorded as a short digest, so a pidfile or a log
        header stays readable while still detecting a change.
        """
        import hashlib

        out: dict[str, str] = {}
        for key, value in self.env_resolved().items():
            out[key] = (value if len(value) <= 80
                        else "sha1:" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12])
        return out

    def matcher(self) -> MatchSpec:
        """``match`` if given, else derived from argv — the last path-like argument and the cwd."""
        if self.match.cmdline_contains or self.match.cwd:
            return MatchSpec(
                tuple(_expand(s, where=f"service {self.name!r} match") for s in self.match.cmdline_contains),
                _expand(self.match.cwd, where=f"service {self.name!r} match cwd") if self.match.cwd else None,
            )
        # Derived fallback: the executable's own name plus the cwd. Deliberately weak — a
        # launcher that execs or forks into something else (serve_bg.sh, `comfy launch`,
        # `npm run dev`) leaves nothing of its argv in the final cmdline, so those entries pin
        # `match:` explicitly. If a service is not being found, that is the field to set.
        argv = self.argv_expanded()
        needle = Path(argv[0]).name if argv else ""
        return MatchSpec((needle,) if needle else (), str(self.cwd_path))

    def to_dict(self, *, resolve: bool = False) -> dict:
        """The entry as parsed. ``resolve=True`` also gives the command you could run by hand."""
        d: dict = {
            "name": self.name, "description": self.description, "manage": self.manage,
            "cwd": self.cwd, "argv": list(self.argv), "env": dict(self.env),
            "health": self.health.to_dict(), "ready_timeout": self.ready_timeout,
            "depends_on": list(self.depends_on), "vram": self.vram.to_dict(),
            "lifetime": self.lifetime, "idle_timeout": self.idle_timeout,
            "adopt": self.adopt.to_dict(), "evict_adopted": self.evict_adopted,
            "log": str(self.log_path), "hint": self.hint,
            "builtin": self.builtin, "source": self.source,
        }
        if resolve:
            d["resolved"] = {
                "cwd": str(self.cwd_path),
                "argv": self.argv_expanded(),
                "match": self.matcher().to_dict(),
                "health_url": self.health_url(),
                "health_target": self.health_target(),
            }
            # env and identity read files, which may be missing — report rather than raise, since
            # this feeds `vorch services list` and the API.
            try:
                d["resolved"]["env"] = self.env_resolved()
            except ServiceError as e:
                d["resolved"]["env_error"] = str(e)
        return d


REPO_ROOT = Path(__file__).resolve().parent.parent


# --- parsing ------------------------------------------------------------------------


def _one_of(name: str, field_name: str, value: Any, allowed: tuple[str, ...], default: str) -> str:
    v = str(value if value is not None else default).strip().lower()
    if v not in allowed:
        raise ServiceError(f"service {name!r}: {field_name} must be one of {allowed}, got {v!r}")
    return v


def _number(name: str, field_name: str, value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ServiceError(f"service {name!r}: {field_name} must be a number, got {value!r}") from None
    if v < 0:
        raise ServiceError(f"service {name!r}: {field_name} must be >= 0")
    return v


def _mapping(name: str, field_name: str, value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ServiceError(f"service {name!r}: {field_name} must be a mapping, got {type(value).__name__}")
    return value


def _strings(name: str, field_name: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ServiceError(f"service {name!r}: {field_name} must be a list, got {value!r}")
    return tuple(str(v) for v in value)


def _health(name: str, raw: Any) -> HealthProbe:
    d = _mapping(name, "health", raw)
    probe = _one_of(name, "health.probe", d.get("probe"), PROBES, "none")
    url = str(d.get("url") or "")
    target = str(d.get("target") or "")
    if probe == "http" and not url:
        raise ServiceError(f"service {name!r}: health.probe http needs a url")
    if probe == "dsn" and not target:
        raise ServiceError(f"service {name!r}: health.probe dsn needs a target")
    return HealthProbe(
        probe=probe, url=url, target=target,
        timeout=_number(name, "health.timeout", d.get("timeout"), 3.0) or 0.0,
        expect_json=_mapping(name, "health.expect_json", d.get("expect_json")),
    )


def _vram(name: str, raw: Any) -> VramSpec:
    d = _mapping(name, "vram", raw)
    kind = _one_of(name, "vram.class", d.get("class"), VRAM_CLASSES, "none")
    needs = int(_number(name, "vram.needs_mb", d.get("needs_mb"), 0) or 0)
    holds = int(_number(name, "vram.holds_mb", d.get("holds_mb"), 0) or 0)
    free_endpoint = str(d["free_endpoint"]) if d.get("free_endpoint") else None
    if kind == "elastic" and not free_endpoint:
        raise ServiceError(f"service {name!r}: vram.class elastic needs a free_endpoint "
                           "(the whole point is that it frees without stopping)")
    if kind == "none" and (needs or holds):
        raise ServiceError(f"service {name!r}: vram.class none cannot also want memory")
    return VramSpec(kind=kind, needs_mb=needs, holds_mb=holds, free_endpoint=free_endpoint)


def _adopt(name: str, raw: Any) -> AdoptSpec:
    d = _mapping(name, "adopt", raw)
    return AdoptSpec(
        allowed=bool(d.get("allowed", True)),
        verify=_one_of(name, "adopt.verify", d.get("verify"), VERIFY, "port"),
        on_unverified=_one_of(name, "adopt.on_unverified", d.get("on_unverified"), ON_UNVERIFIED, "warn"),
    )


def parse_service(name: str, raw: Any, *, builtin: bool = True, source: str = "") -> ServiceSpec:
    if not NAME_RE.match(name):
        raise ServiceError(f"service name {name!r} invalid (want {NAME_RE.pattern}; it names files)")
    d = _mapping(name, "entry", raw)
    manage = _one_of(name, "manage", d.get("manage"), MANAGE, "adopt")
    argv = _strings(name, "argv", d.get("argv"))
    if manage == "full" and not argv:
        raise ServiceError(f"service {name!r}: manage: full needs an argv to start it")
    lifetime = _one_of(name, "lifetime", d.get("lifetime"), LIFETIMES, "long")
    idle_timeout = _number(name, "idle_timeout", d.get("idle_timeout"), None)
    if lifetime == "warm" and not idle_timeout:
        raise ServiceError(f"service {name!r}: lifetime: warm needs an idle_timeout "
                           "(otherwise 'warm' just means 'left running')")
    match = _mapping(name, "match", d.get("match"))
    env = {str(k): str(v) for k, v in _mapping(name, "env", d.get("env")).items()}
    return ServiceSpec(
        name=name,
        description=str(d.get("description") or ""),
        manage=manage,
        cwd=str(d.get("cwd") or "."),
        argv=argv,
        env=env,
        health=_health(name, d.get("health")),
        ready_timeout=_number(name, "ready_timeout", d.get("ready_timeout"), 120.0) or 0.0,
        depends_on=_strings(name, "depends_on", d.get("depends_on")),
        vram=_vram(name, d.get("vram")),
        lifetime=lifetime,
        idle_timeout=idle_timeout,
        adopt=_adopt(name, d.get("adopt")),
        evict_adopted=bool(d.get("evict_adopted", False)),
        match=MatchSpec(_strings(name, "match.cmdline_contains", match.get("cmdline_contains")),
                        str(match["cwd"]) if match.get("cwd") else None),
        log=str(d.get("log") or ""),
        hint=str(d.get("hint") or ""),
        builtin=builtin,
        source=source,
    )


def _entries(doc: Any, *, where: str) -> dict:
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ServiceError(f"{where}: YAML root must be a mapping")
    entries = doc.get("services", doc if "services" not in doc else None)
    if entries is None:
        return {}
    if not isinstance(entries, dict):
        raise ServiceError(f"{where}: `services` must be a mapping of name -> settings")
    return entries


def _merge(base: dict, over: dict) -> dict:
    """``over`` wins, one nested mapping deep — so a user file can change a single field."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def parse(doc: Any, *, user: Any = None, source: str = "", user_source: str = "") -> dict[str, ServiceSpec]:
    """Specs by name from a built-in document and an optional user one merged over it."""
    base = _entries(doc, where=source or "services file")
    over = _entries(user, where=user_source or "user services file")
    out: dict[str, ServiceSpec] = {}
    for name in list(base) + [n for n in over if n not in base]:
        raw = _merge(_mapping(str(name), "entry", base.get(name)), _mapping(str(name), "entry", over.get(name))) \
            if name in over else base[name]
        out[str(name)] = parse_service(
            str(name), raw,
            builtin=name not in over,
            source=(user_source if name in over else source),
        )
    _check_dependencies(out)
    return out


def _check_dependencies(specs: dict[str, ServiceSpec]) -> None:
    for spec in specs.values():
        for dep in spec.depends_on:
            if dep not in specs:
                raise ServiceError(f"service {spec.name!r} depends on {dep!r}, which is not defined")
            if dep == spec.name:
                raise ServiceError(f"service {spec.name!r} depends on itself")
    order(specs)      # raises on a cycle


def order(specs: dict[str, ServiceSpec], names: list[str] | None = None) -> list[str]:
    """Start order: every dependency before the thing that needs it.

    Stable — services with no relation between them keep manifest order, so the output reads the
    same every time. Raises ServiceError naming the cycle.
    """
    wanted = list(specs) if names is None else list(names)
    out: list[str] = []
    state: dict[str, int] = {}          # 1 = visiting, 2 = done

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            cycle = " -> ".join(trail[trail.index(name):] + (name,))
            raise ServiceError(f"services: dependency cycle {cycle}")
        spec = specs.get(name)
        if spec is None:
            raise ServiceError(f"unknown service {name!r}; defined: {sorted(specs)}")
        state[name] = 1
        for dep in spec.depends_on:
            visit(dep, trail + (name,))
        state[name] = 2
        out.append(name)

    for name in wanted:
        visit(name, ())
    return out


# --- loading -------------------------------------------------------------------------

_cache: dict[str, tuple[tuple, dict[str, ServiceSpec]]] = {}


def _signature(p: Path) -> tuple:
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return ("missing",)


def _read(p: Path) -> Any:
    try:
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ServiceError(f"services file {p}: {e}") from None


def load(path: str | Path | None = None, *, force: bool = False) -> dict[str, ServiceSpec]:
    """Every service: the built-ins with ``path`` (default ``config.SERVICES_FILE``) merged over.

    Cached per user path and invalidated when either file appears or changes. ``${VAR}`` values are
    resolved when used, not here, so a changed environment does not need a cache flush.
    """
    p = Path(path) if path is not None else Path(config.SERVICES_FILE)
    key = str(p)
    sig = (_signature(BUILTIN_FILE), _signature(p))
    hit = _cache.get(key)
    if not force and hit is not None and hit[0] == sig:
        return hit[1]
    result = parse(
        _read(BUILTIN_FILE), user=_read(p) if p.is_file() else None,
        source=str(BUILTIN_FILE), user_source=str(p),
    )
    if p.is_file():
        overridden = sorted(n for n, s in result.items() if not s.builtin)
        log.info("services: %s (%d overridden from %s: %s)", ", ".join(result),
                 len(overridden), p, ", ".join(overridden) or "none")
    else:
        log.debug("services: %s (built-in only; no %s)", ", ".join(result), p)
    _cache[key] = (sig, result)
    return result


def get(name: str) -> ServiceSpec:
    try:
        return load()[name]
    except KeyError:
        raise ServiceError(
            f"unknown service {name!r}; defined: {sorted(load())} (see {config.SERVICES_FILE})"
        ) from None


def names() -> list[str]:
    return list(load())


def start_order(names_: list[str] | None = None) -> list[ServiceSpec]:
    """The specs to bring up, dependencies first."""
    specs = load()
    return [specs[n] for n in order(specs, names_)]


def reset_cache() -> None:
    _cache.clear()


__all__ = [
    "AdoptSpec", "HealthProbe", "MatchSpec", "ServiceError", "ServiceSpec", "VramSpec",
    "get", "load", "names", "order", "parse", "parse_service", "replace", "reset_cache",
    "resolve_vars", "start_order",
]
