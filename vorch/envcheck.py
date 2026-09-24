"""Environment check — is this machine ready to run a job with a given profile?

A structured report (ok / warn / fail per check) the UI shows on the Setup
page and the job submit consults: **fail** blocks a submit (unless forced),
**warn** does not. Checks: database + migrations, the backends file, every
ComfyUI backend (``comfy:<name>``; + the local input/output dirs), Gemma (+ an
optional vision probe for the ``--mmproj`` mistake), ffmpeg, GPU headroom,
every registered workflow (YAML vs template, and — when *its* backend
answers — node classes and model files actually installed), the profile
itself, and the export / projects roots.

Remote backends are probed only when the selected profile uses a workflow on
them, or with ``deep=True``: probing wakes a billed GPU container.
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import psycopg
import requests

from . import backends, config, db, registry, vram
from . import services as services_mod
from .backends import ComfyBackend
from .clients import comfy, gemma

log = logging.getLogger("vorch.envcheck")

Status = Literal["ok", "warn", "fail"]
_ORDER = {"ok": 0, "warn": 1, "fail": 2}



def probe_png(width: int = 64, height: int = 64, rgb: tuple[int, int, int] = (200, 40, 40)) -> bytes:
    """A plain 8-bit RGB PNG for the vision probe.

    llama-server's image loader rejects tiny palette PNGs ("Failed to load image
    or audio file"), so the probe is generated as truecolor at a sane size.
    """
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


_PROBE_PNG = probe_png()


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail, "data": self.data}


@dataclass
class EnvReport:
    checks: list[CheckResult]
    checked_at: str
    duration_ms: int
    profile_id: str | None = None
    deep: bool = False

    @property
    def ok(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def status(self) -> Status:
        return max((c.status for c in self.checks), key=_ORDER.get, default="ok")

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status,
            "checked_at": self.checked_at,
            "duration_ms": self.duration_ms,
            "profile_id": self.profile_id,
            "deep": self.deep,
            "checks": [c.to_dict() for c in self.checks],
        }


class EnvironmentNotReady(RuntimeError):
    def __init__(self, report: EnvReport):
        names = ", ".join(f"{c.name}: {c.detail}" for c in report.failures)
        super().__init__(f"environment not ready — {names}")
        self.report = report


# --- individual checks -------------------------------------------------------------


def _check_database(store: Any | None) -> CheckResult:
    try:
        if store is not None:
            version = store.schema_version()
        else:
            with psycopg.connect(config.DATABASE_URL, connect_timeout=3) as conn:
                conn.execute("SELECT 1")
                version = max(db.applied_versions(conn), default=0)
    except (psycopg.Error, db.DatabaseUnavailable) as e:
        return CheckResult("database", "fail", f"PostgreSQL unreachable: {e}".splitlines()[0])
    latest = db.latest_version()
    if version < latest:
        return CheckResult("database", "fail", f"schema version {version} < {latest}; run `vorch db init`",
                           {"version": version, "latest": latest})
    return CheckResult("database", "ok", f"schema version {version}", {"version": version})


def _check_backends() -> tuple[CheckResult, dict[str, ComfyBackend]]:
    """The backends file loads and its auth env vars exist. Always returns a
    usable mapping (local only on failure)."""
    try:
        known = backends.load(force=True)
    except backends.BackendError as e:
        return CheckResult("backends", "fail", str(e), {"file": str(config.BACKENDS_FILE)}), {backends.LOCAL: backends.local()}
    remote = [b.name for b in known.values() if b.remote]
    detail = "local only" if not remote else "local + " + ", ".join(remote)
    if not config.BACKENDS_FILE.is_file():
        detail += f" (no {config.BACKENDS_FILE})"
    return CheckResult("backends", "ok", detail, {
        "file": str(config.BACKENDS_FILE), "backends": [b.to_dict() for b in known.values()],
    }), known


def _check_comfy(backend: ComfyBackend, *, probe: bool = True) -> CheckResult:
    """``comfy:<name>``: is this ComfyUI reachable? ``probe=False`` records a
    remote backend without contacting it (a probe wakes a billed container)."""
    name = f"comfy:{backend.name}"
    base = {"url": backend.url, "remote": backend.remote, "gpu": backend.gpu, "probed": probe}
    if not probe:
        return CheckResult(name, "ok", "not probed (remote GPU; probed when a selected profile uses it, or deep)", base)
    try:
        if backend.remote:
            stats = comfy.wait_ready(backend)  # tolerate a cold start
        else:
            stats = comfy.system_stats(backend, timeout=3)
    except (comfy.ComfyError, ValueError) as e:
        where = "remote " if backend.remote else ""
        return CheckResult(name, "fail", f"{where}ComfyUI unreachable at {backend.url}: {e}", base)
    sysinfo = stats.get("system", {})
    devices = stats.get("devices", [])
    data = {
        **base,
        "version": sysinfo.get("comfyui_version"),
        "devices": [{"name": d.get("name"), "vram_total_mb": int(d.get("vram_total", 0) / 2**20)} for d in devices],
    }
    gpu = f" on {devices[0].get('name')}" if devices and devices[0].get("name") else ""
    return CheckResult(name, "ok", f"ComfyUI {data['version'] or '?'}{gpu}", data)


def _check_comfy_dirs(local_used: bool = True) -> CheckResult:
    """The local ComfyUI input/output dirs. A hard fail only when a local
    workflow is in use; with remote-only workflows the output dir is merely
    created (downloads land there) and a missing input dir is a warning."""
    problems = []
    for label, p in (("input", config.COMFY_INPUT), ("output", config.COMFY_OUTPUT)):
        if not p.is_dir():
            if label == "output" and not local_used:
                try:
                    p.mkdir(parents=True, exist_ok=True)
                    continue
                except OSError as e:
                    problems.append(f"output dir cannot be created: {p} ({e})")
                    continue
            problems.append(f"{label} dir missing: {p}")
        elif not os.access(p, os.W_OK):
            problems.append(f"{label} dir not writable: {p}")
    if problems:
        return CheckResult("comfy_dirs", "fail" if local_used else "warn", "; ".join(problems))
    return CheckResult("comfy_dirs", "ok", f"{config.COMFY_INPUT} / {config.COMFY_OUTPUT}")


def _check_gemma() -> CheckResult:
    try:
        r = requests.get(f"{config.GEMMA_URL}/v1/models", timeout=3)
        r.raise_for_status()
        models = [m.get("id") for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError) as e:
        return CheckResult("gemma", "fail", f"Gemma (llama-server) unreachable at {config.GEMMA_URL}: {e}")
    return CheckResult("gemma", "ok", ", ".join(str(m) for m in models) or "llama-server up",
                       {"url": config.GEMMA_URL, "models": models})


def _check_gemma_vision() -> CheckResult:
    """Deep: send a tiny image; a text-only server (no --mmproj) cannot see it."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_PROBE_PNG)
        probe = f.name
    try:
        raw = gemma.complete_vision(
            "You answer with one word.",
            "Is an image attached to this message? Answer YES or NO.",
            [probe], max_tokens=5, timeout=120,
        )
    except Exception as e:  # noqa: BLE001 - any transport/parse failure is a warn here
        return CheckResult("gemma_vision", "warn", f"vision probe failed: {e}")
    finally:
        try:
            os.unlink(probe)
        except OSError:
            pass
    text = (raw or "").strip().upper()
    if "YES" in text:
        return CheckResult("gemma_vision", "ok", "vision works (mmproj loaded)")
    return CheckResult("gemma_vision", "warn",
                       f"model does not see images (reply: {raw!r}); start llama-server with --mmproj")


def _check_tool(name: str) -> CheckResult:
    path = shutil.which(name)
    if path:
        return CheckResult(name, "ok", path)
    return CheckResult(name, "warn", f"{name} not on PATH (continuity, export probing and video eval need it)")


def _check_gpu() -> CheckResult:
    """Free VRAM, and — the part that used to be missing — *who* has the rest.

    This row used to say "only N MB free; ComfyUI may still hold models" no matter what was
    actually holding the card, which was actively misleading once a third model existed: it read as
    "ComfyUI" while a narration server nobody had stopped held 11 GB and ComfyUI held 222 MB.
    """
    info = vram.attribute()
    if info is None:
        return CheckResult("gpu", "warn", "nvidia-smi unavailable; VRAM cannot be monitored")
    data = {**{k: info[k] for k in ("used", "free", "total")},
            "unattributed_mb": info["unattributed_mb"],
            "holders": [h.to_dict() for h in info["holders"]]}
    if info["free"] >= config.VRAM_GUARD_MB:
        return CheckResult("gpu", "ok", f"{info['free']} / {info['total']} MB free", data)
    who = vram.describe_holders(info["holders"])
    reclaimable = sum(h.used_mb for h in info["holders"] if h.evictable)
    tail = (f"{reclaimable} MB of that can be reclaimed before a render"
            if reclaimable else "none of it can be reclaimed automatically")
    return CheckResult("gpu", "warn",
                       f"only {info['free']} MB free (< {config.VRAM_GUARD_MB} guard); "
                       f"held by {who}; {tail}", data)


def _object_info(backend: ComfyBackend, reachable: bool, force: bool) -> dict | None:
    if not reachable:
        return None
    try:
        return comfy.object_info(refresh=force, backend=backend)
    except Exception as e:  # noqa: BLE001
        log.warning("envcheck: %s /object_info failed: %s", backend.name, e)
        return None


def _check_workflows(
    reg: registry.Registry, used: set[str], infos: dict[str, dict | None],
    known: dict[str, ComfyBackend] | None = None,
) -> list[CheckResult]:
    """One row per workflow, validated against **its own** backend's object_info."""
    out: list[CheckResult] = []
    for spec in reg.list():
        object_info = infos.get(spec.backend)
        problems = reg.validate(spec, object_info=object_info, backends=known)
        in_use = spec.name in used
        data = {"used": in_use, "backend": spec.backend}
        tag = "" if spec.backend == backends.LOCAL else f" @{spec.backend}"
        if not problems:
            detail = spec.template.name + tag + ("" if object_info is not None else " (offline check)")
            out.append(CheckResult(f"workflow:{spec.name}", "ok", detail, data))
            continue
        out.append(CheckResult(
            f"workflow:{spec.name}", "fail" if in_use else "warn", "; ".join(problems),
            {**data, "problems": problems},
        ))
    for name, err in reg.errors.items():
        out.append(CheckResult(f"workflow:{name}", "warn", f"invalid user workflow: {err.splitlines()[0]}"))
    return out


def _check_profile(profile: Any | None, reg: registry.Registry) -> CheckResult:
    if profile is None:
        return CheckResult("profile", "ok", "no profile selected")
    missing = [w for w in profile.workflows_used() if w not in reg]
    if missing:
        return CheckResult("profile", "fail", f"profile {profile.id!r} uses unregistered workflow(s): {missing}")
    try:
        from .genparams import GenParams

        warnings = GenParams.from_profile(profile).validate()
    except ValueError as e:
        return CheckResult("profile", "fail", f"profile {profile.id!r} invalid: {e}")
    if warnings:
        return CheckResult("profile", "warn", "; ".join(warnings), {"id": profile.id})
    return CheckResult("profile", "ok", f"{profile.id} ({profile.kind}, {profile.aspect})", {"id": profile.id})


def _check_dir(name: str, path: Path) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise PermissionError(path)
    except OSError as e:
        return CheckResult(name, "warn", f"{path} not writable: {e}")
    return CheckResult(name, "ok", str(path))


def _check_services() -> list[CheckResult]:
    """One row per manifest service that nothing else already covers.

    ``database``, ``comfy:local`` and ``gemma`` have their own checks above, so they are skipped
    rather than duplicated, and ``api``/``ui`` are skipped because this report is *served by* the
    API to the UI — by the time you can read the answer, both are demonstrably up. What is left is
    the narration server, which until now was invisible to diagnostics entirely (``grep -c tts
    envcheck.py`` was 0) despite being the largest single consumer of the card.

    Not running is **ok** for a service started on demand: it is the correct state, not a fault.
    What is a fault is an orphan holding VRAM, or a server demonstrably running the wrong instance.
    """
    from . import supervisor
    from .services import ServiceError

    covered = {"postgres", "comfy", "gemma", "api", "ui"}
    out: list[CheckResult] = []
    try:
        wanted = [n for n in services_mod.names() if n not in covered]
        # Resolve only what we will report: states() would expand dependencies and probe ComfyUI
        # and PostgreSQL a second time.
        table = [supervisor.resolve(n) for n in wanted]
    except ServiceError as e:
        return [CheckResult("services", "fail", f"the service manifest is unusable: {e}")]
    for s in table:
        name = f"service:{s.name}"
        data = s.to_dict()
        if s.orphan:
            held = f", holding {s.vram_mb} MB" if s.vram_mb else ""
            out.append(CheckResult(name, "fail",
                                   f"orphan from a vorch that is gone (pid {s.pid}{held}) — "
                                   f"release it with: vorch down {s.name}", data))
        elif s.identity == "mismatch":
            out.append(CheckResult(name, "fail", f"wrong instance: {s.identity_detail}", data))
        elif not s.running and s.spec.lifetime in ("warm", "phase"):
            # Started on demand: absent is correct, and must not read as a problem.
            out.append(CheckResult(name, "ok", s.detail, data))
        elif not s.running:
            out.append(CheckResult(name, "warn", s.detail, data))
        else:
            out.append(CheckResult(name, s.status, s.detail, data))
    return out


# --- report ----------------------------------------------------------------------------

_cache: dict[tuple, tuple[float, EnvReport]] = {}


def run_checks(
    *,
    profile: Any | None = None,
    project: Any | None = None,
    store: Any | None = None,
    deep: bool = False,
    force: bool = False,
    ttl: float = 30.0,
) -> EnvReport:
    """Run every check; cached per (profile, deep) for ``ttl`` seconds unless ``force``."""
    from . import projects as projects_mod

    key = (getattr(profile, "id", None), deep)
    now = time.monotonic()
    if not force and key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]

    t0 = time.monotonic()
    checks: list[CheckResult] = [_check_database(store)]
    backends_res, known = _check_backends()
    checks.append(backends_res)

    reg = registry.load_registry(force=force)
    used = set(profile.workflows_used()) if profile is not None else set()
    used_backends = {reg.get(w).backend for w in used if w in reg}
    # No profile -> assume the local ComfyUI matters (the default install).
    local_used = profile is None or not used_backends or any(
        known[b].is_local for b in used_backends if b in known
    )

    comfy_res: dict[str, CheckResult] = {}
    for b in known.values():
        probe = b.is_local or deep or b.name in used_backends
        comfy_res[b.name] = _check_comfy(b, probe=probe)
    checks += list(comfy_res.values())
    checks.append(_check_comfy_dirs(local_used))
    gemma_res = _check_gemma()
    checks.append(gemma_res)
    if deep and gemma_res.status == "ok":
        checks.append(_check_gemma_vision())
    checks += [_check_tool("ffmpeg"), _check_tool("ffprobe"), _check_gpu()]
    checks += _check_services()

    infos = {
        name: _object_info(known[name], res.status == "ok" and bool(res.data.get("probed")), force)
        for name, res in comfy_res.items()
    }
    checks += _check_workflows(reg, used, infos, known)
    checks.append(_check_profile(profile, reg))
    checks.append(_check_dir("export_root", projects_mod.export_root(project)))
    checks.append(_check_dir("projects_root", config.PROJECTS_ROOT))

    report = EnvReport(
        checks=checks,
        checked_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=int((time.monotonic() - t0) * 1000),
        profile_id=getattr(profile, "id", None),
        deep=deep,
    )
    _cache[key] = (now, report)
    return report


def gate(profile: Any | None = None, project: Any | None = None, *, store: Any | None = None) -> EnvReport:
    """Raise :class:`EnvironmentNotReady` on any failing check; return the report otherwise."""
    report = run_checks(profile=profile, project=project, store=store)
    if not report.ok:
        raise EnvironmentNotReady(report)
    return report


def clear_cache() -> None:
    _cache.clear()
