"""Starting, stopping and inspecting the services vorch depends on.

``services.py`` declares the world; this module looks at it and acts on it. The split matters: the
declaration stays pure and testable, and every subprocess, signal and HTTP probe lives here.

**Hand-starting is a first-class mode, not a legacy path.** Everything here is built around the
assumption that a person may have launched any of these in a terminal — to audition a voice, to
debug, or simply out of habit. So: ``up`` adopts whatever already answers and restarts nothing;
``down`` refuses to stop what it did not start unless forced; a service can be set to
``manage: adopt`` permanently and vorch reduces to a monitor; and no process is ever signalled
unless the manifest can name it. The failure mode is *refuse*, never *guess*.

Three questions this module exists to answer, none of which vorch could answer before:

**Is it running, and is it ours?** A pidfile in ``~/vorch/run`` records the pid, the argv and the
``owner_pid`` — the vorch process that spawned it. That is what separates our own child from an
orphan left by a SIGKILLed server from a server someone started by hand (no pidfile at all).
SIGKILL cannot be caught, so the pidfile is the only honest answer to "what did the last crash
leave behind".

**Is it the *right* instance?** For OmniVoice the cloned voice arrives only as an environment
variable and ``/health`` never reports it back, so a server started from ``serve_bg.sh``'s defaults
reads every line in a different announcer's voice and nothing notices. ``/proc/<pid>/environ``
answers it, and ``adopt.verify: env`` is what makes the mismatch visible.

**What did it say when it failed?** Each service's output goes to ``~/vorch/logs/<name>.log`` — one
appended fd, never ``DEVNULL`` and never a pipe that is read only on failure. It survives a SIGKILL
of vorch, and a failed start raises with the tail of it.

This stage is read-only: ``states``, ``resolve``, ``tail``, ``orphans``. Starting and stopping, and
the VRAM arbitration that goes with them, come next.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import requests

from . import config, services, vram
from .services import ServiceError, ServiceSpec

log = logging.getLogger("vorch.supervisor")

#: Overridable so tests can point the process scan at a fixture tree.
PROC = Path("/proc")

Identity = Literal["ours", "verified", "unverified", "mismatch", "n/a"]


class SupervisorError(RuntimeError):
    """A service could not be started, stopped, or safely identified."""


# --- pidfiles -------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """What we wrote down when we started a service. Absent = we did not start it."""

    name: str
    pid: int
    started_at: float
    argv: tuple[str, ...] = ()
    cwd: str = ""
    lifetime: str = "long"
    owner_pid: int = 0
    last_used_at: float = 0.0
    identity: dict[str, str] = field(default_factory=dict)
    log: str = ""
    health_url: str = ""

    @property
    def owner_alive(self) -> bool:
        """Is the vorch process that started this still running?"""
        return self.owner_pid > 0 and _pid_alive(self.owner_pid)

    def to_dict(self) -> dict:
        return {"name": self.name, "pid": self.pid, "started_at": self.started_at,
                "argv": list(self.argv), "cwd": self.cwd, "lifetime": self.lifetime,
                "owner_pid": self.owner_pid, "last_used_at": self.last_used_at,
                "identity": dict(self.identity), "log": self.log, "health_url": self.health_url}


def read_record(name: str) -> Record | None:
    """The pidfile for ``name``, or None when there is none or it is unreadable."""
    path = config.RUN_DIR / f"{name}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return Record(
            name=str(raw.get("name") or name), pid=int(raw["pid"]),
            started_at=float(raw.get("started_at") or 0.0),
            argv=tuple(str(a) for a in raw.get("argv") or ()),
            cwd=str(raw.get("cwd") or ""), lifetime=str(raw.get("lifetime") or "long"),
            owner_pid=int(raw.get("owner_pid") or 0),
            last_used_at=float(raw.get("last_used_at") or 0.0),
            identity={str(k): str(v) for k, v in (raw.get("identity") or {}).items()},
            log=str(raw.get("log") or ""), health_url=str(raw.get("health_url") or ""),
        )
    except (KeyError, TypeError, ValueError):
        log.warning("supervisor: %s is not a readable pidfile; ignoring it", path)
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True              # exists, owned by someone else
    except OSError:
        return False
    return True


# --- looking at processes -------------------------------------------------------------


def _cmdline(pid: int) -> str:
    try:
        return (PROC / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace").strip()
    except OSError:
        return ""


def _cwd(pid: int) -> str | None:
    try:
        return os.readlink(PROC / str(pid) / "cwd")
    except OSError:
        return None


def _environ(pid: int) -> dict[str, str] | None:
    """``/proc/<pid>/environ`` as a mapping, or None when it cannot be read.

    Unreadable is a normal outcome — a different user, a hardened kernel — and must never raise:
    it degrades to ``identity="unverified"`` and the manifest's ``on_unverified`` policy.
    """
    try:
        raw = (PROC / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    out: dict[str, str] = {}
    for chunk in raw.split(b"\0"):
        if not chunk:
            continue
        key, _, value = chunk.decode("utf-8", errors="replace").partition("=")
        out[key] = value
    return out


def _matches(spec: ServiceSpec, pid: int) -> bool:
    try:
        m = spec.matcher()
    except ServiceError:
        return False
    if not m.cmdline_contains:
        return False
    cmdline = _cmdline(pid)
    if not cmdline or not all(n in cmdline for n in m.cmdline_contains):
        return False
    if m.cwd:
        cwd = _cwd(pid)
        if cwd and str(Path(m.cwd)) != str(Path(cwd)):
            return False
    return True


def find_pids(spec: ServiceSpec) -> list[int]:
    """Every pid matching this service's ``match`` rules, lowest first."""
    found: list[int] = []
    try:
        entries = sorted((int(d.name) for d in PROC.iterdir() if d.name.isdigit()))
    except OSError:
        return found
    for pid in entries:
        if _matches(spec, pid):
            found.append(pid)
    return found


# --- health ---------------------------------------------------------------------------


def probe(spec: ServiceSpec, *, timeout: float | None = None) -> tuple[bool | None, dict, str]:
    """``(healthy, payload, detail)``. ``healthy`` is None when the manifest gives no probe."""
    kind = spec.health.probe
    if kind == "none":
        return None, {}, "no health probe declared"
    t = spec.health.timeout if timeout is None else timeout
    if kind == "http":
        url = spec.health_url()
        try:
            r = requests.get(url, timeout=t)
            r.raise_for_status()
        except requests.RequestException as e:
            return False, {}, f"{url}: {_short(e)}"
        try:
            payload = r.json()
            payload = payload if isinstance(payload, dict) else {"value": payload}
        except ValueError:
            payload = {}
        for key, expected in spec.health.expect_json.items():
            if str(payload.get(key)) != str(expected):
                return False, payload, f"{url}: {key}={payload.get(key)!r}, expected {expected!r}"
        return True, payload, "answers"
    if kind == "dsn":
        dsn = spec.health_target()
        try:
            import psycopg

            with psycopg.connect(dsn, connect_timeout=int(t) or 1) as conn:
                conn.execute("SELECT 1")
        except Exception as e:                      # psycopg raises a family of errors
            return False, {}, f"{_mask(dsn)}: {_short(e)}"
        return True, {}, "answers"
    return None, {}, f"unknown probe {kind!r}"


def _short(e: Exception) -> str:
    text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return text[:160]


def _mask(dsn: str) -> str:
    return dsn.split("@")[-1] if "@" in dsn else dsn


# --- identity -------------------------------------------------------------------------


def check_identity(spec: ServiceSpec, pid: int, record: Record | None) -> tuple[Identity, str]:
    """Is the process on that pid the instance the manifest describes?

    ``ours`` we started it · ``verified`` it is demonstrably right · ``mismatch`` it is
    demonstrably wrong · ``unverified`` we could not tell · ``n/a`` nothing to check.
    """
    if record is not None and record.pid == pid and record.owner_alive:
        return "ours", "started by this vorch"
    verify = spec.adopt.verify
    if verify == "port":
        return "n/a", "adopted on the strength of its port alone (adopt.verify: port)"
    if verify == "argv":
        want = spec.argv_expanded()
        have = _cmdline(pid)
        if not have:
            return "unverified", "cannot read /proc/<pid>/cmdline"
        missing = [a for a in want if a and a not in have]
        if missing:
            return "mismatch", ("running command differs from the manifest — missing "
                                + ", ".join(repr(m) for m in missing[:3]))
        return "verified", "command line matches the manifest"
    # verify == "env": the case that matters, because a cloned voice lives only here.
    try:
        want = spec.identity()
    except ServiceError as e:
        return "unverified", f"cannot resolve the expected identity ({e})"
    if not want:
        return "n/a", "no identity env declared"
    env = _environ(pid)
    if env is None:
        return "unverified", "cannot read /proc/<pid>/environ (different user, or a hardened kernel)"
    import hashlib

    wrong: list[str] = []
    for key, expected in want.items():
        actual = env.get(key)
        if actual is None:
            wrong.append(f"{key} is not set on the running server")
            continue
        got = (actual if len(actual) <= 80
               else "sha1:" + hashlib.sha1(actual.encode("utf-8")).hexdigest()[:12])
        if got != expected:
            wrong.append(f"{key} is {_trim(actual)}, expected {_trim(expected)}")
    if wrong:
        return "mismatch", "; ".join(wrong)
    return "verified", ", ".join(f"{k}={_trim(v)}" for k, v in want.items())


def _trim(value: str, limit: int = 48) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


# --- state ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceState:
    """Everything we can say about one service right now."""

    spec: ServiceSpec
    running: bool = False
    pid: int | None = None
    pids: tuple[int, ...] = ()
    healthy: bool | None = None
    health: dict = field(default_factory=dict)
    vram_mb: int | None = None
    adopted: bool = False          # running, but not started by this vorch
    orphan: bool = False           # our pidfile, but the vorch that wrote it is gone
    identity: Identity = "n/a"
    identity_detail: str = ""
    record: Record | None = None
    uptime_sec: float | None = None
    detail: str = ""

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def status(self) -> str:
        """``ok`` · ``warn`` · ``fail`` · ``down`` — the same vocabulary as the env report."""
        if self.identity == "mismatch":
            return "fail"
        if self.orphan:
            return "fail"
        if not self.running:
            # Not running is the *correct* state for something started on demand.
            return "down"
        if self.healthy is False:
            return "fail"
        if self.identity == "unverified" and self.spec.adopt.on_unverified == "refuse":
            return "fail"
        if self.identity == "unverified" or len(self.pids) > 1:
            return "warn"
        return "ok"

    @property
    def idle_sec(self) -> float | None:
        """Seconds since the last recorded use, for a ``warm`` service we started."""
        if self.record is None or not self.record.last_used_at:
            return None
        return max(0.0, time.time() - self.record.last_used_at)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.spec.description,
            "manage": self.spec.manage, "lifetime": self.spec.lifetime,
            "status": self.status, "running": self.running, "pid": self.pid,
            "pids": list(self.pids), "healthy": self.healthy, "health": self.health,
            "vram_mb": self.vram_mb, "adopted": self.adopted, "orphan": self.orphan,
            "identity": self.identity, "identity_detail": self.identity_detail,
            "uptime_sec": self.uptime_sec, "idle_sec": self.idle_sec,
            "detail": self.detail, "hint": self.spec.hint,
            "log": str(self.spec.log_path),
            "vram": self.spec.vram.to_dict(),
        }


def resolve(spec: ServiceSpec | str, *, do_probe: bool = True,
            holders: list[vram.Holder] | None = None) -> ServiceState:
    """What is true about one service: pidfile first, then /proc, then the health probe.

    Ordered by how much it tells us. A pidfile names the pid *and* says we own it; a /proc match
    finds a server someone started by hand; a health probe is the last resort, and answers only
    "something is listening".
    """
    spec = services.get(spec) if isinstance(spec, str) else spec
    record = read_record(spec.name)
    pids = find_pids(spec)

    # A stale pidfile is worse than none: it names a pid that may have been reused.
    if record is not None and (not _pid_alive(record.pid) or not _matches(spec, record.pid)):
        log.info("supervisor: %s pidfile names pid %d, which is gone or is something else now",
                 spec.name, record.pid)
        record = None

    pid = record.pid if record is not None else (pids[0] if pids else None)
    if record is not None and record.pid not in pids:
        pids = [record.pid, *pids]

    healthy, payload, health_detail = (None, {}, "")
    if do_probe:
        healthy, payload, health_detail = probe(spec)

    running = bool(pids) or healthy is True
    identity: Identity = "n/a"
    identity_detail = ""
    if running and pid is not None:
        identity, identity_detail = check_identity(spec, pid, record)

    orphan = bool(record is not None and not record.owner_alive and _pid_alive(record.pid))
    adopted = bool(running and (record is None or orphan))

    used = None
    if pid is not None:
        table = vram.holders() if holders is None else holders
        used = next((h.used_mb for h in table if h.pid == pid), None)

    uptime = None
    if pid is not None:
        started = vram._proc_started_at(pid)
        uptime = None if started is None else max(0.0, time.time() - started)
    elif record is not None:
        uptime = max(0.0, time.time() - record.started_at)

    state = ServiceState(
        spec=spec, running=running, pid=pid, pids=tuple(pids), healthy=healthy, health=payload,
        vram_mb=used, adopted=adopted, orphan=orphan, identity=identity,
        identity_detail=identity_detail, record=record, uptime_sec=uptime,
    )
    return ServiceState(**{**state.__dict__, "detail": _describe(state, health_detail)})


def _describe(s: ServiceState, health_detail: str) -> str:
    spec = s.spec
    if not s.running:
        if spec.manage == "never":
            return f"not running — {spec.hint or 'start it yourself'}"
        if spec.lifetime in ("warm", "phase"):
            return f"not running, which is correct: started on demand ({spec.lifetime})"
        return "not running" + (f" — {spec.hint}" if spec.hint else "")
    bits: list[str] = []
    if s.pid is not None:
        bits.append(f"pid {s.pid}")
    if s.uptime_sec is not None:
        bits.append(f"up {vram._duration(s.uptime_sec)}")
    if s.vram_mb:
        bits.append(f"{s.vram_mb} MB VRAM")
    if s.orphan:
        bits.append("ORPHAN — the vorch that started it is gone")
    elif s.adopted:
        bits.append("started outside vorch")
    if s.identity == "mismatch":
        bits.append(f"WRONG INSTANCE: {s.identity_detail}")
    elif s.identity == "verified":
        bits.append(f"verified ({s.identity_detail})")
    elif s.identity == "unverified":
        bits.append(f"unverified: {s.identity_detail}")
    if s.healthy is False:
        bits.append(f"not answering: {health_detail}")
    if len(s.pids) > 1:
        bits.append(f"{len(s.pids)} processes match: {', '.join(str(p) for p in s.pids)}")
    return "; ".join(bits)


def states(names: list[str] | None = None, *, do_probe: bool = True) -> list[ServiceState]:
    """Every service in dependency order, so the output reads the way a bring-up would run."""
    specs = services.load()
    order = services.order(specs, names)
    table = vram.holders()
    return [resolve(specs[n], do_probe=do_probe, holders=table) for n in order]


def orphans() -> list[ServiceState]:
    """Services still running from a vorch that is gone — the 11 GB a SIGKILL leaves behind."""
    return [s for s in states(do_probe=False) if s.orphan]


# --- logs -----------------------------------------------------------------------------


def tail(name: str, lines: int = 200) -> str:
    """The last ``lines`` of a service's log, or a sentence explaining why there are none."""
    spec = services.get(name)
    path = spec.log_path
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = min(size, max(4096, lines * 200))
            f.seek(size - block)
            text = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return (f"no log at {path} — vorch has not started {name} "
                f"(a service you launched yourself logs to its own terminal)")
    except OSError as e:
        return f"cannot read {path}: {e}"
    return "\n".join(text.splitlines()[-lines:])


def summary(table: list[ServiceState] | None = None) -> dict:
    """The whole picture, for ``vorch status`` and ``GET /api/services``."""
    table = states() if table is None else table
    gpu = vram.attribute()
    return {
        "services_file": str(config.SERVICES_FILE),
        "run_dir": str(config.RUN_DIR),
        "log_dir": str(config.LOG_DIR),
        "services": [s.to_dict() for s in table],
        "gpu": None if gpu is None else {
            **{k: gpu[k] for k in ("used", "free", "total")},
            "unattributed_mb": gpu["unattributed_mb"],
            "holders": [h.to_dict() for h in gpu["holders"]],
        },
        "orphans": [s.name for s in table if s.orphan],
        "problems": [f"{s.name}: {s.detail}" for s in table if s.status == "fail"],
    }


__all__ = [
    "Identity", "Record", "ServiceState", "SupervisorError",
    "check_identity", "find_pids", "orphans", "probe", "read_record", "resolve",
    "states", "summary", "tail",
]
