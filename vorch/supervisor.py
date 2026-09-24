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

import atexit
import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    if record is not None and record.pid == pid:
        return "ours", ("started by this vorch" if record.owner_alive
                        else "started by vorch (that process has since exited)")
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
            # Not a mystery: the launcher was run without the variable, so it fell back to its own
            # built-in default — which for OmniVoice is a different announcer's voice entirely.
            wrong.append(f"{key} is not set, so it is using the launcher's own default "
                         f"instead of {_trim(expected)}")
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
        if self.orphan and self.spec.lifetime == "phase":
            # It should have been stopped when its phase ended, so it has genuinely leaked.
            return "fail"
        if self.orphan and self.vram_mb:
            # Ours, but nothing is watching it any more: not broken, yet holding the card.
            return "warn"
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

    # "Ours" is decided by the pidfile, not by whether the process that wrote it is still alive:
    # `vorch up` exits the moment it finishes, so a long-lived or warm service it started outlives
    # it by design. `orphan` records that fact; it does not by itself mean anything is wrong.
    orphan = bool(record is not None and not record.owner_alive and _pid_alive(record.pid))
    adopted = bool(running and record is None)

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
    if s.orphan and spec.lifetime == "phase":
        bits.append("LEAKED — a phase-scoped service whose vorch is gone")
    elif s.orphan:
        bits.append("started by vorch, whose process has since exited"
                    + ("; nothing is watching it" if s.vram_mb else ""))
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
    "check_identity", "down", "ensure_headroom", "find_pids", "install_exit_handlers",
    "orphans", "phase", "probe", "read_record", "reap_idle", "resolve", "rotate_log",
    "start", "states", "stop", "summary", "tail", "touch_record", "up", "write_record",
]


# --- writing pidfiles -----------------------------------------------------------------


def write_record(spec: ServiceSpec, pid: int, *, started_at: float | None = None,
                 last_used_at: float | None = None) -> Record:
    """Record that we started this, so a later vorch can tell an orphan from a stranger."""
    try:
        identity = spec.identity()
    except ServiceError:
        identity = {}
    rec = Record(
        name=spec.name, pid=pid, started_at=started_at or time.time(),
        argv=tuple(spec.argv_expanded()), cwd=str(spec.cwd_path), lifetime=spec.lifetime,
        owner_pid=os.getpid(), last_used_at=last_used_at or time.time(),
        identity=identity, log=str(spec.log_path), health_url=spec.health_url(),
    )
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RUN_DIR / f"{spec.name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, path)                 # atomic: a reader never sees half a file
    return rec


def touch_record(name: str) -> None:
    """Mark a service as used just now — what the idle reaper measures against."""
    rec = read_record(name)
    if rec is None:
        return
    path = config.RUN_DIR / f"{name}.json"
    try:
        data = rec.to_dict() | {"last_used_at": time.time()}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        log.debug("supervisor: could not touch %s (%s)", path, e)


def clear_record(name: str) -> None:
    (config.RUN_DIR / f"{name}.json").unlink(missing_ok=True)


# --- log files ------------------------------------------------------------------------


def rotate_log(path: Path, *, max_bytes: int | None = None, backups: int | None = None) -> None:
    """Shift the log aside if it has grown too big.

    Done at start rather than continuously: a running child holds the file descriptor, and nothing
    outside that process can rotate an fd it does not own. The cost is that a long-lived, chatty
    service is unbounded until it is restarted — documented rather than pretended away.
    """
    limit = config.LOG_MAX_BYTES if max_bytes is None else max_bytes
    keep = config.LOG_BACKUPS if backups is None else backups
    try:
        if not path.is_file() or path.stat().st_size < limit:
            return
        for i in range(keep - 1, 0, -1):
            older = path.with_name(f"{path.name}.{i}")
            if older.is_file():
                older.replace(path.with_name(f"{path.name}.{i + 1}"))
        path.replace(path.with_name(f"{path.name}.1"))
    except OSError as e:
        log.warning("supervisor: could not rotate %s (%s)", path, e)


def _open_log(spec: ServiceSpec, argv: list[str], env: dict[str, str]):
    """Append-mode fd for the child, with a header saying exactly what is being started.

    The header is what makes a past run answerable: which command, from which directory, and — for
    the narration server — which cloned voice. Weeks later that is the only record.
    """
    path = spec.log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    rotate_log(path)
    fh = open(path, "ab", buffering=0)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ident = spec.identity() if spec.env else {}
    header = [f"\n=== vorch start {spec.name} {stamp} (owner pid {os.getpid()}) ===",
              f"    cwd:  {spec.cwd_path}",
              f"    argv: {' '.join(argv)}"]
    for key in sorted(env):
        shown = ident.get(key, env[key])
        header.append(f"    env:  {key}={shown if len(shown) <= 120 else shown[:117] + '...'}")
    fh.write(("\n".join(header) + "\n").encode("utf-8"))
    return fh


# --- starting -------------------------------------------------------------------------


def start(name: str | ServiceSpec, *, wait: bool = True, timeout: float | None = None,
          free_vram: bool = True) -> ServiceState:
    """Start a service and wait until its health probe answers.

    Adopts instead of starting when something already answers — never restarts, never "corrects"
    a running instance, because it may well be one a person started deliberately.
    """
    spec = services.get(name) if isinstance(name, str) else name
    if spec.manage != "full":
        raise SupervisorError(
            f"{spec.name} is not vorch's to start (manage: {spec.manage})"
            + (f" — {spec.hint}" if spec.hint else ""))

    state = resolve(spec)
    if state.running:
        if state.identity == "mismatch":
            raise SupervisorError(
                f"{spec.name} is already running, but it is the wrong instance: "
                f"{state.identity_detail}. Stop it first (vorch down {spec.name} --force).")
        if state.identity == "unverified" and spec.adopt.on_unverified == "refuse":
            raise SupervisorError(
                f"{spec.name} is already running and its identity cannot be verified "
                f"({state.identity_detail}); the manifest says to refuse rather than guess.")
        log.info("supervisor: %s is already running (%s); adopting it", spec.name, state.detail)
        return state

    if free_vram and spec.vram.needs_mb:
        ensure_headroom(spec.vram.needs_mb, exclude=[spec.name])

    argv = spec.argv_expanded()
    env = spec.env_resolved()          # ServiceError here means a missing @file: — a real problem
    fh = _open_log(spec, argv, env)
    log.info("supervisor: starting %s — %s (log: %s)", spec.name, " ".join(argv), spec.log_path)
    try:
        proc = subprocess.Popen(
            argv, cwd=str(spec.cwd_path), env={**os.environ, **env},
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            # Its own session, so a Ctrl-C in the terminal that ran `vorch up` does not scatter
            # into the services, and so we can later signal the whole group safely.
            start_new_session=True, close_fds=True,
        )
    except OSError as e:
        fh.close()
        raise SupervisorError(f"cannot start {spec.name}: {e}") from None
    finally:
        fh.close()                     # the child holds its own dup of the fd

    _OURS[spec.name] = proc
    record = write_record(spec, proc.pid)
    if not wait:
        return resolve(spec)

    deadline = time.monotonic() + (spec.ready_timeout if timeout is None else timeout)
    while True:
        if proc.poll() is not None:
            clear_record(spec.name)
            _OURS.pop(spec.name, None)
            raise SupervisorError(
                f"{spec.name} exited while starting (code {proc.returncode}). "
                f"Last lines of {spec.log_path}:\n{_indent(tail(spec.name, 20))}")
        healthy, _, detail = probe(spec)
        if healthy is True:
            break
        if healthy is None:                 # nothing to probe: being alive is all we can ask
            time.sleep(1.0)
            break
        if time.monotonic() > deadline:
            raise SupervisorError(
                f"{spec.name} did not answer within {spec.ready_timeout:g}s ({detail}). "
                f"It is still running as pid {proc.pid}; last lines of {spec.log_path}:\n"
                f"{_indent(tail(spec.name, 20))}")
        time.sleep(2.0)

    state = resolve(spec)
    log.info("supervisor: %s ready — %s", spec.name, state.detail)
    if state.identity == "mismatch":
        log.error("supervisor: %s started but reports the wrong identity: %s",
                  spec.name, state.identity_detail)
    touch_record(spec.name)
    return state


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


#: Handles for services this process started, so we can reap them on exit.
_OURS: dict[str, subprocess.Popen] = {}


# --- stopping -------------------------------------------------------------------------


def stop(name: str | ServiceSpec, *, force: bool = False, timeout: float = 30.0) -> None:
    """Stop a service. Refuses to touch one vorch did not start, unless ``force``.

    Two safety rails that matter more than convenience:

    * a pid is only ever signalled when a pidfile names it or the manifest's ``match`` recognises
      it — vorch must never kill a process it cannot name;
    * the *process group* is signalled only for our own children, whose session we created. An
      adopted process may share a group with a person's shell, so it gets a signal to the pid
      alone.
    """
    spec = services.get(name) if isinstance(name, str) else name
    state = resolve(spec, do_probe=False)
    if not state.running or state.pid is None:
        log.info("supervisor: %s is not running", spec.name)
        clear_record(spec.name)
        return
    pid = state.pid
    # A pidfile is what makes it ours, whether or not the vorch that wrote it is still alive —
    # `vorch up` exits immediately, so almost everything it starts outlives it.
    ours = state.record is not None

    if not ours and not force:
        who = "an orphan from an earlier vorch" if state.orphan else "started outside vorch"
        raise SupervisorError(
            f"{spec.name} (pid {pid}) was {who}; vorch will not stop what it did not start. "
            f"Use --force if you mean it.")
    if state.record is None and not _matches(spec, pid):
        raise SupervisorError(
            f"refusing to signal pid {pid}: nothing identifies it as {spec.name}")

    proc = _OURS.get(spec.name)
    use_group = proc is not None and proc.pid == pid
    log.info("supervisor: stopping %s (pid %d%s)%s", spec.name, pid,
             ", process group" if use_group else "",
             " — started outside vorch" if not ours else "")
    freed_before = vram.gpu_free_mb() if spec.vram.holds_mb else None
    _signal(pid, signal.SIGTERM, group=use_group)
    if not _wait_gone(pid, proc, timeout):
        log.warning("supervisor: %s did not exit in %.0fs; killing it", spec.name, timeout)
        _signal(pid, signal.SIGKILL, group=use_group)
        _wait_gone(pid, proc, 5.0)
    _OURS.pop(spec.name, None)
    clear_record(spec.name)
    # Report what was actually freed, not what the manifest predicts: holds_mb is a worst case, and
    # printing it as though it were a measurement was simply untrue (it claimed 11268 MB for a
    # server that held 5726). Measured after the driver has let go, so it can legitimately be 0.
    freed = None
    if spec.vram.holds_mb and (before_free := freed_before) is not None:
        _wait_vram_released(pid, timeout=config.VRAM_RELEASE_TIMEOUT)
        if (after_free := vram.gpu_free_mb()) is not None:
            freed = max(0, after_free - before_free)
    log.info("supervisor: %s stopped%s", spec.name,
             f", {freed} MB released" if freed else "")


def _is_zombie(pid: int) -> bool:
    """Dead but not yet reaped. ``kill(pid, 0)`` still succeeds on one, so it must be asked for."""
    try:
        raw = (PROC / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
        return raw[raw.rindex(")") + 1:].split()[0] == "Z"
    except (OSError, ValueError, IndexError):
        return False


def _wait_gone(pid: int, proc: subprocess.Popen | None, timeout: float) -> bool:
    """Wait for a process to actually be gone. True if it went, False on timeout.

    Our own child must be *reaped*, not merely signalled: a SIGTERMed child sits as a zombie until
    its parent collects it, and ``os.kill(pid, 0)`` happily succeeds on a zombie — so polling
    liveness alone waits out the whole timeout on a process that died immediately. ``Popen.poll``
    reaps. For a process we do not own, its real parent will reap it, so a zombie counts as gone.
    """
    deadline = time.monotonic() + timeout
    while True:
        if proc is not None:
            if proc.poll() is not None:
                return True
        elif not _pid_alive(pid) or _is_zombie(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _signal(pid: int, sig: int, *, group: bool) -> None:
    try:
        if group:
            os.killpg(os.getpgid(pid), sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError as e:
        raise SupervisorError(f"not allowed to signal pid {pid}: {e}") from None
    except OSError as e:
        raise SupervisorError(f"could not signal pid {pid}: {e}") from None


# --- VRAM headroom --------------------------------------------------------------------


def _wait_vram_released(pid: int, timeout: float | None = None) -> bool:
    """Wait until the driver stops attributing memory to ``pid``. True if it let go.

    A process exits before its GPU allocation is reclaimed, so ``nvidia-smi --query-compute-apps``
    keeps listing it briefly. Anything that measures free VRAM right after a stop has to wait for
    this, or it will conclude that stopping the service did nothing.
    """
    limit = config.VRAM_RELEASE_TIMEOUT if timeout is None else timeout
    deadline = time.monotonic() + limit
    while True:
        if not any(p == pid for p, _ in vram.compute_apps()):
            return True
        if time.monotonic() >= deadline:
            log.warning("vram: pid %d still holds GPU memory %gs after stopping it", pid, limit)
            return False
        time.sleep(0.25)


def ensure_headroom(need_mb: int, *, exclude: list[str] | None = None,
                    settle: float | None = None) -> vram.Admission:
    """Make room for a step that needs ``need_mb``, or raise.

    The acting half of the arbiter: ``vram.admit`` decides, this carries the decision out —
    ComfyUI's ``/free`` first because it costs seconds, then stopping a warm or phase-scoped
    service, which costs a reload. Re-measures afterwards and refuses rather than hoping.
    """
    skip = list(exclude or ())
    adm = vram.admit(need_mb, exclude=skip)
    if adm.clear:
        log.debug("vram: %s", adm.reason)
        return adm
    if not adm.ok:
        log.error("vram: %s", adm.reason)
        raise vram.VramUnavailable(adm)

    log.info("vram: %s", adm.reason)
    specs = services.load()
    for svc in adm.evict:
        spec = specs.get(svc)
        if spec is None:
            continue
        if spec.vram.elastic:
            from .clients import comfy

            log.info("vram: asking %s to drop its models (%s)", svc, spec.vram.free_endpoint)
            comfy.free()
            time.sleep(config.VRAM_SETTLE_SEC if settle is None else settle)
            continue
        state = resolve(spec, do_probe=False)
        if not state.running:
            continue
        adopted = state.adopted
        if adopted and not spec.evict_adopted:
            raise vram.VramUnavailable(vram.Admission(
                ok=False, need_mb=need_mb, free_mb=adm.free_mb, deficit_mb=adm.deficit_mb,
                blockers=adm.blockers,
                reason=(f"{svc} holds the card and was started outside vorch, and its manifest "
                        f"says not to evict it (evict_adopted: false) — stop it yourself with: "
                        f"vorch down {svc}")))
        # Say out loud that we are stopping something a person may have started.
        log.info("vram: stopping %s%s to make room", svc,
                 f" (started outside vorch, pid {state.pid})" if adopted else "")
        pid = state.pid
        stop(spec, force=True)
        # The process exits before the driver releases its allocation, so nvidia-smi keeps listing
        # the pid for a moment. Re-measuring immediately made the arbiter refuse a step it had just
        # made room for — measured: it reported 4434 MB short having freed 11268.
        if pid is not None:
            _wait_vram_released(pid)

    after = vram.admit(need_mb, exclude=skip)
    if not after.clear:
        log.error("vram: freeing did not help — %s", after.reason)
        raise vram.VramUnavailable(after)
    log.info("vram: %d MB free; admitted", after.free_mb or 0)
    return after


# --- phases ---------------------------------------------------------------------------


@contextmanager
def phase(name: str, *, need_mb: int | None = None) -> Iterator[ServiceState]:
    """Run a block with ``name`` available, and leave the card in a sane state afterwards.

    A ``warm`` service is *left running* on the way out, with its last-use time stamped: rewording
    a line and pressing Speak again must not pay another model load. The idle reaper takes it away
    later, and ``ensure_headroom`` takes it away sooner if something needs the card. A ``phase``
    service is stopped immediately. A service someone started by hand is never stopped here at all.
    """
    spec = services.get(name)
    started_by_us = False
    state = resolve(spec)
    if not state.running:
        state = start(spec, timeout=None)
        started_by_us = True
    else:
        if state.identity == "mismatch":
            raise SupervisorError(
                f"{spec.name} is running the wrong instance: {state.identity_detail}. "
                f"Stop it (vorch down {spec.name} --force) and let vorch start it.")
        if state.identity == "unverified" and spec.adopt.on_unverified == "refuse":
            raise SupervisorError(
                f"{spec.name} is running but cannot be verified ({state.identity_detail}); "
                "the manifest says to refuse rather than guess.")
        if need_mb:
            ensure_headroom(need_mb, exclude=[spec.name])
    try:
        yield state
    finally:
        touch_record(spec.name)
        if spec.lifetime == "phase" and started_by_us:
            stop(spec)
        elif spec.lifetime == "warm":
            log.info("supervisor: leaving %s warm (idle timeout %gs)",
                     spec.name, spec.idle_timeout or 0)


def reap_idle() -> list[str]:
    """Stop ``warm`` services **we started** that have been unused for their idle timeout.

    Only ever ours: a server someone launched by hand is theirs to close, however long it sits.
    """
    stopped: list[str] = []
    for spec in services.load().values():
        if spec.lifetime != "warm" or not spec.idle_timeout:
            continue
        state = resolve(spec, do_probe=False)
        if not state.running or state.record is None:
            continue          # no pidfile = someone else's, however long it sits there
        idle = state.idle_sec
        if idle is not None and idle >= spec.idle_timeout:
            log.info("supervisor: %s idle for %s; stopping it to release %d MB",
                     spec.name, vram._duration(idle), spec.vram.holds_mb)
            try:
                stop(spec)
                stopped.append(spec.name)
            except SupervisorError as e:
                log.warning("supervisor: could not reap %s (%s)", spec.name, e)
    return stopped


# --- bring-up and teardown ------------------------------------------------------------


def up(names: list[str] | None = None, *, wait: bool = True,
       reclaim_orphans: bool = False) -> list[ServiceState]:
    """Bring the world up in dependency order. Additive: adopts what is already running.

    ``warm`` and ``phase`` services are skipped unless named explicitly — they exist to be started
    on demand, and starting the narration server here would block every render.
    """
    specs = services.load()
    explicit = set(names or ())
    out: list[ServiceState] = []
    for name in services.order(specs, names):
        spec = specs[name]
        state = resolve(spec)
        if state.orphan and reclaim_orphans:
            log.info("supervisor: reclaiming orphaned %s", name)
            try:
                stop(spec, force=True)
                state = resolve(spec)
            except SupervisorError as e:
                log.warning("supervisor: could not reclaim %s (%s)", name, e)
        if state.running:
            out.append(state)
            continue
        if spec.manage != "full":
            out.append(state)
            continue
        if spec.lifetime in ("warm", "phase") and name not in explicit:
            out.append(state)
            continue
        out.append(start(spec, wait=wait))
    return out


def down(names: list[str] | None = None, *, phase_only: bool = False,
         force: bool = False) -> list[str]:
    """Stop services, dependants before dependencies. Skips what vorch did not start."""
    specs = services.load()
    order = list(reversed(services.order(specs, names)))
    stopped: list[str] = []
    for name in order:
        spec = specs[name]
        if phase_only and spec.lifetime == "long":
            continue
        if spec.manage != "full":
            continue
        state = resolve(spec, do_probe=False)
        if not state.running:
            continue
        if state.adopted and not force and not state.orphan:
            log.info("supervisor: leaving %s alone — started outside vorch (use --force)", name)
            continue
        try:
            stop(spec, force=force or state.orphan)
            stopped.append(name)
        except SupervisorError as e:
            log.warning("supervisor: %s", e)
    return stopped


_handlers_installed = False


def install_exit_handlers() -> None:
    """Stop the services we started when this process goes away.

    Only on the main thread, and chaining to whatever handler is already there — uvicorn installs
    its own, and replacing it would break shutdown. ``SIGKILL`` cannot be caught, which is exactly
    why the pidfile exists: the next ``vorch up --reclaim-orphans`` picks up what this misses.
    """
    global _handlers_installed
    if _handlers_installed or threading.current_thread() is not threading.main_thread():
        return
    atexit.register(_stop_ours)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(sig)
        except (OSError, ValueError):
            continue

        def handler(signum, frame, _previous=previous):
            _stop_ours()
            if callable(_previous) and _previous not in (signal.SIG_IGN, signal.SIG_DFL):
                _previous(signum, frame)
            elif _previous is signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        try:
            signal.signal(sig, handler)
        except (OSError, ValueError):
            continue
    _handlers_installed = True


def _stop_ours() -> None:
    """Reap the non-``long`` services this process started; leave long-lived ones up."""
    for name in list(_OURS):
        try:
            spec = services.get(name)
        except ServiceError:
            continue
        if spec.lifetime == "long":
            continue                  # Gemma and ComfyUI outlive one API process on purpose
        try:
            stop(spec)
        except Exception as e:        # exit paths must never raise
            log.warning("supervisor: could not stop %s on exit (%s)", name, e)
