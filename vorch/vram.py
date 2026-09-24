"""VRAM monitoring + reclaim guard (NOTES.md §4, skills/vram-manager.md).

16GB is the binding constraint. Under the default **Option B** (Gemma pinned to
CPU, ComfyUI owns the GPU) this module is intentionally thin: read GPU memory via
``nvidia-smi`` and reclaim ComfyUI's VRAM before a heavy step (Wan especially).
The two worlds don't fight for the card, so there is no live model-swap dance —
just a free + settle before the dual-model Wan phase, with a safety guard.

All queries are best-effort: if ``nvidia-smi`` is absent (CI, no NVIDIA GPU),
the metrics return ``None`` and the guard still reclaims unconditionally.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from . import config
from .backends import ComfyBackend
from .clients import comfy

log = logging.getLogger("vorch.vram")


def _smi(query: str) -> str | None:
    """Run nvidia-smi for one query field; return GPU 0's value or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except (OSError, subprocess.CalledProcessError):
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    return first.strip() or None


def gpu_free_mb() -> int | None:
    """Free VRAM on GPU 0 in MB, or None if nvidia-smi is unavailable."""
    v = _smi("memory.free")
    return int(v) if v and v.isdigit() else None


def snapshot() -> dict | None:
    """{'used', 'free', 'total'} MB for GPU 0, or None if unavailable."""
    v = _smi("memory.used,memory.free,memory.total")
    if not v:
        return None
    try:
        used, free, total = (int(x) for x in v.split(","))
    except ValueError:
        return None
    return {"used": used, "free": free, "total": total}


def log_state(label: str = "") -> None:
    """Log current GPU memory (best-effort) — confirm VRAM before heavy ops."""
    tag = f" [{label}]" if label else ""
    s = snapshot()
    if s is None:
        log.info("vram%s: unavailable (no nvidia-smi)", tag)
    else:
        log.info("vram%s: used=%d free=%d total=%d MB", tag, s["used"], s["free"], s["total"])


def reclaim(
    min_mb: int | None = None, *, settle: float = config.VRAM_SETTLE_SEC,
    backend: ComfyBackend | None = None,
) -> int | None:
    """Reclaim ComfyUI VRAM, wait for it to settle, and return free MB after.

    ``min_mb=None`` (the default) frees unconditionally — use at a phase boundary
    (e.g. Klein/ReActor -> Wan). Pass ``min_mb`` to free only when free VRAM has
    dropped below that guard (a cheaper safety check before a heavy step).
    A **remote** ``backend`` has its own GPU: nothing to do here (no /free, no
    settle, no nvidia-smi) and the result is None.
    """
    if backend is not None and backend.remote:
        log.debug("vram: %s is remote; no local reclaim", backend.name)
        return None
    before = gpu_free_mb()
    if min_mb is not None and before is not None and before >= min_mb:
        log.debug("vram: %d MB free >= %d guard; no reclaim", before, min_mb)
        return before
    comfy.free(backend=backend)
    if settle:
        time.sleep(settle)
    after = gpu_free_mb()
    if before is not None and after is not None:
        log.info("vram: reclaimed ComfyUI models — free %d -> %d MB", before, after)
    else:
        log.info("vram: reclaimed ComfyUI models")
    return after


# --- who actually holds the card ------------------------------------------------------
#
# Everything above answers "how much is free". That was enough while the card had exactly two
# users, Gemma pinned to the CPU and ComfyUI owning the GPU. It stopped being enough when a third
# model arrived: the `gpu` check could report "only 2009 MB free; ComfyUI may still hold models"
# while ComfyUI held 222 MB and a narration server nobody had stopped held 11268.
#
# So: attribute memory to processes, name them from the manifest, and decide admission. This half
# measures and decides; it never kills anything and never imports the supervisor. Acting on an
# Admission is `supervisor.ensure_headroom`, which can also tell whether a process is one of ours.

#: Overridable so tests can point the process scan at a fixture tree instead of the real one.
PROC = Path("/proc")

_CLOCK_TICKS = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0


@dataclass(frozen=True)
class Holder:
    """A process holding GPU memory, named from the service manifest where possible."""

    pid: int
    used_mb: int
    service: str | None          # manifest name, or None for something we cannot identify
    cmdline: str
    cwd: str | None
    evictable: bool              # the manifest says its memory can be taken back
    started_at: float | None     # unix timestamp, from /proc/<pid>/stat

    @property
    def uptime_sec(self) -> float | None:
        return None if self.started_at is None else max(0.0, time.time() - self.started_at)

    @property
    def uptime_str(self) -> str:
        return "?" if self.uptime_sec is None else _duration(self.uptime_sec)

    @property
    def label(self) -> str:
        return self.service or Path(self.cmdline.split(" ")[0] or "?").name or f"pid {self.pid}"

    def describe(self) -> str:
        parts = [f"{self.label} (pid {self.pid}, {self.used_mb} MB"]
        if (up := self.uptime_sec) is not None:
            parts.append(f", up {_duration(up)}")
        parts.append(")")
        return "".join(parts)

    def to_dict(self) -> dict:
        return {"pid": self.pid, "used_mb": self.used_mb, "service": self.service,
                "cmdline": self.cmdline, "cwd": self.cwd, "evictable": self.evictable,
                "uptime_sec": self.uptime_sec, "uptime": self.uptime_str, "label": self.label}


def _duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def compute_apps() -> list[tuple[int, int]]:
    """``(pid, used_mb)`` for every process with memory on GPU 0, or [] when unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except (OSError, subprocess.CalledProcessError):
        return []
    found: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        pid, _, used = line.partition(",")
        try:
            found.append((int(pid.strip()), int(used.strip())))
        except ValueError:
            continue                      # "[N/A]" happens on some drivers
    return found


def _proc_cmdline(pid: int) -> str:
    try:
        return (PROC / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace").strip()
    except OSError:
        return ""


def _proc_cwd(pid: int) -> str | None:
    try:
        return os.readlink(PROC / str(pid) / "cwd")
    except OSError:
        return None


def _proc_started_at(pid: int) -> float | None:
    """Unix time the process began, from field 22 of /proc/<pid>/stat."""
    try:
        raw = (PROC / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
        # The comm field is parenthesised and may itself contain spaces, so split after it.
        fields = raw[raw.rindex(")") + 1:].split()
        ticks = float(fields[19])                 # field 22 overall, 20th after the comm
        boot = time.time() - float((PROC / "uptime").read_text().split()[0])
        return boot + ticks / _CLOCK_TICKS
    except (OSError, ValueError, IndexError):
        return None


def identify(pid: int, cmdline: str, cwd: str | None) -> tuple[str | None, bool]:
    """``(manifest name, evictable)`` for a process, by the manifest's ``match`` rules.

    ``evictable`` reflects the *manifest* only: ComfyUI frees without stopping, and a
    warm/phase-scoped service may be stopped when ``evict_adopted`` allows it. Whether this
    particular process is one vorch started is the supervisor's question, not ours.
    """
    from . import services

    try:
        specs = services.load()
    except services.ServiceError as e:
        log.debug("vram: cannot read the service manifest (%s); holders stay unnamed", e)
        return None, False
    for spec in specs.values():
        try:
            m = spec.matcher()
        except services.ServiceError:
            continue                      # an unresolvable ${VAR} must not break attribution
        if not m.cmdline_contains or not all(n in cmdline for n in m.cmdline_contains):
            continue
        if m.cwd and cwd and str(Path(m.cwd)) != str(Path(cwd)):
            continue
        evictable = spec.vram.elastic or (spec.lifetime in ("warm", "phase") and spec.evict_adopted)
        return spec.name, evictable
    return None, False


def holders() -> list[Holder]:
    """Every process holding GPU memory, largest first, named where the manifest recognises it."""
    out: list[Holder] = []
    for pid, used_mb in compute_apps():
        cmdline = _proc_cmdline(pid)
        cwd = _proc_cwd(pid)
        name, evictable = identify(pid, cmdline, cwd)
        out.append(Holder(pid=pid, used_mb=used_mb, service=name, cmdline=cmdline, cwd=cwd,
                          evictable=evictable, started_at=_proc_started_at(pid)))
    return sorted(out, key=lambda h: -h.used_mb)


def attribute(snap: dict | None = None) -> dict | None:
    """``snapshot()`` plus the holder table, or None when nvidia-smi is unavailable.

    ``unattributed_mb`` is the difference between what the driver reports used and what the
    per-process figures add up to — context and driver overhead, plus anything we could not see.
    """
    s = snap if snap is not None else snapshot()
    if s is None:
        return None
    hs = holders()
    accounted = sum(h.used_mb for h in hs)
    return {**s, "holders": hs, "unattributed_mb": max(0, s["used"] - accounted)}


def describe_holders(hs: list[Holder]) -> str:
    return ", ".join(h.describe() for h in hs) or "nothing we can see"


# --- admission ------------------------------------------------------------------------


class VramUnavailable(RuntimeError):
    """Not enough VRAM for a step, and nothing left that we are allowed to free."""

    def __init__(self, admission: Admission):
        super().__init__(admission.reason)
        self.admission = admission


@dataclass(frozen=True)
class Admission:
    """Whether a step needing ``need_mb`` can run, and what would have to be freed first."""

    ok: bool
    need_mb: int
    free_mb: int | None
    deficit_mb: int = 0
    evict: tuple[str, ...] = ()          # manifest names, in the order they should be freed
    blockers: tuple[Holder, ...] = ()    # holding memory we may not take
    reason: str = ""

    @property
    def clear(self) -> bool:
        """It fits right now, with nothing to free first."""
        return self.ok and not self.evict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "need_mb": self.need_mb, "free_mb": self.free_mb,
                "deficit_mb": self.deficit_mb, "evict": list(self.evict),
                "blockers": [b.to_dict() for b in self.blockers], "reason": self.reason}


def admit(need_mb: int, *, exclude: Iterable[str] = (), snap: dict | None = None) -> Admission:
    """Can a step wanting ``need_mb`` run, and what must be freed first?

    Pure arithmetic over a measurement — it frees nothing. ``exclude`` names services that must be
    left alone (the one we are about to start, above all: a narration server should not be asked to
    evict itself). Eviction order is cheapest first, because ComfyUI's ``/free`` costs about three
    seconds while stopping the narration model costs a 90 s reload.

    With no nvidia-smi there is nothing to arbitrate on, so the answer is yes — the same stance
    ``reclaim`` already takes when it cannot measure.
    """
    info = attribute(snap)
    if info is None:
        return Admission(ok=True, need_mb=need_mb, free_mb=None,
                         reason="nvidia-smi unavailable; not arbitrating")
    free = info["free"]
    skip = set(exclude)
    if free >= need_mb:
        return Admission(ok=True, need_mb=need_mb, free_mb=free,
                         reason=f"{free} MB free, {need_mb} needed")

    candidates = [h for h in info["holders"] if h.evictable and h.service not in skip]
    # Cheapest first: elastic /free before stopping a process we would have to reload.
    candidates.sort(key=lambda h: (0 if _is_elastic(h.service) else 1, -h.used_mb))
    plan: list[str] = []
    gained = 0
    for h in candidates:
        if free + gained >= need_mb:
            break
        if h.service and h.service not in plan:
            plan.append(h.service)
        gained += h.used_mb

    # Everything holding memory that the plan will NOT free — whether because the manifest forbids
    # it or because the caller excluded it. Nothing may silently disappear from the accounting, or
    # the arithmetic in the message stops adding up for whoever reads it.
    blockers = tuple(h for h in info["holders"]
                     if h.used_mb > 0 and (h.service is None or h.service not in plan))
    if free + gained >= need_mb:
        return Admission(
            ok=True, need_mb=need_mb, free_mb=free, evict=tuple(plan),
            reason=(f"needs {need_mb} MB, {free} free; freeing {', '.join(plan)} "
                    f"releases about {gained} MB"),
        )

    # Not every blocker is a mistake. Gemma is `resident` on purpose — it is the evaluator, and
    # stopping it would break every verdict — so it is reported as a fixed cost, never as something
    # to go and kill. Only a service that should not have been left running earns a "stop it" hint.
    fixed = [h for h in blockers if _lifetime_of(h.service) == "long"]
    stoppable = [h for h in blockers if h not in fixed]
    deficit = need_mb - (free + gained)
    parts = [f"needs {need_mb} MB, {free} free"]
    if gained:
        parts.append(f"freeing {', '.join(plan)} would add about {gained} MB")
    parts.append(f"still {deficit} MB short")
    if stoppable:
        parts.append("held by " + describe_holders(stoppable))
    if fixed:
        parts.append(describe_holders(fixed) + " is a fixed cost and stays")
    hint = next((f"; stop it with: vorch down {h.service}" for h in stoppable if h.service), "")
    if not stoppable and not fixed:
        parts.append("and nothing we can identify is holding the rest")
    return Admission(
        ok=False, need_mb=need_mb, free_mb=free, deficit_mb=deficit,
        evict=tuple(plan), blockers=blockers,
        reason="; ".join(parts) + hint,
    )


def _lifetime_of(service: str | None) -> str | None:
    if not service:
        return None
    from . import services

    try:
        return services.get(service).lifetime
    except services.ServiceError:
        return None


def _is_elastic(service: str | None) -> bool:
    if not service:
        return False
    from . import services

    try:
        return services.get(service).vram.elastic
    except services.ServiceError:
        return False
