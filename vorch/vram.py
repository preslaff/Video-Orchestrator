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
import subprocess
import time

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
