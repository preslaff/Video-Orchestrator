"""Endpoints, paths, and generation defaults.

Single place to point the orchestrator at the local services and ComfyUI
directories. Values come from NOTES.md §2. Override any of these via the
matching VORCH_* environment variable.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# --- Service endpoints -------------------------------------------------------
GEMMA_URL = os.environ.get("VORCH_GEMMA_URL", "http://localhost:8080")
COMFY_URL = os.environ.get("VORCH_COMFY_URL", "http://localhost:8188")

# --- Filesystem locations ----------------------------------------------------
# ComfyUI reads LoadImage nodes from input/ and writes results to output/.
COMFY_ROOT = Path(os.environ.get("VORCH_COMFY_ROOT", "~/comfy/ComfyUI")).expanduser()
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"

# Workflow JSON templates live alongside this package, in the repo root.
WORKFLOW_DIR = Path(
    os.environ.get("VORCH_WORKFLOW_DIR", Path(__file__).resolve().parent.parent / "workflows")
)
# User-installed workflows (JSON + YAML pairs) override built-ins by name.
USER_WORKFLOW_DIR = Path(
    os.environ.get("VORCH_USER_WORKFLOW_DIR", "~/.vorch/workflows")
).expanduser()

# --- Generation defaults -----------------------------------------------------
# Keyframe (FLUX.2 Klein). Base model + 24 steps + CFG 5 for oil texture (NOTES §3).
KEYFRAME_WIDTH = 1024
KEYFRAME_HEIGHT = 576  # video-ready 16:9
KEYFRAME_STEPS = 24
KEYFRAME_CFG = 5

# Animation (Wan 2.2 i2v). length MUST be 4n+1 (NOTES §3 / SKILL.md invariant 3).
WAN_WIDTH = 832
WAN_HEIGHT = 480
WAN_LENGTH = 49
WAN_FPS = 16

# Face swap (ReActor).
REACTOR_RESTORE_MODEL = "GFPGANv1.4.pth"

# --- Behaviour ---------------------------------------------------------------
RUN_TIMEOUT = int(os.environ.get("VORCH_RUN_TIMEOUT", "600"))  # seconds to wait per workflow

# Seeds. 0 is our "pick a random one" sentinel — NOT sent to ComfyUI verbatim
# (the API has no randomize step, so seed 0 would be deterministic). resolve_seed
# materializes a concrete seed so the value we record actually reproduces the run.
SEED_RANDOM = 0
_SEED_MIN, _SEED_MAX = 1, 2**31 - 1


def resolve_seed(seed: int) -> int:
    """Concrete seed: pass a fixed one through; materialize a random one for SEED_RANDOM."""
    return seed if seed != SEED_RANDOM else random.randint(_SEED_MIN, _SEED_MAX)


# --- VRAM (16GB ceiling; Option B = Gemma on CPU, ComfyUI owns the GPU) -------
# Reclaim ComfyUI VRAM before a heavy step if free falls below the guard.
VRAM_GUARD_MB = int(os.environ.get("VORCH_VRAM_GUARD_MB", "10000"))
VRAM_SETTLE_SEC = float(os.environ.get("VORCH_VRAM_SETTLE_SEC", "3"))

# Job state + decision log: PostgreSQL. Default = local socket, peer auth, db "vorch"
# (`vorch db init` creates it). Tests use VORCH_TEST_DATABASE_URL (tests/conftest.py).
DATABASE_URL = os.environ.get("VORCH_DATABASE_URL", "postgresql:///vorch")

# Web UI (Next.js) and the origins FastAPI trusts for direct browser calls.
UI_URL = os.environ.get("VORCH_UI_URL", "http://localhost:3000")
CORS_ORIGINS = tuple(
    o.strip()
    for o in os.environ.get(
        "VORCH_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
)

# Legacy SQLite file from before the PostgreSQL move — only `vorch db import-sqlite`
# reads it now.
STATE_DB = Path(os.environ.get("VORCH_STATE_DB", "runs/vorch.db")).expanduser()
