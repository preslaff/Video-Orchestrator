"""Endpoints, paths, and generation defaults.

Single place to point the orchestrator at the local services and ComfyUI
directories. Values come from NOTES.md §2. Override any of these via the
matching VORCH_* environment variable.
"""

from __future__ import annotations

import os
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
SEED_RANDOM = 0  # ComfyUI convention: 0 = random seed
