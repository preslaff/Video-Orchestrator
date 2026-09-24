"""Endpoints, paths, and generation defaults.

Single place to point the orchestrator at the local services and ComfyUI
directories. Values come from NOTES.md §2. Override any of these via the
matching VORCH_* environment variable.
"""

from __future__ import annotations

import os
import random
from pathlib import Path


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> list[str]:
    """Load ``KEY=VALUE`` lines (``export`` prefix and quotes allowed, ``#`` comments)
    from ``path`` (default ``VORCH_ENV_FILE`` or ``~/vorch/vorch.env``) into the
    process environment. Existing variables win unless ``override``. Returns the
    names set. Secrets such as the Modal proxy token live here so `vorch serve`
    and the CLI find them regardless of the shell that started them.
    """
    p = Path(path or os.environ.get("VORCH_ENV_FILE", "~/vorch/vorch.env")).expanduser()
    if not p.is_file():
        return []
    names: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
            names.append(key)
    return names


load_env_file()  # before any VORCH_* / token lookup below

# --- Service endpoints -------------------------------------------------------
GEMMA_URL = os.environ.get("VORCH_GEMMA_URL", "http://localhost:8080")
COMFY_URL = os.environ.get("VORCH_COMFY_URL", "http://localhost:8188")
# OmniVoice-BG: Bulgarian narration, cloned from a reference clip. A third local service, spoken
# to over HTTP exactly like the other two — but unlike them it is started only for the narration
# phase, because it wants 11268 MiB of a 16303 MiB card (measured 2026-09; earlier comments said
# ~5.3 GB and were wrong by over 2x). It cannot coexist with a Wan render at all — see
# docs/services.md and NOTES.md §4 for the arithmetic.
TTS_URL = os.environ.get("VORCH_TTS_URL", "http://localhost:8002")
OMNIVOICE_DIR = Path(os.environ.get("VORCH_OMNIVOICE_DIR", "~/WORK/OmniVoice")).expanduser()
# The cloned voice: a reference clip and its EXACT transcript. Viktoria/11s is the clean pair —
# the 20 s and 30 s transcripts carry a typo ("нибо" for "ниво"), and ref_text must match.
TTS_REF_AUDIO = os.environ.get("VORCH_TTS_REF_AUDIO", "Viktoria/11s.wav")
TTS_REF_TEXT_FILE = os.environ.get("VORCH_TTS_REF_TEXT_FILE", "Viktoria/11s.txt")

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
    os.environ.get("VORCH_USER_WORKFLOW_DIR", "~/vorch/workflows")
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
# ComfyUI client: per-HTTP-call timeout, /history polling interval, and how long
# connection errors are tolerated while a *remote* backend's container cold-starts.
COMFY_REQUEST_TIMEOUT = float(os.environ.get("VORCH_COMFY_REQUEST_TIMEOUT", "30"))
COMFY_POLL_INTERVAL = float(os.environ.get("VORCH_COMFY_POLL_INTERVAL", "2"))
COMFY_COLD_START_GRACE = float(os.environ.get("VORCH_COMFY_COLD_START_GRACE", "240"))
# Remote ComfyUI backends (e.g. Modal): name -> url/auth/timeouts. Missing file = local only.
BACKENDS_FILE = Path(os.environ.get("VORCH_BACKENDS_FILE", "~/vorch/backends.yaml")).expanduser()
ASSEMBLE_TIMEOUT = int(os.environ.get("VORCH_ASSEMBLE_TIMEOUT", "600"))  # ffmpeg clip assembly
# Narration. One line is about a second to synthesise; the model takes ~90 s to load, which is
# why the phase opens the server once for a whole job. Chunking mirrors OmniVoice's own
# read_book.py: its accentor is trained on sentences under 250 characters.
TTS_REQUEST_TIMEOUT = float(os.environ.get("VORCH_TTS_REQUEST_TIMEOUT", "180"))
TTS_START_TIMEOUT = float(os.environ.get("VORCH_TTS_START_TIMEOUT", "300"))
TTS_CHUNK_CHARS = int(os.environ.get("VORCH_TTS_CHUNK_CHARS", "220"))
TTS_CHUNK_GAP = float(os.environ.get("VORCH_TTS_CHUNK_GAP", "0.35"))

# Seeds. 0 is our "pick a random one" sentinel — NOT sent to ComfyUI verbatim
# (the API has no randomize step, so seed 0 would be deterministic). resolve_seed
# materializes a concrete seed so the value we record actually reproduces the run.
SEED_RANDOM = 0
_SEED_MIN, _SEED_MAX = 1, 2**31 - 1


def resolve_seed(seed: int) -> int:
    """Concrete seed: pass a fixed one through; materialize a random one for SEED_RANDOM."""
    return seed if seed != SEED_RANDOM else random.randint(_SEED_MIN, _SEED_MAX)


def derive_seed(base: int, *, scene: int = 0, variant: int = 0, attempt: int = 1) -> int:
    """Per-scene / per-variant / per-attempt seed from a job's base seed.

    A random base stays random (each generation materializes its own seed, which
    the manifest records); a fixed base is offset deterministically so every
    scene, variant and retry differs yet reproduces: scene 0 / variant 0 /
    attempt 1 is the base itself.
    """
    if base == SEED_RANDOM:
        return SEED_RANDOM
    return base + scene * 10_000 + variant * 100 + (attempt - 1)


# --- VRAM (16GB ceiling; Option B = Gemma on CPU, ComfyUI owns the GPU) -------
# Reclaim ComfyUI VRAM before a heavy step if free falls below the guard.
VRAM_GUARD_MB = int(os.environ.get("VORCH_VRAM_GUARD_MB", "10000"))
VRAM_SETTLE_SEC = float(os.environ.get("VORCH_VRAM_SETTLE_SEC", "3"))

# Job state + decision log: PostgreSQL. Default = local socket, peer auth, db "vorch"
# (`vorch db init` creates it). Tests use VORCH_TEST_DATABASE_URL (tests/conftest.py).
DATABASE_URL = os.environ.get("VORCH_DATABASE_URL", "postgresql:///vorch")

# Service manifest: what must be running, what it costs the card, and who may start it.
# Built-in defaults in vorch/services_builtin/services.yaml; this file overrides them per
# service, key by key. Missing file = built-ins only. See docs/services.md.
SERVICES_FILE = Path(os.environ.get("VORCH_SERVICES_FILE", "~/vorch/services.yaml")).expanduser()
# Where `vorch serve` answers, for the manifest's own health probe of the API.
API_URL = os.environ.get("VORCH_API_URL", "http://127.0.0.1:8000")

# --- Runtime state and logs ---------------------------------------------------
# Pidfiles for supervised services (so an orphan survives a hard kill and can be found again)
# and per-service log files. Both live under ~/vorch/, beside backends.yaml and vorch.env.
RUN_DIR = Path(os.environ.get("VORCH_RUN_DIR", "~/vorch/run")).expanduser()
LOG_DIR = Path(os.environ.get("VORCH_LOG_DIR", "~/vorch/logs")).expanduser()
# Rotation, applied to vorch's own log by logsetup and to each service's log at start.
LOG_MAX_BYTES = int(os.environ.get("VORCH_LOG_MAX_BYTES", str(8 * 1024 * 1024)))
LOG_BACKUPS = int(os.environ.get("VORCH_LOG_BACKUPS", "5"))
# How often `vorch serve` looks for a warm service that has gone idle (see supervisor.reap_idle).
REAP_INTERVAL_SEC = float(os.environ.get("VORCH_REAP_INTERVAL_SEC", "60"))

# Setup phase: where projects keep their assets and where finished jobs are exported.
PROJECTS_ROOT = Path(os.environ.get("VORCH_PROJECTS_ROOT", "~/vorch/projects")).expanduser()
EXPORT_ROOT = Path(os.environ.get("VORCH_EXPORT_ROOT", "~/vorch/export")).expanduser()
# Describe (caption + suggest a label for) every uploaded asset in the background with Gemma.
AUTO_DESCRIBE = os.environ.get("VORCH_AUTO_DESCRIBE", "1") not in ("0", "false", "no", "")

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
