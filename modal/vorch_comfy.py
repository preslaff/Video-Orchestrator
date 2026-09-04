"""vorch ComfyUI backends on Modal — one deployed app, one ComfyUI per GPU class.

    modal deploy modal/vorch_comfy.py

Each function below is a ComfyUI server behind Modal's proxy authentication
(`requires_proxy_auth=True`: callers need a Modal proxy token in the
``Modal-Key`` / ``Modal-Secret`` headers — never expose a bare GPU endpoint).
The local orchestrator talks to it exactly like the local ComfyUI: uploads,
``POST /prompt``, ``GET /history``, ``GET /view`` (see docs/modal-backend.md).

Model weights live in the ``vorch-models`` Volume, mounted at ComfyUI's
``models/`` directory; fill it with ``modal run modal/download_models.py``.

Design choices (docs/modal-backend.md §4):
- ComfyUI is pinned to the same version as the local install (version drift
  changes node signatures the registry YAMLs rely on).
- ``max_containers=1``: one ComfyUI per GPU class, it queues prompts itself;
  ``@modal.concurrent`` lets that single container serve uploads and polls.
- ``scaledown_window=600``: a running job keeps the container warm through its
  polling; idle time is billed, so never set ``min_containers`` in this file.
"""

from __future__ import annotations

import subprocess
import time
import urllib.request

import modal

COMFY_VERSION = "v0.34.0"  # == local ComfyUI (`vorch setup check` shows it); bump both together
COMFY_DIR = "/root/comfy/ComfyUI"
PORT = 8188
MINUTES = 60

# Custom node packs (git URL, ref). Phase B needs none: the Wan 2.2 fp16/fp8
# templates use core nodes only (UNETLoader, CLIPLoader, WanImageToVideo,
# CreateVideo/SaveVideo). Phase C adds e.g.
#   ("https://github.com/Gourieff/ComfyUI-ReActor", "main"),
#   ("https://github.com/Lightricks/ComfyUI-LTXVideo", "main"),
CUSTOM_NODES: list[tuple[str, str]] = []


def _node_commands() -> list[str]:
    cmds = []
    for url, ref in CUSTOM_NODES:
        name = url.rstrip("/").rsplit("/", 1)[-1]
        dest = f"{COMFY_DIR}/custom_nodes/{name}"
        cmds.append(f"git clone --depth 1 --branch {ref} {url} {dest}")
        cmds.append(f"if [ -f {dest}/requirements.txt ]; then pip install -r {dest}/requirements.txt; fi")
    return cmds


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        f"git clone --depth 1 --branch {COMFY_VERSION} https://github.com/comfyanonymous/ComfyUI {COMFY_DIR}",
        # PyPI's Linux torch wheels bundle CUDA; ComfyUI's requirements pin the rest.
        f"pip install -r {COMFY_DIR}/requirements.txt",
        *_node_commands(),
    )
    # Modal can only mount a Volume on an empty path; the clone ships `models/`
    # with placeholder files and subfolders, so drop it (separate layer: cheap rebuild).
    .run_commands(f"rm -rf {COMFY_DIR}/models")
)

models = modal.Volume.from_name("vorch-models", create_if_missing=True)
app = modal.App("vorch-comfy", image=image)

COMMON = dict(
    timeout=24 * 60 * MINUTES,      # a container may serve a whole day of jobs
    scaledown_window=10 * MINUTES,  # idle time before the GPU is released (billed)
    max_containers=1,               # one ComfyUI per GPU class; it queues prompts itself
    volumes={f"{COMFY_DIR}/models": models},
)


def _serve() -> None:
    """Start ComfyUI and return once it answers; Modal then routes the web port."""
    cmd = [
        "python", "main.py",
        "--listen", "0.0.0.0", "--port", str(PORT),
        "--max-upload-size", "1024",  # MB; reference images and clips staged by the orchestrator
        "--disable-auto-launch",
    ]
    proc = subprocess.Popen(cmd, cwd=COMFY_DIR)
    deadline = time.monotonic() + 8 * MINUTES
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during startup (code {proc.returncode})")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/system_stats", timeout=2)
            return
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(1)
    raise RuntimeError("ComfyUI did not answer /system_stats in time")


@app.function(gpu="L40S", memory=64 * 1024, **COMMON)
@modal.concurrent(max_inputs=8)
@modal.web_server(PORT, startup_timeout=10 * MINUTES, requires_proxy_auth=True)
def comfy_l40s() -> None:
    """48 GB: Wan 2.2 fp8 at 480p, Klein 4B/9B bf16, ReActor, interpolation."""
    _serve()


@app.function(gpu="H100", memory=128 * 1024, **COMMON)
@modal.concurrent(max_inputs=8)
@modal.web_server(PORT, startup_timeout=10 * MINUTES, requires_proxy_auth=True)
def comfy_h100() -> None:
    """80 GB: Wan 2.2 fp16 (2 x 28.6 GB) at 480p/720p, LTX-2.5 bf16."""
    _serve()
