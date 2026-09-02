"""ComfyUI client — dispatch a workflow, poll for completion, collect outputs.

Mirrors the flow documented in skills/comfyui-api.md:
POST /prompt -> poll /history/{id} -> read output files -> (optionally) /free.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import requests

from .. import config


class ComfyError(RuntimeError):
    """Raised when ComfyUI rejects a prompt or a run does not complete."""


def run(wf: dict, timeout: int | None = None) -> list[dict]:
    """Submit a workflow dict and block until it finishes.

    Returns the list of output file descriptors ({filename, subfolder, type}).
    """
    timeout = config.RUN_TIMEOUT if timeout is None else timeout
    client_id = str(uuid.uuid4())

    resp = requests.post(
        f"{config.COMFY_URL}/prompt",
        json={"prompt": wf, "client_id": client_id},
    )
    if resp.status_code != 200:
        # ComfyUI returns a detailed validation error body on 400 — surface it.
        raise ComfyError(f"/prompt rejected ({resp.status_code}): {resp.text}")
    prompt_id = resp.json()["prompt_id"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = requests.get(f"{config.COMFY_URL}/history/{prompt_id}").json()
        if prompt_id in history:
            return _extract_outputs(history[prompt_id])
        time.sleep(2)
    raise ComfyError(f"Workflow {prompt_id} did not finish within {timeout}s")


def _extract_outputs(history_entry: dict) -> list[dict]:
    """Pull file descriptors from a finished history entry.

    Output files appear under outputs[node_id] keyed by media type. Filenames
    auto-increment, so always read the real name rather than assuming one.
    """
    files: list[dict] = []
    for node_id, out in history_entry.get("outputs", {}).items():
        for key in ("images", "gifs", "videos"):
            for d in out.get(key, []):
                files.append({**d, "node_id": str(node_id)})
    return files


def output_path(descriptor: dict) -> Path:
    """Resolve an output descriptor to an absolute path under COMFY_OUTPUT."""
    sub = descriptor.get("subfolder", "")
    return config.COMFY_OUTPUT / sub / descriptor["filename"]


def upload_image(filepath: str | Path) -> str:
    """Stage a local image into ComfyUI input/ so a LoadImage node can read it.

    Returns the name to assign to the LoadImage node's ``image`` field.
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{config.COMFY_URL}/upload/image",
            files={"image": (filepath.name, f)},
        )
    resp.raise_for_status()
    return resp.json()["name"]


def free(unload_models: bool = True, free_memory: bool = True) -> None:
    """Reclaim VRAM between heavy workflows (e.g. Klein -> Wan). Best effort."""
    try:
        requests.post(
            f"{config.COMFY_URL}/free",
            json={"unload_models": unload_models, "free_memory": free_memory},
            timeout=30,
        )
    except requests.RequestException:
        pass  # /free is an optimization; never fail the run over it.
