"""Face swap — ReActor (image-based template)."""

from __future__ import annotations

from pathlib import Path

from .. import config
from . import run_step


def faceswap(
    input_image_name: str,
    source_image_name: str,
    *,
    restore_model: str = config.REACTOR_RESTORE_MODEL,
    timeout: int | None = None,
) -> Path:
    """Swap the ``source`` face onto ``input``; return the output path.

    Both names must already be staged in ComfyUI input/ (LoadImage reads by name).
    """
    return run_step(
        "faceswap",
        timeout=timeout,
        input_image=input_image_name,
        source_image=source_image_name,
        restore_model=restore_model,
    )[0]
