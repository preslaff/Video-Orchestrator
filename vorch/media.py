"""Media helpers — extract a still frame from a generated video clip.

Used by sequential continuity (DESIGN.md / docs/sequential-continuity-plan.md):
the last frame of scene N's clip becomes the start image for scene N+1. Pure I/O
via ffprobe + ffmpeg (no GPU); mirrors the frame-extraction pattern in
skills/gemma-evaluate.md.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger("vorch.media")


class MediaError(RuntimeError):
    """Raised when ffprobe/ffmpeg fail or produce no frame."""


def _count_frames(video_path: Path) -> int | None:
    """Return the exact frame count via ffprobe, or None if probing fails."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", str(video_path),
            ],
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning("ffprobe failed for %s: %s", video_path, e)
        return None
    text = out.decode().strip()
    return int(text) if text.isdigit() else None


def last_frame(video_path: str | Path, dest: str | Path | None = None) -> Path:
    """Extract the final frame of ``video_path`` to a PNG and return its path.

    Probes the exact frame count and selects frame ``TOTAL-1``; if probing fails
    (e.g. a container without frame counts), falls back to seeking near the end
    (``-sseof``). ``dest`` defaults to ``<video>_last.png`` beside the clip.
    """
    video_path = Path(video_path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    dest = Path(dest) if dest is not None else video_path.with_name(f"{video_path.stem}_last.png")

    total = _count_frames(video_path)
    if total is not None and total > 0:
        # Exact: pick the last frame by index. Clips are short (~49 frames), trivial.
        select = f"select=eq(n\\,{total - 1})"
        cmd = ["ffmpeg", "-y", "-i", str(video_path),
               "-vf", select, "-frames:v", "1", str(dest)]
    else:
        # Fallback: seek 0.2s before the end and grab one frame.
        log.warning("frame count unavailable for %s; using -sseof fallback", video_path)
        cmd = ["ffmpeg", "-y", "-sseof", "-0.2", "-i", str(video_path),
               "-frames:v", "1", str(dest)]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise MediaError(f"ffmpeg failed to extract last frame of {video_path}: {e}") from e
    if not dest.is_file():
        raise MediaError(f"ffmpeg produced no frame for {video_path}")
    log.info("last frame: %s -> %s", video_path, dest)
    return dest


def fit_image(src: str | Path, width: int, height: int, dest: str | Path | None = None) -> Path:
    """Cover-fit ``src`` to exactly ``width``x``height`` (scale up/down, centre crop).

    Used for edit references so a reference-driven workflow (KV-edit derives its
    output size from the first reference) produces the profile's format.
    ``dest`` defaults to ``<src>_<W>x<H>.png`` beside the source.
    """
    src = Path(src).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"image not found: {src}")
    dest = Path(dest) if dest is not None else src.with_name(f"{src.stem}_{width}x{height}.png")
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-frames:v", "1", str(dest)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise MediaError(f"ffmpeg failed to fit {src} to {width}x{height}: {e}") from e
    if not dest.is_file():
        raise MediaError(f"ffmpeg produced no image for {src}")
    return dest


def sample_frames(video_path: str | Path, n: int = 5, dest_dir: str | Path | None = None) -> list[Path]:
    """``n`` evenly spaced frames (first .. last) as PNGs, for vision evaluation."""
    video_path = Path(video_path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    n = max(1, n)
    dest_dir = Path(dest_dir) if dest_dir is not None else video_path.parent
    total = _count_frames(video_path)
    if not total:
        raise MediaError(f"cannot count frames of {video_path}")
    if n == 1:
        indices = [total - 1]
    else:
        indices = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})
    out: list[Path] = []
    for idx in indices:
        dest = dest_dir / f"{video_path.stem}_f{idx:04d}.png"
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"select=eq(n\\,{idx})",
               "-frames:v", "1", str(dest)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise MediaError(f"ffmpeg failed to extract frame {idx} of {video_path}: {e}") from e
        if not dest.is_file():
            raise MediaError(f"ffmpeg produced no frame {idx} for {video_path}")
        out.append(dest)
    return out
