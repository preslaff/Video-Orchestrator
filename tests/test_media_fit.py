"""media.fit_image against the real ffmpeg (cover-fit + centre crop)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from vorch import media

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _dims(path) -> tuple[int, int]:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(path),
    ]).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


def test_fit_image_covers_and_crops(tmp_path):
    src = tmp_path / "wide.png"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=400x100", "-frames:v", "1", str(src)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = media.fit_image(src, 576, 1024)
    assert out == tmp_path / "wide_576x1024.png"
    assert _dims(out) == (576, 1024)
    with pytest.raises(FileNotFoundError):
        media.fit_image(tmp_path / "missing.png", 64, 64)
