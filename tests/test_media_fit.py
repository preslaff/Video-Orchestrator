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


def test_sample_frames_evenly_spaced(tmp_path):
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=8:duration=2",
                    "-pix_fmt", "yuv420p", str(clip)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = media.sample_frames(clip, 4)
    assert [f.name for f in frames] == ["c_f0000.png", "c_f0005.png", "c_f0010.png", "c_f0015.png"]
    assert all(f.is_file() for f in frames)
    assert media.sample_frames(clip, 1)[0].name == "c_f0015.png"
