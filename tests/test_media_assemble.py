"""Clip assembly with xfade transitions (vorch/media.py) — real ffmpeg where needed."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from vorch import media
from vorch.media import Cut

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")


def _clip(tmp_path, name, colour, secs=2, size="64x64", rate=8):
    p = tmp_path / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s={size}:rate={rate}:duration={secs}",
                    "-pix_fmt", "yuv420p", str(p)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p


def test_build_filtergraph_groups_hard_cuts():
    graph, out = media.build_filtergraph([2, 2, 2, 2], [Cut("fade", 0.5), Cut("none"), Cut("wipeleft", 0.3)],
                                         width=64, height=64, fps=8)
    assert out == "[out]"
    assert "[v0][v1]xfade=transition=fade:duration=0.5:offset=1.5000[g0]" in graph
    assert "[v2][v3]xfade=transition=wipeleft:duration=0.3:offset=1.7000[g1]" in graph
    assert graph.endswith("[g0][g1]concat=n=2:v=1:a=0,format=yuv420p[out]")
    assert graph.count("setpts=PTS-STARTPTS") == 4 and "fps=8," in graph
    # one chain, two transitions: second offset = 2+2-0.5-0.3
    graph, out = media.build_filtergraph([2, 2, 2], [Cut("fade", 0.5), Cut("wipeleft", 0.3)], width=64, height=64, fps=8)
    assert out == "[g0]" and "offset=3.2000[g0]" in graph and "concat" not in graph
    # single clip / all hard cuts
    assert media.build_filtergraph([2], [], width=64, height=64, fps=8)[1] == "[v0]"
    graph, out = media.build_filtergraph([2, 2], [Cut("none")], width=64, height=64, fps=8)
    assert out == "[out]" and "[v0][v1]concat=n=2" in graph


def test_total_duration_formula():
    assert media.total_duration([2, 2, 2, 2], [Cut("fade", 0.5), Cut("none"), Cut("wipeleft", 0.3)]) == pytest.approx(7.2)
    assert media.total_duration([3.0625], []) == 3.0625


def test_validate_cuts_rules():
    media.validate_cuts([2, 2], [Cut("fade", 0.5)], fps=8)
    with pytest.raises(ValueError, match="unknown transition"):
        media.validate_cuts([2, 2], [Cut("swoosh", 0.5)], fps=8)
    with pytest.raises(ValueError, match="need 1 cut"):
        media.validate_cuts([2, 2], [], fps=8)
    with pytest.raises(ValueError, match="shorter than both clips"):
        media.validate_cuts([2, 1], [Cut("fade", 1.0)], fps=8)
    with pytest.raises(ValueError, match="shorter than one frame"):
        media.validate_cuts([2, 2], [Cut("fade", 0.05)], fps=8)
    with pytest.raises(ValueError, match="exceed"):
        media.validate_cuts([2, 1, 2], [Cut("fade", 0.6), Cut("fade", 0.6)], fps=8)  # middle clip 1s: 0.6+0.6 > 1-0.125
    media.validate_cuts([2, 1, 2], [Cut("fade", 0.4), Cut("none")], fps=8)
    with pytest.raises(ValueError, match="at least one clip"):
        media.validate_cuts([], [], fps=8)


@needs_ffmpeg
def test_transitions_list_has_none_first_and_fade():
    t = media.transitions()
    assert t[0] == "none" and "fade" in t and "wipeleft" in t and "custom" not in t
    assert len(t) >= 40


@needs_ffmpeg
def test_probe_reports_size_fps_duration_frames(tmp_path):
    info = media.probe(_clip(tmp_path, "a", "red"))
    assert (info.width, info.height, info.fps, info.nb_frames) == (64, 64, 8.0, 16)
    assert info.duration == pytest.approx(2.0, abs=0.01) and info.frame_duration == 2.0
    assert info.codec == "h264" and info.has_audio is False
    with pytest.raises(FileNotFoundError):
        media.probe(tmp_path / "missing.mp4")


@needs_ffmpeg
def test_assemble_xfade_and_hard_cut_duration(tmp_path):
    clips = [_clip(tmp_path, n, c) for n, c in (("a", "red"), ("b", "green"), ("c", "blue"), ("d", "yellow"))]
    out = media.assemble(clips, [Cut("fade", 0.5), Cut("none"), Cut("wipeleft", 0.3)], tmp_path / "out" / "assembly.mp4")
    assert out.is_file() and not (tmp_path / "out" / "assembly.mp4.part").exists()
    info = media.probe(out)
    assert (info.width, info.height, info.fps, info.codec) == (64, 64, 8.0, "h264")
    assert info.frame_duration == pytest.approx(7.2, abs=0.13)  # ± one frame


@needs_ffmpeg
def test_assemble_single_clip_reencodes(tmp_path):
    out = media.assemble([_clip(tmp_path, "a", "red")], [], tmp_path / "single.mp4", crf=20)
    assert media.probe(out).frame_duration == pytest.approx(2.0, abs=0.13)


@needs_ffmpeg
def test_assemble_rejects_bad_specs_without_leaving_files(tmp_path):
    a, b = _clip(tmp_path, "a", "red"), _clip(tmp_path, "b", "green")
    dest = tmp_path / "x.mp4"
    with pytest.raises(ValueError):
        media.assemble([a, b], [Cut("swoosh", 0.5)], dest)
    with pytest.raises(ValueError):
        media.assemble([a, b], [Cut("fade", 2.5)], dest)
    with pytest.raises(ValueError):
        media.assemble([a, b], [], dest)
    with pytest.raises(ValueError):
        media.assemble([a, b], [Cut("fade", 0.5)], dest, encoder="magic")
    assert not dest.exists() and not dest.with_name("x.mp4.part").exists()


@needs_ffmpeg
def test_assemble_normalises_mismatched_sizes_and_fps(tmp_path):
    a = _clip(tmp_path, "a", "red")                       # 64x64 @ 8
    b = _clip(tmp_path, "b", "green", size="100x40", rate=4)
    out = media.assemble([a, b], [Cut("fade", 0.5)], tmp_path / "mix.mp4", fps=8, size=(64, 64))
    info = media.probe(out)
    assert (info.width, info.height, info.fps) == (64, 64, 8.0)
    assert info.frame_duration == pytest.approx(3.5, abs=0.13)
