"""Clip assembly with xfade transitions (vorch/media.py) — real ffmpeg where needed."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
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
def test_probe_reports_the_pixel_format_and_leaves_an_untagged_range_unsaid(tmp_path):
    """What an h264 clip actually says about itself — and what it leaves unsaid."""
    info = media.probe(_clip(tmp_path, "pf", "red"))
    assert info.pix_fmt == "yuv420p"
    # ffprobe reports no usable range for a stream that does not tag one, which is what
    # ComfyUI's SaveVideo produces. The field must not pretend otherwise.
    assert info.color_range in ("", "unknown")


@needs_ffmpeg
def test_a_mask_survives_h264_even_though_it_is_stored_limited_range(tmp_path):
    """Measured, because the plausible-sounding opposite would have cost a day.

    An untagged yuv420p stream **does** store a white mask as 235, not 255 — so reading its Y
    plane directly would cap alpha at 0.92 and let 8% of the render through the middle of a lock.
    But ffmpeg's own yuv->gray conversion already expands limited to full, which is the path
    ``_lock_tracked`` uses, so the naive read is correct and needs no `in_range` correction.
    Restating the expansion is idempotent rather than a second application.

    This test exists to stop someone "fixing" the non-bug by adding an expansion, and to catch it
    if a future ffmpeg stops expanding on its own — in which case masks silently go translucent.
    """
    clip = tmp_path / "mask.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "color=c=white:s=32x32:rate=8:duration=1",
                    "-pix_fmt", "yuv420p", "-crf", "0", str(clip)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def first_byte(pix_fmt, *vf):
        cmd = ["ffmpeg", "-v", "error", "-i", str(clip)]
        if vf:
            cmd += ["-vf", *vf]
        cmd += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
        return int(np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.uint8)[0])

    assert first_byte("yuv420p") == 235, "white is stored limited-range; that part is real"
    assert first_byte("gray") == 255, "ffmpeg expands yuv->gray itself, so a mask reads full"
    assert first_byte("gray", "scale=in_range=tv:out_range=pc") == 255, "idempotent, not doubled"
    assert media.probe(clip).color_range in ("", "unknown")


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


# --- per-clip speed --------------------------------------------------------------


def test_filtergraph_speed_chains():
    base, _ = media.build_filtergraph([2, 2], [Cut("fade", 0.5)], width=64, height=64, fps=8)
    same, _ = media.build_filtergraph([2, 2], [Cut("fade", 0.5)], width=64, height=64, fps=8, speeds=[1.0, 1.0])
    assert same == base  # speed 1 keeps the historical chain byte for byte
    fast, out = media.build_filtergraph([2, 2], [Cut("fade", 0.5)], width=64, height=64, fps=8, speeds=[0.5, 2.0])
    assert "setpts=PTS/0.5,fps=8,tpad=stop_mode=clone:stop_duration=2,trim=duration=4.0000,format=yuv420p" in fast
    assert "setpts=PTS/2,fps=8,tpad=stop_mode=clone:stop_duration=2,trim=duration=1.0000," in fast
    assert "offset=3.5000[g0]" in fast and out == "[g0]"  # retimed 4.0 - 0.5
    smooth, _ = media.build_filtergraph([2, 2], [Cut("fade", 0.5)], width=64, height=64, fps=8, speeds=[0.5, 2.0], smooth=True)
    assert "setpts=PTS/0.5,tpad=stop_mode=clone:stop_duration=2,minterpolate=fps=8:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,trim=duration=4.0000," in smooth
    assert "setpts=PTS/2,fps=8,tpad" in smooth  # speed-ups never interpolate
    assert media.total_duration([2, 2], [Cut("fade", 0.5)], [0.5, 2.0]) == pytest.approx(4.5)
    assert media.retimed([3.0625, 2.0], [0.5, 1.0]) == [6.125, 2.0]


def test_validate_speeds():
    media.validate_speeds(None, 3)
    media.validate_speeds([0.25, 4.0, 1.0], 3)
    with pytest.raises(ValueError, match="need 2 speed"):
        media.validate_speeds([1.0], 2)
    with pytest.raises(ValueError, match="clip 1: speed 0.2"):
        media.validate_speeds([0.2, 1.0], 2)
    with pytest.raises(ValueError, match="clip 2"):
        media.validate_speeds([1.0, 5.0], 2)


@needs_ffmpeg
def test_assemble_with_speeds_fast_and_smooth(tmp_path):
    a, b = _clip(tmp_path, "a", "red"), _clip(tmp_path, "b", "green")
    out = media.assemble([a, b], [Cut("fade", 0.5)], tmp_path / "fast.mp4", speeds=[0.5, 2.0])
    info = media.probe(out)
    assert info.frame_duration == pytest.approx(4.5, abs=0.13) and info.fps == 8.0
    out = media.assemble([a, b], [Cut("fade", 0.5)], tmp_path / "smooth.mp4", speeds=[0.5, 2.0], smooth=True)
    assert media.probe(out).frame_duration == pytest.approx(4.5, abs=0.26)
    with pytest.raises(ValueError):
        media.assemble([a, b], [Cut("fade", 0.5)], tmp_path / "bad.mp4", speeds=[0.1, 1.0])
    assert not (tmp_path / "bad.mp4").exists() and not (tmp_path / "bad.mp4.part").exists()
    # A transition longer than a sped-up clip is caught against the RETIMED length (1 s).
    with pytest.raises(ValueError, match="shorter than both"):
        media.assemble([a, b], [Cut("fade", 1.5)], tmp_path / "bad2.mp4", speeds=[1.0, 2.0])


# --- audio (Phase C: LTX clips carry a track) --------------------------------------------


def _clip_with_audio(path, seconds, *, silent=False, fps=8):
    import subprocess

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc=size=64x64:rate={fps}:duration={seconds}"]
    if not silent:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return path


def test_audio_filtergraph_mirrors_video_cuts():
    graph, out = media.build_audio_filtergraph([2.0, 2.0, 2.0], [Cut("fade", 0.5), Cut("none", 0)], [True, False, True], speeds=[1.0, 1.0, 4.0])
    assert "[0:a]aresample=48000" in graph and "apad=whole_dur=2.0000,atrim=duration=2.0000" in graph
    assert "anullsrc=r=48000:cl=stereo,atrim=duration=2.0000" in graph       # silent clip -> silence
    assert "[2:a]" in graph and "atempo=2,atempo=2,apad=whole_dur=0.5000" in graph  # 4x -> two atempo stages
    assert "[a0][a1]acrossfade=d=0.5:c1=tri:c2=tri[ag0]" in graph
    assert graph.endswith("[ag0][a2]concat=n=2:v=0:a=1[aout]") and out == "[aout]"
    assert media._tempo(0.25) == "atempo=0.5,atempo=0.5," and media._tempo(1.0) == "" and media._tempo(1.5) == "atempo=1.5,"


def test_assemble_keeps_audio_when_asked(tmp_path):
    import shutil

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    a = _clip_with_audio(tmp_path / "a.mp4", 2)
    b = _clip_with_audio(tmp_path / "b.mp4", 2, silent=True)
    assert media.probe(a).has_audio and not media.probe(b).has_audio
    silent = media.assemble([a, b], [Cut("fade", 0.5)], tmp_path / "silent.mp4")
    assert not media.probe(silent).has_audio                       # default unchanged
    out = media.assemble([a, b], [Cut("fade", 0.5)], tmp_path / "audio.mp4", audio=True)
    info = media.probe(out)
    assert info.has_audio and abs(info.frame_duration - 3.5) < 0.2
    both_silent = media.assemble([b, b], [Cut("fade", 0.5)], tmp_path / "bs.mp4", audio=True)
    assert not media.probe(both_silent).has_audio                  # no track anywhere -> silent file


# --- mastering: the assembly is where a delivery format is made ------------------------


def test_the_scaler_is_lanczos_and_grain_is_opt_in():
    graph, _ = media.build_filtergraph([2, 2], [Cut("fade", 0.5)], width=1920, height=1080, fps=24)
    assert "scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos" in graph
    assert "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1" in graph   # letterbox, never stretch
    assert "noise=" not in graph

    grainy, _ = media.build_filtergraph([2], [], width=1920, height=1080, fps=24, grain=1.5)
    assert "noise=alls=1.5:allf=t+u" in grainy
    clamped, _ = media.build_filtergraph([2], [], width=64, height=64, fps=8, grain=99)
    assert "noise=alls=3:allf=t+u" in clamped


@needs_ffmpeg
def test_masters_small_clips_to_a_real_1080p_file(tmp_path):
    """The whole point: 480p-class clips delivered as a 1080p master."""
    clips = [_clip(tmp_path, "a", "teal", size="320x180", rate=16),
             _clip(tmp_path, "b", "maroon", size="320x180", rate=16)]
    dest = tmp_path / "master.mp4"
    media.assemble(clips, [Cut("fade", 0.5)], dest, fps=24, size=(1920, 1080), grain=1)
    info = media.probe(dest)
    assert (info.width, info.height) == (1920, 1080) and round(info.fps) == 24

    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=profile,color_primaries,color_transfer,color_space,pix_fmt", "-of", "default=nw=1", str(dest),
    ]).decode()
    assert "profile=High" in out and "pix_fmt=yuv420p" in out
    assert out.count("bt709") == 3          # primaries, transfer and matrix are tagged, not guessed


@needs_ffmpeg
def test_encoder_flags_carry_gop_and_an_optional_bitrate_cap(tmp_path, monkeypatch):
    seen = {}
    real_run = subprocess.run

    def spy(cmd, *a, **kw):
        if cmd and cmd[0] == "ffmpeg" and "-filter_complex" in cmd:
            seen["cmd"] = cmd
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(media.subprocess, "run", spy)
    clip = _clip(tmp_path, "c", "navy", size="64x64", rate=8)
    media.assemble([clip], [], tmp_path / "out.mp4", fps=24, size=(64, 64), max_mbps=25)
    cmd = seen["cmd"]
    assert cmd[cmd.index("-g") + 1] == "48"                     # a closed GOP of 2 s at 24 fps
    assert cmd[cmd.index("-keyint_min") + 1] == "24"
    assert cmd[cmd.index("-maxrate") + 1] == "25M" and cmd[cmd.index("-bufsize") + 1] == "50M"
    assert "+faststart" in cmd[cmd.index("-movflags") + 1]      # streams without a full download


# --- trims ------------------------------------------------------------------------------


def test_trim_span_snaps_to_the_frame_grid():
    """A person types milliseconds; a cut can only land on a frame. At 16 fps that grid is
    62.5 ms, so 1.234 s is not available and 1.25 s is."""
    assert media.trim_span(4.0, 1.234, None, 16.0) == (1.25, 4.0)
    assert media.trim_span(4.0, None, 2.03, 16.0) == (0.0, 2.0)
    assert media.trim_span(4.0, None, None, 16.0) == (0.0, 4.0)
    assert media.trim_span(4.0, -5.0, 99.0, 16.0) == (0.0, 4.0)          # clamped to the clip
    # At 8 fps the same request snaps somewhere else: the grid is the CLIP's, not the output's.
    assert media.trim_span(4.0, 1.234, None, 8.0) == (1.25, 4.0)
    assert media.trim_span(4.0, 1.1, None, 8.0) == (1.125, 4.0)
    for start, end in ((3.0, 3.01), (2.0, 2.0), (3.0, 1.0)):
        with pytest.raises(ValueError, match="less than one frame"):
            media.trim_span(4.0, start, end, 16.0)


def test_the_span_is_cut_before_anything_else_in_the_graph():
    graph, _ = media.build_filtergraph([1.0], [], width=64, height=64, fps=8, spans=[(0.5, 1.5)])
    assert graph.startswith("[0:v]trim=start=0.5000:end=1.5000,setpts=PTS-STARTPTS,scale=")
    plain, _ = media.build_filtergraph([1.0], [], width=64, height=64, fps=8)
    assert "trim=start=" not in plain                                    # no trim asked, none added


def test_the_audio_is_cut_to_the_same_span():
    """A trimmed picture against an untrimmed track drifts, and nothing downstream would notice."""
    graph, _ = media.build_audio_filtergraph([1.0], [], [True], spans=[(0.5, 1.5)])
    assert graph.startswith("[0:a]atrim=start=0.5000:end=1.5000,asetpts=PTS-STARTPTS,aresample=")
    silent, _ = media.build_audio_filtergraph([1.0], [], [False], spans=[(0.5, 1.5)])
    assert "anullsrc" in silent and "atrim=duration=1.0000" in silent     # generated, already exact


@needs_ffmpeg
def test_a_trimmed_clip_contributes_only_the_part_it_kept(tmp_path):
    a, b = _clip(tmp_path, "a", "red", secs=4, rate=16), _clip(tmp_path, "b", "blue", secs=4, rate=16)
    out = media.assemble([a, b], [Cut("none")], tmp_path / "j.mp4",
                         trims=[(1.234, 2.0), (0.0, 0.5)])               # snaps to 1.25 -> 2.0
    assert media.probe(out).frame_duration == pytest.approx(0.75 + 0.5, abs=0.03)
    whole = media.assemble([a], [], tmp_path / "w.mp4")
    assert media.probe(whole).frame_duration == pytest.approx(4.0, abs=0.03)


@needs_ffmpeg
def test_an_impossible_trim_names_the_clip_it_belongs_to(tmp_path):
    a = _clip(tmp_path, "a", "red", secs=2, rate=16)
    with pytest.raises(ValueError, match=r"clip 1 \(a\.mp4\)"):
        media.assemble([a], [], tmp_path / "x.mp4", trims=[(1.0, 1.01)])
    with pytest.raises(ValueError, match="need 2 trim"):
        media.assemble([a, a], [Cut("none")], tmp_path / "y.mp4", trims=[(0.0, 1.0)])


# --- narration ---------------------------------------------------------------------------


def test_item_offsets_agree_with_the_picture():
    """A voice is placed by time, so its arithmetic must be the picture's arithmetic — a
    transition overlaps two clips, and every later clip starts earlier than the plain sum."""
    d, c = [3.0, 3.0, 3.0], [Cut("fade", 0.5), Cut("none", 0.0)]
    offs = media.item_offsets(d, c)
    assert offs == [0.0, 2.5, 5.5]
    assert offs[-1] + d[-1] == pytest.approx(media.total_duration(d, c)), "the last clip ends at the total"
    # Retiming moves everything after it.
    assert media.item_offsets([2.0, 4.0], [Cut("fade", 1.0)], [1.0, 2.0]) == [0.0, 1.0]
    assert media.item_offsets([2.0], []) == [0.0]


def test_a_voice_forces_a_track_onto_silent_clips():
    """A Wan timeline has no audio at all; a voice over it is the entire point, so the gate that
    turns audio off when no clip has a track must not apply to narration."""
    graph, out = media.build_filtergraph([2.0], [], width=64, height=64, fps=8)
    assert "[0:a]" not in graph                       # unchanged without narration
    parts, label = media._narration_graph([(Path("/v.wav"), 1.25)], first_input=3, bed=None,
                                          total=4.0, duck=6.0)
    assert "[3:a]" in parts and "adelay=1250|1250" in parts
    assert "apad=whole_dur=4.0000,atrim=duration=4.0000" in parts, "padded short, cut long"
    assert label == "[n0]" and "sidechaincompress" not in parts, "nothing to duck"


def test_the_clips_duck_under_the_voice():
    parts, label = media._narration_graph(
        [(Path("/a.wav"), 0.0), (Path("/b.wav"), 2.0)], first_input=2, bed="[aout]", total=5.0, duck=6.0)
    assert "amix=inputs=2:normalize=0" in parts, "normalize=0 or the clips are quietly halved"
    assert "[voice]asplit=2[vkey][vmix]" in parts, "the voice is heard AND presses the bed down"
    assert "[aout][vkey]sidechaincompress=" in parts and "ratio=6" in parts
    assert label == "[amixed]"


@needs_ffmpeg
def test_a_voice_lands_at_its_offset_and_the_bed_ducks_under_it(tmp_path):
    bed = _clip_with_audio(tmp_path / "bed.mp4", 4, fps=16)
    voice = tmp_path / "v.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=900:sample_rate=24000:duration=1", str(voice)], check=True)
    out = media.assemble([bed], [], tmp_path / "n.mp4", audio=True, narration=[(voice, 2.0)])
    assert media.probe(out).has_audio
    assert media.probe(out).frame_duration == pytest.approx(4.0, abs=0.2), "the voice never lengthens the film"

    def tone(path, hz):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                              "-af", f"bandpass=f={hz}:width_type=h:w=40", "-ac", "1", "-ar", "8000",
                              "-f", "f32le", "-"], capture_output=True).stdout
        a = np.frombuffer(raw, np.float32)
        return [float(np.sqrt((a[i * 4000:(i + 1) * 4000] ** 2).mean())) for i in range(len(a) // 4000)]

    voiced = tone(out, 900)
    assert max(voiced[4:6]) > 3 * (voiced[0] + 1e-6), "the voice is there, and only after 2 s"
    bedline = tone(out, 440)
    assert min(bedline[4:6]) < 0.7 * bedline[0], "the bed ducks while it speaks"
    assert bedline[-1] > 0.8 * bedline[0], "and comes back up after"


@needs_ffmpeg
def test_joining_spoken_pieces_puts_a_breath_between_them(tmp_path):
    src = tmp_path / "p.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:sample_rate=24000:duration=1", str(src)], check=True)
    blob = src.read_bytes()
    one = media.concat_audio([blob], tmp_path / "one.wav")
    assert media.audio_seconds(one) == pytest.approx(1.0, abs=0.05)
    two = media.concat_audio([blob, blob], tmp_path / "two.wav", gap=0.35)
    assert media.audio_seconds(two) == pytest.approx(2.35, abs=0.1)
    with pytest.raises(media.MediaError, match="nothing to join"):
        media.concat_audio([], tmp_path / "none.wav")


@needs_ffmpeg
def test_probe_refuses_audio_but_audio_seconds_reads_it(tmp_path):
    """`probe` requires a video stream, so a narration wav needs its own reader."""
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:sample_rate=24000:duration=1.5", str(wav)], check=True)
    with pytest.raises(media.MediaError, match="no video stream"):
        media.probe(wav)
    assert media.audio_seconds(wav) == pytest.approx(1.5, abs=0.05)
    with pytest.raises(FileNotFoundError):
        media.audio_seconds(tmp_path / "gone.wav")


# --- captions: the one filter that draws them ------------------------------------------


def _ass(path: Path, text="ЗДРАВЕЙ", *, start=0.0, end=2.0, size=480, height=480) -> Path:
    path.write_text(
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {size}\nPlayResY: {height}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: t0,DejaVu Sans,28,&H00FFFFFF,&H00000000,&H00000000,-1,2,0,5,10,10,10,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,0:00:{start:05.2f},0:00:{end:05.2f},t0,,0,0,0,,{text}\n",
        encoding="utf-8")
    return path


def test_filter_path_escaping_is_not_ass_escaping(tmp_path):
    """Two unrelated layers. This one is ffmpeg's filter-argument syntax; ASS escaping is about
    braces and line breaks in the subtitle text. Confusing them corrupts the graph."""
    assert media._filter_path(Path("/tmp/a b/x.ass")) == "/tmp/a b/x.ass"
    assert media._filter_path(Path("/tmp/od:d/x.ass")) == r"/tmp/od\:d/x.ass"
    assert media._filter_path(Path("/tmp/it's/x.ass")) == r"/tmp/it\'s/x.ass"


@pytest.mark.parametrize("durations,cuts,label", [
    ([2], [], "[v0]"),                                        # one clip, no join
    ([2, 2], [Cut("fade", 0.5)], "[g0]"),                     # one xfade group
    ([2, 2], [Cut("none")], "[out]"),                         # a hard cut splits into groups
])
def test_the_text_stage_consumes_whatever_label_the_join_produced(durations, cuts, label):
    """The final video label is not fixed — hardcoding one breaks the other two shapes."""
    _, out = media.build_filtergraph(durations, cuts, width=64, height=64, fps=8)
    assert out == label
    stage = media._text_stage(out, Path("/t/titles.ass"), width=480, height=832)
    assert stage.startswith(f"{label}ass=/t/titles.ass")
    assert stage.endswith(":original_size=480x832:shaping=complex[txt]")


@needs_ffmpeg
def test_no_titles_leaves_the_command_untouched(tmp_path, monkeypatch):
    """A job without text must produce exactly today's ffmpeg invocation."""
    seen = []
    real = media.subprocess.run

    def spy(cmd, **kw):
        seen.append(list(cmd))
        return real(cmd, **kw)

    monkeypatch.setattr(media.subprocess, "run", spy)
    clips = [_clip(tmp_path, "a", "red"), _clip(tmp_path, "b", "blue")]
    media.assemble(clips, [Cut("fade", 0.5)], tmp_path / "plain.mp4", fps=8, size=(64, 64))
    graph = seen[-1][seen[-1].index("-filter_complex") + 1]
    assert "ass=" not in graph
    assert seen[-1][seen[-1].index("-map") + 1] == "[g0]", "the join's own label is mapped"


@needs_ffmpeg
def test_titles_are_drawn_over_the_joined_picture(tmp_path):
    """End to end: the pixels change while a caption is on screen and not after it."""
    clips = [_clip(tmp_path, "a", "black", secs=3), _clip(tmp_path, "b", "black", secs=3)]
    cuts = [Cut("fade", 0.5)]
    plain = media.assemble(clips, cuts, tmp_path / "plain.mp4", fps=8, size=(64, 64))
    ass = _ass(tmp_path / "t.ass", start=0.0, end=1.5, size=64, height=64)
    titled = media.assemble(clips, cuts, tmp_path / "titled.mp4", fps=8, size=(64, 64), titles=ass)

    def frame(path, at):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(path),
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8)

    assert frame(titled, 0.7).mean() > frame(plain, 0.7).mean() + 1, "text is on screen"
    assert abs(float(frame(titled, 3.0).mean()) - float(frame(plain, 3.0).mean())) < 0.5, \
        "and gone again once the event ends"


@needs_ffmpeg
def test_a_single_clip_with_titles_still_renders(tmp_path):
    """The `[v0]` shape — no xfade at all — is the one a hardcoded `[out]` would break."""
    clip = _clip(tmp_path, "solo", "green", secs=2)
    ass = _ass(tmp_path / "t.ass", start=0.0, end=1.0, size=64, height=64)
    out = media.assemble([clip], [], tmp_path / "solo_titled.mp4", fps=8, size=(64, 64), titles=ass)
    assert out.is_file() and media.probe(out).width == 64


def test_text_fonts_only_offers_what_can_draw_cyrillic(monkeypatch):
    """Asked of fontconfig, because that is the same database libass resolves a family through —
    a font it omits would come out as boxes, silently."""
    monkeypatch.setattr(media, "_FONTS", None)
    monkeypatch.setattr(media.subprocess, "check_output",
                        lambda *a, **k: b"DejaVu Sans\nNoto Sans,Noto Sans Regular\nLiberation Serif\n")
    fonts = media.text_fonts(refresh=True)
    assert fonts[0] == "DejaVu Sans", "the default leads, the rest are alphabetical"
    assert fonts[1:] == ["Liberation Serif", "Noto Sans", "Noto Sans Regular"]


def test_text_fonts_degrades_without_fontconfig(monkeypatch):
    monkeypatch.setattr(media, "_FONTS", None)

    def boom(*a, **k):
        raise OSError("no fc-list")

    monkeypatch.setattr(media.subprocess, "check_output", boom)
    assert media.text_fonts(refresh=True) == ["DejaVu Sans"]
