"""GPU-free checks for sequential continuity — run_sequence and media.last_frame.

No live ComfyUI/ffmpeg: the step calls (keyframe/faceswap/animate), the frame
extraction (last_frame), and the staging helpers (comfy.upload_image/free) are
monkeypatched, so we exercise the chaining logic — which scenes pull a fresh
keyframe vs. carry the previous clip's last frame, and how identity re-lock is
applied.
"""

from __future__ import annotations

import subprocess

import pytest

from vorch import media, pipeline
from vorch.pipeline import Scene, run_sequence


@pytest.fixture
def master(tmp_path):
    p = tmp_path / "master.jpg"
    p.write_bytes(b"\xff\xd8\xff")  # not a real JPEG; only existence is checked
    return p


@pytest.fixture
def recorder(monkeypatch):
    """Monkeypatch the pipeline's step/staging seams and record every call."""
    calls = {"keyframe": [], "faceswap": [], "animate": [], "last_frame": [], "upload": []}

    def fake_keyframe(prompt, *, seed=0, **kw):
        calls["keyframe"].append(prompt)
        return f"kf_{len(calls['keyframe'])}.png"

    def fake_faceswap(input_name, source_name, **kw):
        calls["faceswap"].append((input_name, source_name))
        return f"swapped_{input_name}"

    def fake_animate(start_name, prompt, **kw):
        calls["animate"].append((start_name, prompt))
        return f"video_{len(calls['animate'])}.mp4"

    def fake_last_frame(video_path, dest=None):
        calls["last_frame"].append(str(video_path))
        return f"{video_path}_last.png"

    def fake_upload(path, **kw):
        calls["upload"].append(str(path))
        return str(path)

    monkeypatch.setattr(pipeline, "keyframe", fake_keyframe)
    monkeypatch.setattr(pipeline, "faceswap", fake_faceswap)
    monkeypatch.setattr(pipeline, "animate", fake_animate)
    monkeypatch.setattr(pipeline, "last_frame", fake_last_frame)
    monkeypatch.setattr(pipeline.comfy, "upload_image", fake_upload)
    monkeypatch.setattr(pipeline.comfy, "free", lambda *a, **k: None)
    # animate_frame is the real helper here; keep VRAM reclaim out of the unit test
    # (no nvidia-smi / sleep).
    monkeypatch.setattr(pipeline.vram, "reclaim", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.vram, "log_state", lambda *a, **k: None)
    return calls


def test_scene0_continuous_is_ignored_uses_fresh_keyframe(recorder, master):
    """Scene 0 is always fresh even if flagged continuous — nothing precedes it."""
    run_sequence([Scene("a castle", "pan left", continuous=True)], master)
    assert len(recorder["keyframe"]) == 1
    assert recorder["last_frame"] == []  # no previous clip to carry


def test_continuous_scene_carries_previous_last_frame(recorder, master):
    results = run_sequence(
        [
            Scene("a castle", "pan left"),
            Scene("(unused)", "push in", continuous=True),
        ],
        master,
    )
    # Only the first scene builds a keyframe; the second carries the frame.
    assert len(recorder["keyframe"]) == 1
    assert recorder["last_frame"] == ["video_1.mp4"]  # scene 1's clip fed scene 2
    assert results[1].continued_from_previous is True
    assert results[1].keyframe is None
    assert results[0].continued_from_previous is False


def test_cut_scene_builds_fresh_keyframe(recorder, master):
    """A non-continuous later scene gets its own keyframe, not the carried frame."""
    run_sequence(
        [
            Scene("a castle", "pan left"),
            Scene("a forest", "tilt up", continuous=False),
        ],
        master,
    )
    assert len(recorder["keyframe"]) == 2
    assert recorder["last_frame"] == []


def test_relock_swaps_master_onto_carried_frame(recorder, master):
    results = run_sequence(
        [Scene("a castle", "pan left"), Scene("x", "push in", continuous=True)],
        master,
        relock_identity=True,
    )
    # Carried frame goes through faceswap; start_image is the swapped result.
    carried = "video_1.mp4_last.png"
    assert (carried, str(master)) in recorder["faceswap"]
    assert results[1].swapped == f"swapped_{carried}"
    assert results[1].start_image == f"swapped_{carried}"


def test_no_relock_animates_raw_carried_frame(recorder, master):
    results = run_sequence(
        [Scene("a castle", "pan left"), Scene("x", "push in", continuous=True)],
        master,
        relock_identity=False,
    )
    carried = "video_1.mp4_last.png"
    # No swap for the continuous scene; only scene 0's keyframe was swapped.
    assert len(recorder["faceswap"]) == 1
    assert results[1].swapped is None
    assert results[1].start_image == carried


def test_video_carries_forward_through_three_scenes(recorder, master):
    run_sequence(
        [
            Scene("a", "m1"),
            Scene("b", "m2", continuous=True),
            Scene("c", "m3", continuous=True),
        ],
        master,
    )
    # Each continuous scene pulls the immediately preceding clip.
    assert recorder["last_frame"] == ["video_1.mp4", "video_2.mp4"]


def test_missing_master_raises(recorder, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_sequence([Scene("a", "m")], tmp_path / "nope.jpg")


def test_face_swap_off_does_no_swaps(recorder, tmp_path):
    """With face_swap=False, no master is needed and no faceswap call happens."""
    results = run_sequence(
        [Scene("a", "m1"), Scene("b", "m2", continuous=True)],
        master_face="",  # no master
        face_swap=False,
    )
    assert recorder["faceswap"] == []  # zero swaps (fresh + carried)
    assert all(r.swapped is None for r in results)
    # scene 1 still carries the previous clip's last frame
    assert recorder["last_frame"] == ["video_1.mp4"]


# --- media.last_frame --------------------------------------------------------


def test_last_frame_extracts_exact_index(monkeypatch, tmp_path):
    """ffprobe reports 49 frames -> ffmpeg selects index 48; dest is created."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    seen = {}

    def fake_check_output(cmd, **kw):
        seen["probe"] = cmd
        return b"49\n"

    def fake_run(cmd, **kw):
        seen["run"] = cmd
        # ffmpeg writes the dest (last positional arg).
        from pathlib import Path

        Path(cmd[-1]).write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(media.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    out = media.last_frame(video)
    assert out.name == "clip_last.png"
    assert out.is_file()
    assert "select=eq(n\\,48)" in seen["run"]


def test_last_frame_falls_back_when_probe_fails(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    def fake_check_output(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    def fake_run(cmd, **kw):
        from pathlib import Path

        Path(cmd[-1]).write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(media.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    out = media.last_frame(video)
    assert out.is_file()


def test_last_frame_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        media.last_frame(tmp_path / "nope.mp4")
