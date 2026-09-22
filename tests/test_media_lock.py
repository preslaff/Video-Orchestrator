"""media.lock_regions against the real ffmpeg — the pixels a video model must not redraw."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from vorch import media

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _lavfi(dest, source: str, *extra: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", source, *extra, str(dest)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _pixel(path, x: int, y: int, frame: int = 0) -> tuple[int, int, int]:
    # 2x2, not 1x1: a yuv420p frame has no chroma plane left to crop at one pixel.
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame}),crop=2:2:{x}:{y}",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return tuple(raw[:3])


def _close(got, want, tol: int = 14) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(got, want))


def test_a_region_comes_from_the_source_and_the_rest_from_the_clip(tmp_path):
    """The whole point: inside the rectangle the source's pixels win on every frame,
    outside it the rendered clip is untouched."""
    clip, source = tmp_path / "clip.mp4", tmp_path / "kf.png"
    _lavfi(clip, "color=c=blue:s=128x128:rate=8:duration=1", "-pix_fmt", "yuv420p")
    _lavfi(source, "color=c=red:s=128x128", "-frames:v", "1")

    out = media.lock_regions(clip, source, [media.Region(x=0.25, y=0.25, w=0.5, h=0.5)],
                             tmp_path / "locked.mp4", feather=0, track=False)

    assert media.probe(out).nb_frames == media.probe(clip).nb_frames
    for frame in (0, 4, 7):
        assert _close(_pixel(out, 64, 64, frame), (255, 0, 0)), "the marked region must be the source"
        assert _close(_pixel(out, 4, 4, frame), (0, 0, 255)), "the rest must be the render"


def test_the_source_is_scaled_to_the_clip_so_the_sizes_need_not_match(tmp_path):
    """A keyframe is often larger than the clip it fed; the patch follows the clip."""
    clip, source = tmp_path / "clip.mp4", tmp_path / "kf.png"
    _lavfi(clip, "color=c=black:s=64x64:rate=8:duration=0.5", "-pix_fmt", "yuv420p")
    _lavfi(source, "color=c=white:s=256x256", "-frames:v", "1")
    out = media.lock_regions(clip, source, [media.Region(x=0, y=0, w=1, h=1)],
                             tmp_path / "locked.mp4", feather=0, track=False)
    assert media.probe(out).width == 64
    assert _close(_pixel(out, 32, 32), (255, 255, 255))


def test_the_paste_follows_a_shot_that_pushes_in(tmp_path):
    """A fixed rectangle would slide off the moment the camera dollies — which is most of
    what a product shot does. The mark has to be found again in every frame."""
    clip, source = tmp_path / "clip.mp4", tmp_path / "kf.png"
    # A marked square on a textured ground, zoomed steadily in: a synthetic dolly.
    _lavfi(clip,
           "testsrc=size=320x320:rate=10:duration=1",
           "-vf", "zoompan=z='1+0.035*on':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=320x320",
           "-pix_fmt", "yuv420p")
    first = tmp_path / "first.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-frames:v", "1", str(first)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(first), str(source)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    region = media.Region(x=0.3, y=0.3, w=0.4, h=0.4)
    out = media.lock_regions(clip, source, [region], tmp_path / "locked.mp4", feather=0)
    assert media.probe(out).nb_frames == media.probe(clip).nb_frames

    # The region grew with the shot, so the patch must cover more of the last frame than a
    # fixed rectangle would: sample a point just outside the original box on the final frame
    # and check it now carries the source's content rather than the render's.
    last = media.probe(clip).nb_frames - 1
    x, y, w, h = region.box(320, 320)
    edge = _pixel(out, x - 10, y + h // 2, last)
    plain = _pixel(clip, x - 10, y + h // 2, last)
    assert edge != plain, "the paste did not grow with the push-in"


def test_a_rectangle_off_the_frame_is_refused_before_anything_runs():
    media.Region(x=0.5, y=0.5, w=0.5, h=0.5)              # exactly to the edge is fine
    with pytest.raises(ValueError, match="outside the frame"):
        media.Region(x=0.8, y=0.1, w=0.3, h=0.1)
    with pytest.raises(ValueError, match="positive size"):
        media.Region(x=0.1, y=0.1, w=0, h=0.1)


def test_region_boxes_are_pixels_grown_and_clamped():
    r = media.Region(x=0.25, y=0.5, w=0.25, h=0.25)
    assert r.box(400, 400) == (100, 200, 100, 100)
    assert r.box(400, 400, grow=10) == (90, 190, 120, 120)
    edge = media.Region(x=0, y=0, w=0.5, h=0.5)
    assert edge.box(100, 100, grow=20) == (0, 0, 70, 70)   # clamped, not negative


def test_the_patch_follows_the_scene_light():
    """A patch cut from one frame is frozen at that frame's exposure. The render around it is
    not, and a constant patch against a drifting surround reads as the label pulsing."""
    import numpy as np

    steady = media._relight([100.0] * 10)
    assert np.allclose(steady, 1.0)                     # nothing drifted: nothing to correct

    brightening = media._relight([100.0 + 2 * i for i in range(12)])
    assert brightening[0] == pytest.approx(1.0, abs=0.02)   # frame 0 is the reference
    assert brightening[-1] > brightening[0]                 # and the patch brightens with it
    assert np.all(np.diff(brightening) >= -1e-9)            # smoothly, never backwards

    # A wild reading cannot wash the brand out, whichever way it goes.
    assert media._relight([100.0, 900.0, 900.0, 900.0]).max() <= 1.18 + 1e-9
    assert media._relight([100.0, 1.0, 1.0, 1.0]).min() >= 1 - 0.18 - 1e-9
    assert np.allclose(media._relight([100.0] * 6, limit=0.0), 1.0)
    assert media._relight([]).tolist() == [1.0]         # nothing measured: no correction


def test_the_light_is_read_from_around_the_mark_not_inside_it():
    """Inside is what the model got wrong; outside is the scene still lighting the object."""
    import numpy as np

    from vorch.tracking import Placement

    grey = np.full((100, 100), 80.0, np.float32)
    grey[40:60, 40:60] = 250.0                          # the marked region: bright, and ignored
    assert media._ring_level(grey, Placement(40, 40, 20, 20)) == pytest.approx(80.0)

    grey[20:80, 20:80] = np.where(grey[20:80, 20:80] == 250.0, 250.0, 130.0)
    assert media._ring_level(grey, Placement(40, 40, 20, 20)) == pytest.approx(130.0)
    # A region flush against the edge still reads, and a degenerate one does not raise.
    assert media._ring_level(grey, Placement(0, 0, 20, 20)) > 0
    assert media._ring_level(grey, Placement(0, 0, 1, 1), pad=0) == 0.0


def test_missing_inputs_and_empty_marks_are_errors(tmp_path):
    clip, source = tmp_path / "clip.mp4", tmp_path / "kf.png"
    _lavfi(clip, "color=c=black:s=32x32:rate=8:duration=0.25", "-pix_fmt", "yuv420p")
    _lavfi(source, "color=c=white:s=32x32", "-frames:v", "1")
    with pytest.raises(FileNotFoundError):
        media.lock_regions(tmp_path / "nope.mp4", source, [media.Region(x=0, y=0, w=1, h=1)],
                           tmp_path / "o.mp4")
    with pytest.raises(ValueError, match="no regions"):
        media.lock_regions(clip, source, [], tmp_path / "o.mp4")
