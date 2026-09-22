"""The similarity tracker that lets a locked region follow a moving shot."""

from __future__ import annotations

import numpy as np
import pytest

from vorch import tracking
from vorch.tracking import Placement


def _mark(h: int, w: int) -> np.ndarray:
    """A patch with structure in both directions — a blank one cannot be tracked."""
    a = np.full((h, w), 40.0, np.float32)
    a[h // 4: h // 2, w // 6: w // 2] = 230.0
    a[h // 2: 3 * h // 4, w // 3: 5 * w // 6] = 150.0
    a[:, : max(1, w // 12)] = 255.0
    return a


def _scene(size: int, patch: np.ndarray, x: int, y: int) -> np.ndarray:
    """The patch dropped into a noisy background, so the match has to be earned."""
    rng = np.random.default_rng(0)
    frame = rng.uniform(0, 60, (size, size)).astype(np.float32)
    frame[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
    return frame


def test_a_region_that_does_not_move_stays_exactly_where_it_was():
    """A locked-off shot must not shiver: noise alone must never shift the paste."""
    tpl = _mark(40, 60)
    at = Placement(80, 70, 60, 40)
    rng = np.random.default_rng(1)
    for _ in range(5):
        frame = _scene(200, tpl, 80, 70)
        frame += rng.normal(0, 2.0, frame.shape)       # a different grain every frame
        at = tracking.follow(tpl, frame, at)
        assert (at.x, at.y, at.w, at.h) == (80, 70, 60, 40)


def test_it_follows_a_drift_frame_by_frame():
    tpl = _mark(40, 60)
    at = Placement(60, 60, 60, 40)
    for step in range(1, 6):
        x, y = 60 + 3 * step, 60 + 2 * step
        at = tracking.follow(tpl, _scene(220, tpl, x, y), at)
        assert abs(at.x - x) <= 1 and abs(at.y - y) <= 1


def test_it_follows_a_push_in_and_grows_with_it():
    """The whole point: under a dolly the region must scale, not just translate."""
    tpl = _mark(40, 60)
    at = Placement(70, 70, 60, 40)
    size = 1.0
    for _ in range(12):
        size *= 1.03
        h, w = int(round(40 * size)), int(round(60 * size))
        grown = tracking.resample(tpl, h, w)
        x, y = 70 - (w - 60) // 2, 70 - (h - 40) // 2
        at = tracking.follow(tpl, _scene(260, grown, x, y), at)
    assert at.w > 75 and at.h > 50                      # it grew with the shot (~1.42x)
    assert abs(at.w / at.h - 1.5) < 0.12                # and kept the label's proportions


def test_a_frame_it_cannot_match_holds_the_last_placement_and_says_so():
    """Better a paste that stays put than one that jumps to a false match."""
    tpl = _mark(40, 60)
    at = Placement(80, 70, 60, 40)
    blank = np.full((200, 200), 128.0, np.float32)
    after = tracking.follow(tpl, blank, at)
    assert (after.x, after.y, after.w, after.h) == (80, 70, 60, 40)
    assert after.confidence < tracking.MIN_CONFIDENCE

    flat = np.zeros((40, 60), np.float32)               # a featureless mark is untrackable
    assert tracking.follow(flat, _scene(200, tpl, 80, 70), at).x == 80


def test_a_placement_is_kept_inside_the_frame():
    tpl = _mark(20, 20)
    at = tracking.follow(tpl, _scene(100, tpl, 5, 5), Placement(-30, -30, 20, 20))
    assert at.x >= 0 and at.y >= 0


def test_resample_changes_size_without_inventing_content():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert tracking.resample(a, 8, 8).shape == (8, 8)
    assert tracking.resample(a, 2, 2).shape == (2, 2)
    assert tracking.resample(a, 4, 4).tolist() == a.tolist()
    assert tracking.resample(a, 8, 8).min() == a.min() and tracking.resample(a, 8, 8).max() == a.max()


def test_the_feather_fades_in_from_every_edge():
    m = tracking.feather_mask(20, 20, 4)
    assert m[10, 10] == pytest.approx(1.0)              # solid in the middle
    for corner in ((0, 0), (0, 19), (19, 0), (19, 19)):
        assert m[corner] < 0.2                          # and transparent at the corners
    assert tracking.feather_mask(6, 6, 0).min() == 1.0  # no feather asked, none applied
    tiny = tracking.feather_mask(3, 3, 9)               # a feather wider than the patch
    assert tiny.shape == (3, 3) and tiny.max() <= 1.0


def test_paste_blends_in_place_and_clips_at_the_edge():
    frame = np.zeros((40, 40, 3), np.float32)
    patch = np.full((10, 10, 3), 200.0, np.float32)
    tracking.paste(frame, patch, Placement(5, 5, 10, 10), feather=0)
    assert frame[9, 9].tolist() == [200, 200, 200]
    assert frame[0, 0].tolist() == [0, 0, 0]

    # Half off the frame: what fits is drawn, and nothing raises.
    tracking.paste(frame, patch, Placement(35, 35, 10, 10), feather=0)
    assert frame[39, 39].tolist() == [200, 200, 200]
    before = frame.copy()
    tracking.paste(frame, patch, Placement(80, 80, 10, 10), feather=0)   # entirely outside
    assert np.array_equal(frame, before)
