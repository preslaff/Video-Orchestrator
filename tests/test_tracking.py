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
    frame = np.zeros((40, 40, 3), np.float32)
    patch = np.full((20, 20, 3), 200.0, np.float32)
    tracking.paste(frame, patch, Placement(10, 10, 20, 20), feather=4)
    assert frame[20, 20, 0] == pytest.approx(200.0)     # solid in the middle
    assert 0 < frame[10, 10, 0] < 60                    # and nearly transparent at the corner
    assert frame[9, 9, 0] == 0                          # nothing outside the rectangle

    # With no feather the inside is solid, but the boundary pixel is still only painted as
    # far as the rectangle covers it — here a quarter, being a corner.
    plain = np.zeros((40, 40, 3), np.float32)
    tracking.paste(plain, patch, Placement(10, 10, 20, 20), feather=0)
    assert plain[12, 12, 0] == pytest.approx(200.0)
    assert plain[10, 10, 0] == pytest.approx(50.0)
    assert plain[10, 15, 0] == pytest.approx(100.0)     # an edge pixel: half covered


def test_a_fractional_placement_moves_the_patch_by_less_than_a_pixel():
    """Whole-pixel steps are what a tracked label shimmers on, so the paste must be able to
    land between pixels: a half-pixel shift has to change the picture, smoothly."""
    patch = np.zeros((8, 8, 3), np.float32)
    patch[:, 4:] = 255.0                                 # a hard vertical edge to look at
    edges = []
    for offset in (0.0, 0.25, 0.5, 0.75):
        frame = np.zeros((20, 20, 3), np.float32)
        tracking.paste(frame, patch, Placement(6 + offset, 6, 8, 8), feather=0)
        edges.append(float(frame[10, 10, 0]))
    assert len(set(edges)) > 2                           # each sub-pixel step is distinct
    assert edges == sorted(edges, reverse=True) or edges == sorted(edges)


def test_a_patch_that_grows_a_fraction_of_a_pixel_moves_its_edge_a_fraction():
    """Growing in whole pixels is the other half of the shimmer: the edge has to travel
    smoothly, not jump a whole column at a time."""
    patch = np.zeros((8, 8, 3), np.float32)
    patch[:, 4:] = 255.0

    def edge_at(width: float) -> float:
        """Where the black/white boundary lands, to sub-pixel precision."""
        frame = np.zeros((40, 40, 3), np.float32)
        tracking.paste(frame, patch, Placement(5, 5, width, 8), feather=0)
        row = frame[8, :, 0]
        return float(row.sum() / 255.0)                  # total white = the edge's position

    widths = [16, 16.25, 16.5, 16.75, 17]
    found = [edge_at(w) for w in widths]
    assert all(b > a for a, b in zip(found, found[1:])), f"the edge must creep, not jump: {found}"
    assert max(b - a for a, b in zip(found, found[1:])) < 0.75


def test_smoothing_fills_in_the_frames_the_search_could_not_trust():
    """An untrusted frame holds its predecessor, so the region freezes and then lurches to
    catch up — which is exactly what flickers. Interpolate across the gap instead."""
    track = [Placement(100 - 2 * i, 100, 60 + 4 * i, 40, 0.9) for i in range(11)]
    # Frames 4-7 lost the match: the search held them where frame 3 was.
    for i in range(4, 8):
        track[i] = Placement(track[3].x, track[3].y, track[3].w, track[3].h, 0.05)
    out = tracking.smooth(track, window=3)

    widths = [p.w for p in out]
    assert all(b >= a for a, b in zip(widths, widths[1:])), "the push-in must not go backwards"
    assert widths[4] > widths[3] and widths[7] < widths[8]      # the frozen run now advances
    assert out[0].w == pytest.approx(60, abs=2) and out[-1].w == pytest.approx(100, abs=2)


def test_smoothing_takes_the_steps_out_of_a_quantised_track():
    """A whole-pixel search advances in bursts — 0, 0, 3, 0, 4. On screen that is a stutter."""
    steps = [0, 0, 3, 0, 0, 4, 0, 3, 0, 0, 4, 0, 3, 0, 4]
    track, w = [], 60.0
    for i, d in enumerate(steps):
        w += d
        track.append(Placement(100 - i, 100, w, w * 2 / 3, 0.9))
    out = tracking.smooth(track, window=7)
    deltas = [b.w - a.w for a, b in zip(out, out[1:])]
    assert max(deltas) - min(deltas) < max(steps) - min(steps)   # the bursts are evened out
    assert any(0 < d < 1 for d in deltas), "the trajectory should move by fractions of a pixel"


def test_smoothing_keeps_a_region_centred_while_it_grows():
    """Smoothing centre-and-scale, not corner-and-size: a label growing about its middle must
    not creep sideways."""
    track = [Placement(100 - 2 * i, 100 - i, 60 + 4 * i, 40 + 2 * i, 0.9) for i in range(9)]
    out = tracking.smooth(track, window=5)
    centres = [p.centre[0] for p in out]
    assert all(abs(c - 130) < 1.5 for c in centres)


def test_smoothing_leaves_a_track_it_cannot_help_alone():
    assert tracking.smooth([]) == []
    one = [Placement(1, 2, 3, 4, 0.9)]
    assert tracking.smooth(one) == one
    lost = [Placement(1, 2, 3, 4, 0.0) for _ in range(5)]     # nothing was ever trusted
    assert tracking.smooth(lost) == lost


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
