"""Follow a locked region through a moving shot.

A brand label pasted at a fixed rectangle is only right while the camera holds still. The
moment a shot dollies, the bottle grows and moves and a fixed paste slides off it — which
makes locking useless for exactly the shots advertising wants most.

So find the region again in every frame. The marked object is rigid and planar, the motion
between neighbouring frames is small, and the plate is high-contrast against the product, so a
**similarity search** — uniform scale plus translation, seeded from the previous frame — is
both sufficient and cheap: a few hundred normalised correlations per frame, no optical flow,
no model. Rotation and perspective are deliberately not estimated; a product hero shot does
not roll, and a wrong warp is worse than none.

The search is local by construction, so it cannot jump to a false match elsewhere in the
frame; when nothing correlates (the object left the shot, a cut) the placement simply holds,
and ``Track.confidence`` says so.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("vorch.tracking")

# One frame to the next, a camera move is small. These bound the per-frame search: a scale
# step wide enough for a fast push-in (~3%/frame is a 10x zoom over 81 frames) and a shift
# wide enough for a handheld drift, but narrow enough that the search stays local and quick.
SCALE_STEPS = (0.99, 0.995, 1.0, 1.005, 1.01, 1.02, 1.03)
SEARCH_SPAN = 14        # px, either side of the predicted position
COARSE_STEP = 2         # px between coarse candidates; the best is then refined to 1 px
MIN_CONFIDENCE = 0.25   # below this the frame is not trusted and the placement holds
# A region only moves when moving clearly explains the frame better. Without this margin,
# image noise alone tips the winner by a pixel on a locked-off shot and the paste shivers.
STICKINESS = 0.004


@dataclass(frozen=True)
class Placement:
    """Where a region sits in one frame, top-left anchored.

    The search produces whole pixels; ``smooth`` then produces fractional ones, because a
    rectangle that can only land on integers has to step, and a stepping label shimmers.
    """

    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def clamped(self, width: int, height: int) -> "Placement":
        return Placement(max(0, min(self.x, width - 1)), max(0, min(self.y, height - 1)),
                         self.w, self.h, self.confidence)


def _moving_average(a: np.ndarray, window: int) -> np.ndarray:
    """Centred mean over ``window`` samples, with the ends held rather than fading to zero."""
    if window < 3 or a.size < 3:
        return a
    window = min(window, a.size if a.size % 2 else a.size - 1)
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(a, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def smooth(track: Sequence[Placement], *, min_confidence: float = MIN_CONFIDENCE,
           window: int = 9) -> list[Placement]:
    """Turn a per-frame search into the trajectory a camera actually follows.

    Two things make a tracked paste flicker, and neither is the search being wrong. A frame
    the search could not trust holds its predecessor, so the region freezes and then lurches
    to catch up; and an integer rectangle can only grow in whole pixels, so it advances in
    visible steps. A real camera move is smooth, so: drop the untrusted frames and interpolate
    across them, then low-pass what remains, in centre-and-scale rather than corner-and-size so
    the region stays put while it grows.
    """
    n = len(track)
    if n < 2:
        return list(track)
    w0, h0 = track[0].w, track[0].h
    if w0 <= 0 or h0 <= 0:
        return list(track)
    cx = np.array([p.centre[0] for p in track], float)
    cy = np.array([p.centre[1] for p in track], float)
    scale = np.array([p.w / w0 for p in track], float)
    trusted = np.array([p.confidence >= min_confidence for p in track])
    if not trusted.any():
        return list(track)
    idx = np.arange(n, dtype=float)
    for a in (cx, cy, scale):
        a[~trusted] = np.interp(idx[~trusted], idx[trusted], a[trusted])
    cx, cy, scale = (_moving_average(a, window) for a in (cx, cy, scale))
    out = []
    for i, p in enumerate(track):
        w, h = scale[i] * w0, scale[i] * h0
        out.append(Placement(cx[i] - w / 2, cy[i] - h / 2, w, h, p.confidence))
    return out


def resample(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize. Good enough to *score* a match; the pixels that end up in
    the frame are resampled once, from the highest-resolution source there is."""
    sh, sw = a.shape[:2]
    yi = np.clip((np.arange(h) * (sh / h)).astype(np.intp), 0, sh - 1)
    xi = np.clip((np.arange(w) * (sw / w)).astype(np.intp), 0, sw - 1)
    return a[yi][:, xi]


def _ncc(window: np.ndarray, tpl_centred: np.ndarray, tpl_norm: float) -> float:
    """Normalised cross-correlation of one window against a pre-centred template.

    Normalised, so a shot that darkens or brightens between frames does not drag the match.
    """
    wc = window - window.mean()
    denom = float(np.sqrt((wc * wc).sum())) * tpl_norm
    if denom <= 1e-6:
        return -1.0
    return float((wc * tpl_centred).sum() / denom)


def _best_offset(frame: np.ndarray, tpl: np.ndarray, cx: int, cy: int,
                 *, span: int, step: int) -> tuple[float, int, int]:
    """Best (score, x, y) for ``tpl`` in a window around (cx, cy)."""
    th, tw = tpl.shape
    fh, fw = frame.shape
    t = tpl - tpl.mean()
    tn = float(np.sqrt((t * t).sum()))
    if tn <= 1e-6:                       # a featureless patch cannot be tracked
        return -1.0, cx, cy
    best = (-1.0, cx, cy)
    for dy in range(-span, span + 1, step):
        for dx in range(-span, span + 1, step):
            x, y = cx + dx, cy + dy
            if x < 0 or y < 0 or x + tw > fw or y + th > fh:
                continue
            s = _ncc(frame[y:y + th, x:x + tw], t, tn)
            if s > best[0]:
                best = (s, x, y)
    return best


def follow(template: np.ndarray, frame: np.ndarray, previous: Placement) -> Placement:
    """Where ``template`` sits in ``frame``, given where it sat in the frame before.

    ``template`` and ``frame`` are 2-D float arrays (greyscale). The region is searched at a
    few scales around the previous one — the object grows as the camera pushes in — and about
    its own centre, which is how a rectangle on a dollying object actually moves. A frame that
    correlates worse than ``MIN_CONFIDENCE`` keeps the previous placement rather than jumping.
    """
    fh, fw = frame.shape
    # The search indexes arrays, so it works in whole pixels; only `smooth` is fractional.
    px, py = int(round(previous.x)), int(round(previous.y))
    pw, ph = max(1, int(round(previous.w))), max(1, int(round(previous.h)))
    previous = Placement(px, py, pw, ph, previous.confidence)
    # How well the region explains this frame where it already is. Anything else has to beat
    # it by a margin, so a locked-off shot stays put instead of shivering on noise.
    held_score = -1.0
    if 0 <= px and 0 <= py and px + pw <= fw and py + ph <= fh:
        tpl = resample(template, ph, pw)
        t = tpl - tpl.mean()
        tn = float(np.sqrt((t * t).sum()))
        if tn > 1e-6:
            held_score = _ncc(frame[py:py + ph, px:px + pw], t, tn)

    best: tuple[float, Placement] | None = None
    for m in SCALE_STEPS:
        h, w = int(round(previous.h * m)), int(round(previous.w * m))
        if h < 4 or w < 4 or h > fh or w > fw:
            continue
        tpl = resample(template, h, w)
        # The patch grows about its centre, so its top-left moves by half the growth.
        cx = previous.x - (w - previous.w) // 2
        cy = previous.y - (h - previous.h) // 2
        score, x, y = _best_offset(frame, tpl, cx, cy, span=SEARCH_SPAN, step=COARSE_STEP)
        if score < 0:
            continue
        # Refine the winner to single-pixel accuracy: a 2 px coarse grid would otherwise
        # let the paste shiver by a pixel from frame to frame.
        score, x, y = _best_offset(frame, tpl, x, y, span=COARSE_STEP - 1, step=1)
        if best is None or score > best[0]:
            best = (score, Placement(x, y, w, h, score))
    moved_enough = best is not None and best[0] > held_score + STICKINESS
    if best is None or best[0] < MIN_CONFIDENCE or not moved_enough:
        held = Placement(previous.x, previous.y, previous.w, previous.h,
                         max(held_score, best[0] if best else 0.0))
        return held.clamped(fw, fh)
    return best[1].clamped(fw, fh)


def bilinear(patch: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Sample ``patch`` (HxWxC) at fractional coordinates — the grid ``ys`` x ``xs``.

    Nearest-neighbour is fine for *scoring* a match but not for the pixels that reach the
    screen: as a tracked patch grows a fraction of a pixel, nearest-neighbour swaps whole rows
    of the source in and out and the lettering crawls.
    """
    ph, pw = patch.shape[:2]
    ys = np.clip(ys, 0, ph - 1)
    xs = np.clip(xs, 0, pw - 1)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, ph - 1)
    x1 = np.minimum(x0 + 1, pw - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = patch[np.ix_(y0, x0)] * (1 - wx) + patch[np.ix_(y0, x1)] * wx
    bottom = patch[np.ix_(y1, x0)] * (1 - wx) + patch[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bottom * wy


def _integral(a: np.ndarray, axis: int) -> np.ndarray:
    """Running sum along ``axis`` with a leading zero, for exact area integration."""
    a = np.moveaxis(a, axis, 0)
    zero = np.zeros((1,) + a.shape[1:], np.float32)
    return np.concatenate([zero, np.cumsum(a, axis=0, dtype=np.float32)])


def _integrate(a: np.ndarray, edges: np.ndarray, axis: int) -> np.ndarray:
    """Mean of ``a`` over each interval of ``edges``, along ``axis``.

    Linear interpolation into the running sum gives the exact area under a box-pixel model,
    which is what makes this stable: as an interval grows by a fraction of a pixel the mean
    changes by a fraction, where a point sample would swap a whole row in or out.
    """
    integral = _integral(a, axis)
    n = integral.shape[0] - 1

    def at(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, 0, n)
        i0 = np.floor(e).astype(np.intp)
        i1 = np.minimum(i0 + 1, n)
        t = (e - i0).reshape((-1,) + (1,) * (integral.ndim - 1))
        return integral[i0] * (1 - t) + integral[i1] * t

    lo, hi = edges[:-1], edges[1:]
    width = np.maximum(hi - lo, 1e-6).reshape((-1,) + (1,) * (a.ndim - 1))
    return np.moveaxis((at(hi) - at(lo)) / width, 0, axis)


def area_resample(patch: np.ndarray, y_edges: np.ndarray, x_edges: np.ndarray) -> np.ndarray:
    """Resample ``patch`` by averaging it over each output pixel's footprint.

    Point sampling — bilinear included — reconstructs a *different* set of source pixels each
    time the scale shifts a fraction, so fine lettering crawls as a tracked region grows. An
    area average integrates instead, so the output changes as smoothly as the scale does.
    """
    return _integrate(_integrate(patch, y_edges, 0), x_edges, 1)


def _edge_alpha(centres: np.ndarray, lo: float, hi: float, feather: float) -> np.ndarray:
    """A 0..1 ramp along one axis: fading in over ``feather`` px after ``lo`` and out before
    ``hi``. Computed from the pixel centres, so it moves smoothly with a fractional edge.

    The ramp is never narrower than one pixel, even when no feather was asked for: a pixel the
    rectangle only half covers must only be half painted, or the boundary jumps a whole pixel
    each time the edge crosses an integer — the same stutter the fractional placement exists
    to remove.
    """
    span = max(1.0, min(feather, (hi - lo) / 2))
    return np.clip((centres - lo) / span, 0, 1) * np.clip((hi - centres) / span, 0, 1)


def paste(frame: np.ndarray, patch: np.ndarray, at: Placement, *, feather: float = 4) -> None:
    """Alpha-blend ``patch`` into ``frame`` (both HxWx3) at ``at``, in place.

    ``at`` may be fractional: the patch is resampled straight onto the destination pixel grid,
    so scale and sub-pixel position are one operation and neither quantises. Clipped to the
    frame, so a region leaving the edge simply draws what fits.
    """
    fh, fw = frame.shape[:2]
    if at.w <= 0 or at.h <= 0:
        return
    x0, y0 = max(0, int(np.floor(at.x))), max(0, int(np.floor(at.y)))
    x1, y1 = min(fw, int(np.ceil(at.x + at.w))), min(fh, int(np.ceil(at.y + at.h)))
    if x1 <= x0 or y1 <= y0:
        return
    ph, pw = patch.shape[:2]
    xc = np.arange(x0, x1, dtype=float) + 0.5
    yc = np.arange(y0, y1, dtype=float) + 0.5
    # Each destination pixel covers one unit of the frame; map its two edges into the patch and
    # average what lies between them. ``patch`` is expected to be at least as large as the
    # placement (see ``media._lock_tracked``), so this is always a minification.
    sub = area_resample(patch,
                        (np.arange(y0, y1 + 1, dtype=float) - at.y) / at.h * ph,
                        (np.arange(x0, x1 + 1, dtype=float) - at.x) / at.w * pw)
    alpha = (_edge_alpha(yc, at.y, at.y + at.h, feather)[:, None]
             * _edge_alpha(xc, at.x, at.x + at.w, feather)[None, :])[..., None]
    region = frame[y0:y1, x0:x1]
    region[:] = region * (1.0 - alpha) + sub * alpha
