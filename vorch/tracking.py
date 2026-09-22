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
    """Where a region sits in one frame: integer pixels, top-left anchored."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    @property
    def scale(self) -> float:
        return 1.0

    def clamped(self, width: int, height: int) -> "Placement":
        return Placement(max(0, min(self.x, width - 1)), max(0, min(self.y, height - 1)),
                         self.w, self.h, self.confidence)


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
    # How well the region explains this frame where it already is. Anything else has to beat
    # it by a margin, so a locked-off shot stays put instead of shivering on noise.
    held_score = -1.0
    if 0 <= previous.x and 0 <= previous.y and previous.x + previous.w <= fw and previous.y + previous.h <= fh:
        tpl = resample(template, previous.h, previous.w)
        t = tpl - tpl.mean()
        tn = float(np.sqrt((t * t).sum()))
        if tn > 1e-6:
            held_score = _ncc(frame[previous.y:previous.y + previous.h,
                                    previous.x:previous.x + previous.w], t, tn)

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


def feather_mask(h: int, w: int, feather: int) -> np.ndarray:
    """A 0..1 alpha for an h x w patch, fading ``feather`` px in from every edge.

    The pasted rectangle must not announce itself with a hard edge, which on a glass bottle
    reads as a sticker rather than a label.
    """
    if feather <= 0:
        return np.ones((h, w), np.float32)
    def ramp(n: int) -> np.ndarray:
        a = np.ones(n, np.float32)
        f = min(feather, max(1, n // 2))
        edge = (np.arange(f) + 0.5) / f
        a[:f] = edge
        a[n - f:] = edge[::-1]
        return a
    return ramp(h)[:, None] * ramp(w)[None, :]


def paste(frame: np.ndarray, patch: np.ndarray, at: Placement, *, feather: int = 4) -> None:
    """Alpha-blend ``patch`` into ``frame`` (both HxWx3) at ``at``, in place, clipped to the
    frame so a region leaving the edge fades out instead of raising."""
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, at.x), max(0, at.y)
    x1, y1 = min(fw, at.x + at.w), min(fh, at.y + at.h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = patch[y0 - at.y:y1 - at.y, x0 - at.x:x1 - at.x]
    alpha = feather_mask(at.h, at.w, feather)[y0 - at.y:y1 - at.y, x0 - at.x:x1 - at.x][..., None]
    region = frame[y0:y1, x0:x1]
    region[:] = region * (1.0 - alpha) + sub * alpha
