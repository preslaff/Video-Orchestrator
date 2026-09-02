"""Delivery formats: aspect presets -> Klein / Wan sizes, validation, VRAM sanity.

Klein sizes target ~0.6 MP (video-ready, 16-aligned); Wan sizes are the 480p
class the 16 GB card handles (832x480 baseline). Any aspect string ``W:H``
works — unknown ratios are computed and rounded to multiples of 16.
"""

from __future__ import annotations

import math

from . import registry

ASPECTS: dict[str, tuple[int, int]] = {
    "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "4:3": (4, 3), "3:4": (3, 4),
}
KEYFRAME_PRESETS: dict[str, tuple[int, int]] = {
    "16:9": (1024, 576), "9:16": (576, 1024), "1:1": (768, 768),
    "4:5": (704, 880), "4:3": (896, 672), "3:4": (672, 896),
}
WAN_PRESETS: dict[str, tuple[int, int]] = {
    "16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640),
    "4:5": (576, 720), "4:3": (640, 480), "3:4": (480, 640),
}
FRAMING_HINTS = {
    "portrait": "vertical 9:16 framing, subject centered, headroom at the top for captions, nothing important at the edges",
    "landscape": "wide cinematic framing, subject on a third, room for lateral camera motion",
    "square": "centered square composition, subject fills the middle",
}

_BASELINE_PIXEL_FRAMES = 832 * 480 * 49  # the tested 16 GB budget


def parse_aspect(aspect: str) -> tuple[int, int]:
    if aspect in ASPECTS:
        return ASPECTS[aspect]
    try:
        w, h = aspect.split(":")
        w, h = int(w), int(h)
    except (ValueError, AttributeError):
        raise ValueError(f"aspect must look like '16:9', got {aspect!r}") from None
    if w <= 0 or h <= 0:
        raise ValueError(f"aspect must be positive, got {aspect!r}")
    return w, h


def ratio(aspect: str) -> float:
    w, h = parse_aspect(aspect)
    return w / h


def round16(x: float) -> int:
    return max(16, int(round(x / 16.0)) * 16)


def keyframe_size(aspect: str, *, megapixels: float = 0.6) -> tuple[int, int]:
    if aspect in KEYFRAME_PRESETS:
        return KEYFRAME_PRESETS[aspect]
    r = ratio(aspect)
    h = math.sqrt(megapixels * 1_000_000 / r)
    return round16(h * r), round16(h)


def wan_size(aspect: str, *, short_side: int = 480) -> tuple[int, int]:
    if aspect in WAN_PRESETS:
        return WAN_PRESETS[aspect]
    r = ratio(aspect)
    if r >= 1:
        return round16(short_side * r), round16(short_side)
    return round16(short_side), round16(short_side / r)


def orientation(aspect: str) -> str:
    r = ratio(aspect)
    if abs(r - 1.0) < 1e-6:
        return "square"
    return "landscape" if r > 1 else "portrait"


def framing_hint(aspect: str) -> str:
    return FRAMING_HINTS[orientation(aspect)]


def size_ratio(w: int, h: int) -> float:
    return w / h


def aspects_match(a: tuple[int, int], b: tuple[int, int], *, tolerance: float = 0.05) -> bool:
    """Same orientation/ratio within ``tolerance`` (16-alignment skews 9:16 pairs by ~2.5%)."""
    ra, rb = size_ratio(*a), size_ratio(*b)
    return abs(ra - rb) / max(ra, rb) <= tolerance


def validate_size(w: int, h: int, *, minimum: int = 256, maximum: int = 2048) -> None:
    for name, v in (("width", w), ("height", h)):
        if not isinstance(v, int) or v % 16 != 0:
            raise ValueError(f"{name} must be a multiple of 16, got {v!r}")
        if not minimum <= v <= maximum:
            raise ValueError(f"{name} must be within {minimum}..{maximum}, got {v!r}")


def validate_length(n: int) -> None:
    registry.CONSTRAINTS["4n+1"](n)


def clip_seconds(length: int, fps: int) -> float:
    return round(length / fps, 2) if fps else 0.0


def vram_warnings(w: int, h: int, length: int) -> list[str]:
    """Soft warnings when a Wan run is well above the tested 832x480x49 budget."""
    r = (w * h * length) / _BASELINE_PIXEL_FRAMES
    out: list[str] = []
    if r > 2.0:
        out.append(
            f"{w}x{h} x {length} frames is {r:.1f}x the tested 16 GB budget; "
            "expect RAM offload, long runs or OOM"
        )
    elif r > 1.5 and h > w:
        out.append(
            f"portrait {w}x{h} x {length} frames is {r:.1f}x the tested budget; watch VRAM"
        )
    return out


def _animate_hook(spec: registry.WorkflowSpec, values: dict) -> None:
    w, h, n = values.get("width"), values.get("height"), values.get("length")
    if isinstance(w, int) and isinstance(h, int) and isinstance(n, int):
        for msg in vram_warnings(w, h, n):
            registry.log.warning("%s: %s", spec.name, msg)


registry.KIND_HOOKS.setdefault("animate", _animate_hook)
