"""Delivery formats: aspect presets -> Klein / Wan sizes, validation, VRAM sanity.

Klein sizes target ~0.6 MP (video-ready, 16-aligned); Wan sizes are the 480p
class the 16 GB card handles (832x480 baseline) plus a 720p class for remote
backends. Any aspect string ``W:H`` works — unknown ratios are computed and
rounded to multiples of 16.

The size/length rules are **workflow-driven**: ``workflow_rules(name)`` reads
the animate workflow's declared constraints (``4n+1`` for Wan, ``8n+1`` and
``multiple_of_32`` for LTX-2.x) so profiles and params validate against the
model they actually use, and VRAM warnings only apply to local backends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import backends, registry

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
# 720p class (Wan 2.2 fp16 on an 80 GB backend). 16-aligned; LTX-2.x needs 32-aligned
# sizes, which ``wan_size(..., multiple=32)`` computes instead (16:9 -> 1280x704).
WAN_PRESETS_720: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720), "9:16": (720, 1280), "1:1": (960, 960),
    "4:5": (720, 896), "4:3": (960, 720), "3:4": (720, 960),
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


def round_to(x: float, multiple: int = 16) -> int:
    return max(multiple, int(round(x / multiple)) * multiple)


def round16(x: float) -> int:
    return round_to(x, 16)


def keyframe_size(aspect: str, *, megapixels: float = 0.6) -> tuple[int, int]:
    if aspect in KEYFRAME_PRESETS:
        return KEYFRAME_PRESETS[aspect]
    r = ratio(aspect)
    h = math.sqrt(megapixels * 1_000_000 / r)
    return round16(h * r), round16(h)


def wan_size(aspect: str, *, short_side: int = 480, multiple: int = 16) -> tuple[int, int]:
    """Video size for ``aspect``: the 480p presets by default, the 720p presets for
    ``short_side=720``; other ratios/sides are computed and rounded to ``multiple``."""
    presets = WAN_PRESETS if short_side == 480 else WAN_PRESETS_720 if short_side == 720 else {}
    preset = presets.get(aspect)
    if preset and preset[0] % multiple == 0 and preset[1] % multiple == 0:
        return preset
    r = ratio(aspect)
    if r >= 1:
        return round_to(short_side * r, multiple), round_to(short_side, multiple)
    return round_to(short_side, multiple), round_to(short_side / r, multiple)


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


def validate_size(w: int, h: int, *, minimum: int = 256, maximum: int = 2048, multiple: int = 16) -> None:
    for name, v in (("width", w), ("height", h)):
        if not isinstance(v, int) or v % multiple != 0:
            raise ValueError(f"{name} must be a multiple of {multiple}, got {v!r}")
        if not minimum <= v <= maximum:
            raise ValueError(f"{name} must be within {minimum}..{maximum}, got {v!r}")


def validate_length(n: int, constraint: str = "4n+1") -> None:
    registry.CONSTRAINTS[constraint](n)


@dataclass(frozen=True)
class WorkflowRules:
    """What an animate workflow accepts, read from its registry YAML."""

    length_constraint: str = "4n+1"
    size_multiple: int = 16
    size_max: int = 1280
    remote: bool = False
    known: bool = True

    def short_side_default(self) -> int:
        return 720 if self.remote else 480


def workflow_rules(workflow: str | None) -> WorkflowRules:
    """Rules of ``workflow`` (an animate workflow name). Unknown names fall back to
    the Wan defaults so a profile can be validated before its workflow is installed."""
    if not workflow:
        return WorkflowRules()
    try:
        spec = registry.load_registry().get(workflow)
    except (KeyError, registry.RegistryError):
        return WorkflowRules(known=False)
    length = spec.params.get("length")
    width = spec.params.get("width")
    lc = length.constraint if length and length.constraint in registry.LENGTH_CONSTRAINTS else "4n+1"
    mult = registry.SIZE_CONSTRAINTS.get(width.constraint, 16) if width and width.constraint else 16
    size_max = int(width.max) if width and width.max else 1280
    return WorkflowRules(lc, mult, size_max, remote=spec.backend != backends.LOCAL)


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
    if spec.backend != backends.LOCAL:
        return  # a remote backend has its own GPU; the 16 GB budget is irrelevant
    w, h, n = values.get("width"), values.get("height"), values.get("length")
    if isinstance(w, int) and isinstance(h, int) and isinstance(n, int):
        for msg in vram_warnings(w, h, n):
            registry.log.warning("%s: %s", spec.name, msg)


registry.KIND_HOOKS.setdefault("animate", _animate_hook)
