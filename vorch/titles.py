"""Text over the picture — the ASS a caption compiles to.

Pure: text items, an output size and the second each one starts at go in; the body of an
``.ass`` file comes out. No ffmpeg, no filesystem, no knowledge of scenes or jobs — so the two
rules that are easy to get wrong (escaping and colour) have exactly one home, and the whole thing
is checkable without a GPU.

**Why text belongs here and not in a prompt.** A video model cannot hold a brand name: the VAE
compresses 8× spatially and type below roughly 32 px of cap height does not survive it — the
reason ``Scene.locked_regions`` and ``docs/label-fidelity.md`` exist. Text composited at assembly
time never passes through a generator, so it is pixel-exact at output resolution.

Three things this module exists to get right, each of which failed in testing first:

**Colour is ``&HBBGGRR``**, not RGB. A gold written as ``D8B26A`` renders blue unless the bytes are
reversed. :func:`ass_colour` is the only place allowed to format one.

**Sizes are fractions of the frame, never pixels.** ``PlayRes`` must equal the real output size, and
a 9:16 job is 480×832 while a 16:9 one is 832×480 — the same pixel size is a caption in one and a
headline in the other. Everything here scales off height (or width, for horizontal margins).

**A centred typewriter jitters.** Each revealed character re-centres the line, so the text crawls
leftwards as it types. :func:`build` forces a left anchor for that effect rather than letting a
person discover it in a render.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import NamedTuple

#: A ceiling on how many captions one list may hold. Not a style rule — a pathological spec could
#: otherwise emit a ten-thousand-event script, and a typewriter multiplies each item by its length.
MAX_ITEMS = 50
#: Long text reads badly as a caption and, typed out character by character, is an event explosion.
MAX_CHARS = 500
MAX_TYPEWRITER_CHARS = 80

#: What an item's ``effect`` may be. Exposed by the assembly status like ``media.transitions()``,
#: so the UI is a dropdown rather than a free-text field nobody can spell.
EFFECTS: tuple[str, ...] = ("none", "fade", "slide", "pop", "typewriter")

#: Starting values the editor copies into a new item. Nothing refers back to a preset afterwards —
#: an item carries its whole appearance, so changing a preset never moves text that already exists.
PRESETS: dict[str, dict] = {
    "title":       {"size": 0.075, "align": 5, "bold": True, "effect": "pop",
                    "outline_width": 0.005},
    "lower-third": {"size": 0.048, "align": 2, "bold": True, "effect": "slide",
                    "margin_y": 0.10},
    "kicker":      {"size": 0.030, "align": 8, "bold": True, "effect": "fade",
                    "colour": "D8B26A", "margin_y": 0.06},
    "credits":     {"size": 0.036, "align": 2, "bold": False, "effect": "fade",
                    "margin_y": 0.08},
}

_HEX = re.compile(r"^[0-9A-Fa-f]{6}$")
#: ASS numpad anchors whose horizontal part is "centre" or "right", and the left one to use instead
#: when a typewriter needs to grow rightwards from a fixed point.
_LEFT_OF = {1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4, 7: 7, 8: 7, 9: 7}


@dataclass(frozen=True)
class TextItem:
    """One caption. Self-describing: every appearance value lives on the item.

    ``start`` and ``duration`` are seconds — within the clip for a scene's text, on the finished
    timeline for the job's. Resolving one into the other is the caller's job (see
    ``assembly._texts_for``); this module only ever sees the resolved times.
    """

    text: str
    start: float = 0.0
    duration: float = 2.0
    effect: str = "fade"
    effect_seconds: float = 0.35
    font: str = "DejaVu Sans"
    size: float = 0.048            # fraction of frame height
    colour: str = "FFFFFF"         # RGB as a person writes it
    outline: str = "202020"
    outline_width: float = 0.004   # fraction of frame height
    shadow: float = 0.0            # fraction of frame height
    bold: bool = True
    italic: bool = False
    align: int = 2                 # ASS numpad anchor, 1-9
    margin_x: float = 0.08         # fraction of frame width
    margin_y: float = 0.08         # fraction of frame height
    angle: float = 0.0
    opacity: float = 1.0

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_dict(self) -> dict:
        return {
            "text": self.text, "start": self.start, "duration": self.duration,
            "effect": self.effect, "effect_seconds": self.effect_seconds,
            "font": self.font, "size": self.size, "colour": self.colour,
            "outline": self.outline, "outline_width": self.outline_width, "shadow": self.shadow,
            "bold": self.bold, "italic": self.italic, "align": self.align,
            "margin_x": self.margin_x, "margin_y": self.margin_y,
            "angle": self.angle, "opacity": self.opacity,
        }


def from_dict(d: dict) -> TextItem:
    """One item from stored JSON. Every field optional, so an old job loads unchanged."""
    def num(key: str, default: float) -> float:
        v = d.get(key, default)
        try:
            return float(v if v is not None else default)
        except (TypeError, ValueError):
            raise ValueError(f"text {key}: expected a number, got {v!r}") from None

    return TextItem(
        text=str(d.get("text", "") or ""),
        start=num("start", 0.0), duration=num("duration", 2.0),
        effect=str(d.get("effect") or "fade").strip().lower(),
        effect_seconds=num("effect_seconds", 0.35),
        font=str(d.get("font") or "DejaVu Sans"),
        size=num("size", 0.048),
        colour=str(d.get("colour") or "FFFFFF").lstrip("#").upper(),
        outline=str(d.get("outline") or "202020").lstrip("#").upper(),
        outline_width=num("outline_width", 0.004), shadow=num("shadow", 0.0),
        bold=bool(d.get("bold", True)), italic=bool(d.get("italic", False)),
        align=int(d.get("align", 2) or 2),
        margin_x=num("margin_x", 0.08), margin_y=num("margin_y", 0.08),
        angle=num("angle", 0.0), opacity=num("opacity", 1.0),
    )


def validate(item: TextItem, *, limit: float | None = None) -> None:
    """Refuse an item that cannot be drawn. ``ValueError``, like ``media.Region``.

    ``limit`` is how long the thing it sits on lasts — the clip for a scene's text, the timeline for
    the job's — so a caption cannot be scheduled past the end of what it is written over.
    """
    if not item.text.strip():
        raise ValueError("text: nothing to draw")
    if item.effect not in EFFECTS:
        raise ValueError(f"text effect {item.effect!r} unknown (want one of {EFFECTS})")
    if item.start < 0:
        raise ValueError(f"text start {item.start:g}s: cannot be negative")
    if item.duration <= 0:
        raise ValueError(f"text duration {item.duration:g}s: must be positive")
    if not 0 < item.size <= 1:
        raise ValueError(f"text size {item.size:g}: a fraction of the frame height, 0-1")
    if item.align not in range(1, 10):
        raise ValueError(f"text align {item.align}: an ASS anchor, 1-9")
    if not 0 <= item.opacity <= 1:
        raise ValueError(f"text opacity {item.opacity:g}: 0-1")
    if len(item.text) > MAX_CHARS:
        raise ValueError(f"text is {len(item.text)} characters: at most {MAX_CHARS}")
    if item.effect == "typewriter" and len(item.text) > MAX_TYPEWRITER_CHARS:
        raise ValueError(
            f"typewriter text is {len(item.text)} characters: at most {MAX_TYPEWRITER_CHARS} "
            "(one event per character, and it reads badly longer than that)")
    for name, value in (("colour", item.colour), ("outline", item.outline)):
        if not _HEX.match(value):
            raise ValueError(f"text {name} {value!r}: six hex digits, e.g. D8B26A")
    for name, value in (("margin_x", item.margin_x), ("margin_y", item.margin_y)):
        if not 0 <= value < 0.5:
            raise ValueError(f"text {name} {value:g}: a fraction of the frame, 0-0.5")
    if limit is not None and item.start >= limit:
        raise ValueError(f"text starts at {item.start:g}s, past the end ({limit:.2f}s)")


class Placed(NamedTuple):
    """An item with its times already resolved onto the finished timeline."""

    item: TextItem
    start: float
    duration: float


# --- the two rules nothing else may implement ----------------------------------------


def escape(text: str) -> str:
    """Text as an ASS event field.

    ``{`` and ``}`` delimit override blocks, so a literal brace has to be escaped or everything
    between two of them silently disappears. A newline becomes a hard break. Commas need nothing —
    Text is the last field on the line and takes the rest of it.
    """
    out = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return out.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\N")


def ass_colour(rgb: str, opacity: float = 1.0) -> str:
    """``&HAABBGGRR`` from RGB hex — the byte order that turns gold into blue when forgotten.

    ``AA`` is *transparency*, not alpha: 00 is opaque and FF is invisible.
    """
    v = rgb.lstrip("#").upper()
    if not _HEX.match(v):
        raise ValueError(f"colour {rgb!r}: six hex digits, e.g. D8B26A")
    alpha = min(255, max(0, round((1.0 - min(1.0, max(0.0, opacity))) * 255)))
    return f"&H{alpha:02X}{v[4:6]}{v[2:4]}{v[0:2]}"


# --- building ------------------------------------------------------------------------


def _ts(seconds: float) -> str:
    """ASS timestamp: ``H:MM:SS.cc``, centisecond resolution."""
    s = max(0.0, seconds)
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{int(h):d}:{int(m):02d}:{sec:05.2f}"


def _anchor(item: TextItem, width: int, height: int, align: int | None = None) -> tuple[int, int]:
    """Where the text's own anchor point sits, from its alignment and margins.

    ``\\pos``/``\\move`` place the anchor *named by the alignment* — a bottom-centred item is
    positioned by its bottom centre — so a position can be computed without measuring the text,
    which is what keeps this module free of font metrics.
    """
    a = item.align if align is None else align
    col, row = (a - 1) % 3, (a - 1) // 3          # 0,1,2 = left/centre/right and bottom/middle/top
    mx, my = item.margin_x * width, item.margin_y * height
    x = {0: mx, 1: width / 2, 2: width - mx}[col]
    y = {0: height - my, 1: height / 2, 2: my}[row]
    return round(x), round(y)


def _style_line(name: str, item: TextItem, height: int, width: int, align: int) -> str:
    size = max(8, round(item.size * height))
    outline = round(item.outline_width * height, 1)
    shadow = round(item.shadow * height, 1)
    return (
        f"Style: {name},{item.font},{size},"
        f"{ass_colour(item.colour, item.opacity)},{ass_colour(item.colour, item.opacity)},"
        f"{ass_colour(item.outline, item.opacity)},&H96000000,"
        f"{-1 if item.bold else 0},{-1 if item.italic else 0},0,0,100,100,0,{item.angle:g},"
        f"1,{outline:g},{shadow:g},{align},"
        f"{round(item.margin_x * width)},{round(item.margin_x * width)},"
        f"{round(item.margin_y * height)},1"
    )


def _overrides(item: TextItem, start: float, duration: float,
               width: int, height: int, align: int) -> str:
    """The ``{\\...}`` block that gives an item its effect."""
    fx = max(0.0, min(item.effect_seconds, duration / 2))
    ms = round(fx * 1000)
    if item.effect == "none":
        return ""
    if item.effect == "fade":
        return f"{{\\fad({ms},{ms})}}"
    if item.effect == "pop":
        return f"{{\\fad({ms},{ms})\\fscx60\\fscy60\\t(0,{ms},\\fscx100\\fscy100)}}"
    if item.effect == "slide":
        x, y = _anchor(item, width, height, align)
        rise = round(height * 0.06)
        return f"{{\\move({x},{y + rise},{x},{y},0,{ms})\\fad(0,{ms})}}"
    if item.effect == "typewriter":
        # Pinned, because each revealed character would otherwise re-centre the whole line.
        x, y = _anchor(item, width, height, align)
        return f"{{\\pos({x},{y})\\fad(0,{ms})}}"
    return ""


def _events(name: str, item: TextItem, start: float, duration: float,
            width: int, height: int, align: int) -> list[str]:
    tag = _overrides(item, start, duration, width, height, align)
    end = start + duration
    if item.effect != "typewriter":
        return [f"Dialogue: 0,{_ts(start)},{_ts(end)},{name},,0,0,0,,{tag}{escape(item.text)}"]
    # One event per revealed character: the reveal spreads over effect_seconds (never more than
    # the item itself), and the last event holds the finished line until the end.
    chars = list(item.text)
    span = max(0.0, min(item.effect_seconds, duration))
    step = (span / len(chars)) if chars and span else 0.0
    out: list[str] = []
    for i in range(1, len(chars) + 1):
        t0 = start + step * (i - 1)
        t1 = end if i == len(chars) else min(end, start + step * i)
        if t1 <= t0:
            continue
        out.append(f"Dialogue: 0,{_ts(t0)},{_ts(t1)},{name},,0,0,0,,{tag}{escape(''.join(chars[:i]))}")
    return out


def build(placed: list[Placed], *, width: int, height: int) -> str:
    """The body of an ``.ass`` file for every caption on the finished timeline.

    ``PlayRes`` is the real output size: libass scales its own coordinate space to the frame, so a
    mismatch moves everything and resizes the type. Callers take the size from
    ``assembly.target_format``, never from a clip — a delivery target can change it.
    """
    lines = [
        "[Script Info]",
        "; generated by vorch — edit the job, not this file",
        "ScriptType: v4.00+",
        f"PlayResX: {int(width)}",
        f"PlayResY: {int(height)}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    events = [
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for n, p in enumerate(placed):
        item = p.item
        align = _LEFT_OF[item.align] if item.effect == "typewriter" else item.align
        name = f"t{n}"
        lines.append(_style_line(name, item, height, width, align))
        events += _events(name, item, p.start, p.duration, width, height, align)
    return "\n".join(lines + events) + "\n"


__all__ = [
    "EFFECTS", "MAX_CHARS", "MAX_ITEMS", "MAX_TYPEWRITER_CHARS", "PRESETS", "Placed",
    "TextItem",
    "ass_colour", "build", "escape", "from_dict", "replace", "validate",
]
