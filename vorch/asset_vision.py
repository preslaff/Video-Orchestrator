"""Vision description of project assets — Gemma proposes a label, the human confirms it.

The storyboard picks a face or a reference for each scene by reading ``Asset.label``
(``storyboard.PromptContext.storyboard_lines``), so an unlabelled asset is invisible to
per-scene assignment and a file named ``780482127_18415081828156570_n.jpeg`` is worse than
useless. This module points the existing vision client at the assets themselves.

What comes back is a *suggestion*: ``projects`` records it as ``label_source="auto"`` and the
human edits or confirms it (``"human"``), after which no captioning run overwrites it. Same
division of labour as the evaluator (NOTES.md §6): the model does the mechanical description,
the person keeps the judgement.

Output is constrained at the source with a json_schema ``response_format`` AND validated with
Pydantic, because llama.cpp grammar can "fail open" (DESIGN.md §11).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from . import media
from .clients import gemma

log = logging.getLogger("vorch.asset_vision")

# The vision prefill is the cost on CPU Gemma and it scales with image tokens.
VISION_MAX_SIDE = 768
VIDEO_FRAMES = 3


class AssetVisionError(RuntimeError):
    """Raised when Gemma cannot produce a valid AssetNote."""


class AssetNote(BaseModel):
    """What Gemma sees in one asset. Descriptive only — never a quality gate.

    Every field except ``caption`` has a default, so a model that omits half the
    schema still yields a usable note rather than failing the whole run.
    """

    caption: str = Field(
        description="ONE sentence, at most 25 words, describing what is visible. Not a paragraph.")
    subjects: list[str] = Field(
        default_factory=list,
        description="2-6 SHORT noun phrases naming the main things, e.g. ['lighthouse','stone wall','sea']. Never empty.")
    setting: str = Field(
        default="", description="A SHORT phrase of 2-4 words, e.g. 'rocky coastline', 'spiral staircase', "
                                "'harbour town'. Not a sentence.")
    people_count: int = Field(ge=0, default=0, description="How many people are visible.")
    is_portrait: bool = Field(default=False, description="Exactly one clear human face, usable as an identity reference.")
    time_of_day: Literal["dawn", "morning", "midday", "afternoon", "dusk", "night", "unknown"] = Field(
        default="unknown", description="Infer from the light; use 'unknown' only when indoors or genuinely unclear.")
    weather: str = Field(
        default="", description="1-3 words: 'overcast', 'rain', 'bright sun', 'fog'. '' if indoors.")
    text_in_image: str = Field(default="", description="Legible text on signs/plaques/labels, verbatim. '' if none.")
    landmark_guess: str = Field(default="", description="A named place ONLY if you are certain; otherwise ''.")
    landmark_confidence: int = Field(ge=0, le=10, default=0, description="0 when landmark_guess is empty.")
    tags: list[str] = Field(
        default_factory=list,
        description="3-6 short lowercase tags. Use the caller's vocabulary when it fits, and add your own.")
    quality_issues: list[str] = Field(default_factory=list, description="blurry, underexposed, screenshot, cropped…")


SYSTEM = (
    "You catalogue reference images for a video production. Describe ONLY what is visible. "
    "Never invent place names, people's names or backstory. If you are unsure, say less.\n"
    "Write for a browsable index, not for prose: `caption` is ONE sentence, and `setting`, "
    "`weather` and every entry in `subjects`/`tags` are SHORT phrases of a few words — never "
    "sentences. Fill `subjects` and `tags`; they are what the catalogue is grouped by.\n"
    "Return ONLY valid JSON matching the schema."
)

# Shown to the model so the *shape* of a good answer is unambiguous; the schema
# descriptions alone let it drift into paragraphs.
_SHAPE = (
    'Shape of a good answer:\n'
    '{"caption": "A white lighthouse with a red lantern railing stands on a rocky headland above a choppy sea.", '
    '"subjects": ["lighthouse", "stone buildings", "sea"], "setting": "rocky headland", '
    '"people_count": 0, "is_portrait": false, "time_of_day": "midday", "weather": "bright sun", '
    '"text_in_image": "", "landmark_guess": "", "landmark_confidence": 0, '
    '"tags": ["lighthouse", "exterior", "coast"], "quality_issues": []}'
)

# What to look for, per asset kind: the label is read as a menu entry by the storyboard,
# so a face has to be described as a person and a reference as a place.
_KIND_FOCUS = {
    "master_face": (
        "This is a candidate IDENTITY reference. Describe the PERSON physically: apparent age range, "
        "hair, build, distinguishing features. Set is_portrait true only if exactly one face is clearly "
        "visible and usable as a reference. Do not guess their name."
    ),
    "reference": (
        "This is a visual reference for generating new shots. Describe the setting, the light, the weather "
        "and the composition — what a prompt would need to recreate the feel of it."
    ),
    "product": (
        "This is a product reference. Describe the object, its material and surface, its scale and how it "
        "is lit and presented."
    ),
    "logo": (
        "This is a logo or mark. Describe its shapes, colours and any wordmark, and transcribe the text exactly."
    ),
}
_KIND_FOCUS["other"] = _KIND_FOCUS["reference"]


def _response_format() -> dict:
    return {"type": "json_schema",
            "json_schema": {"name": "asset_note", "schema": AssetNote.model_json_schema()}}


def _user_prompt(*, kind: str, hint: str, n_frames: int = 0) -> str:
    parts = [_KIND_FOCUS.get(kind, _KIND_FOCUS["other"])]
    if n_frames:
        parts.append(
            f"The {n_frames} images are frames sampled in order from ONE short video clip. "
            "Describe the clip as a whole, not each frame."
        )
    if hint:
        # The caller's project vocabulary. Kept out of the schema so the codebase stays
        # general-purpose: the tags are steered per run, not compiled in.
        parts.append(f"Context for this collection: {hint}\nUse that vocabulary for `tags` where it applies.")
    parts.append(_SHAPE)
    parts.append("Return JSON only.")
    return "\n\n".join(parts)


def _parse_note(raw: str) -> AssetNote:
    """Validate raw model output into an AssetNote, tolerating ```json fences."""
    try:
        return AssetNote.model_validate_json(raw)
    except ValidationError:
        # Grammar may have failed open or wrapped the JSON — clean and revalidate.
        return AssetNote.model_validate(gemma.parse_json_lenient(raw))


def _ask(images: list[Path], prompt: str, *, attempts: int, label: str,
         max_tokens: int, timeout: int) -> AssetNote:
    last_err: Exception | None = None
    rf = _response_format()
    for attempt in range(1, attempts + 1):
        raw = gemma.complete_vision(SYSTEM, prompt, list(images), response_format=rf,
                                    max_tokens=max_tokens, temperature=0.2, timeout=timeout)
        try:
            note = _parse_note(raw)
        except (ValidationError, ValueError) as e:
            last_err = e
            log.warning("describe %s: bad JSON on attempt %d/%d: %s", label, attempt, attempts, e)
            continue
        log.info("describe %s: %s", label, note.caption)
        return note
    raise AssetVisionError(f"{label}: Gemma returned no valid AssetNote in {attempts} attempts: {last_err}")


def describe_video(path: str | Path, *, hint: str = "", kind: str = "other", frames: int = VIDEO_FRAMES,
                   attempts: int = 2, max_tokens: int = 400, timeout: int = 300) -> AssetNote:
    """Describe a clip from ``frames`` evenly sampled stills (same trick as evaluate_video)."""
    stills = media.sample_frames(path, frames)
    small = [media.downscale(s, max_side=VISION_MAX_SIDE) for s in stills]
    return _ask(small, _user_prompt(kind=kind, hint=hint, n_frames=len(small)),
                attempts=attempts, label=Path(path).name, max_tokens=max_tokens, timeout=timeout)


def describe(path: str | Path, *, hint: str = "", kind: str = "other", attempts: int = 2,
             max_tokens: int = 400, timeout: int = 300) -> AssetNote:
    """Describe one asset file. Videos are dispatched to ``describe_video``."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"asset not found: {p}")
    if p.suffix.lower() in media.VIDEO_EXTS:
        return describe_video(p, hint=hint, kind=kind, attempts=attempts,
                              max_tokens=max_tokens, timeout=timeout)
    try:
        small = media.downscale(p, max_side=VISION_MAX_SIDE)
    except media.MediaError as e:
        log.warning("describe %s: could not downscale (%s); sending as is", p.name, e)
        small = p
    return _ask([small], _user_prompt(kind=kind, hint=hint), attempts=attempts,
                label=p.name, max_tokens=max_tokens, timeout=timeout)


def _truncate(text: str, max_len: int) -> str:
    """Trim to ``max_len`` on a word boundary, without a dangling separator."""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0] or text[:max_len]
    return cut.rstrip(" ,;:.-") + "…"


def _phrase(text: str, *, max_words: int = 5) -> str:
    """The first clause of ``text``, capped at ``max_words``.

    ``setting`` and ``weather`` are asked for as short phrases, but a model that
    answers with a sentence anyway must not push the real description out of a
    110-character label.
    """
    text = " ".join(text.split()).rstrip(".")
    for sep in (",", ";", " with ", " featuring ", " overlooking "):
        head = text.split(sep)[0].strip()
        if head:
            text = head
    words = text.split()
    return " ".join(words[:max_words])


def label_for(note: AssetNote, *, kind: str = "other", fallback: str = "", max_len: int = 110) -> str:
    """Build the human-facing label the storyboard reads as a menu entry.

    Faces are described as people (the storyboard picks a face by *who* it is); everything
    else leads with the setting. ``fallback`` (normally the file name) is used when the note
    carries nothing usable.
    """
    if kind == "master_face":
        bits = [note.caption]
        if not note.is_portrait:
            # Worth seeing in the picker: this file probably cannot drive an identity.
            bits.append("no clear single face")
    else:
        bits = [_phrase(note.setting), note.caption]
        if note.landmark_guess and note.landmark_confidence >= 7:
            bits.insert(0, note.landmark_guess)
    text = " — ".join(b.strip() for b in bits if b and b.strip())
    if not text:
        text = fallback.strip()
    return _truncate(text, max_len)
