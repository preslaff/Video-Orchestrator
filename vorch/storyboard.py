"""Narrative pre-production via Gemma text mode (skills/gemma-prompt.md).

Three tasks, all returning structured JSON the orchestrator chains:
1. ``generate_storyboard`` — idea -> list of scene briefs.
2. ``keyframe_prompt`` — scene brief -> FLUX.2 Klein image prompt.
3. ``animation_prompt`` — scene brief -> Wan motion prompt.

System prompts are Bulgarian (Georgi's working language, and what the prompts
are tuned for); output is validated leniently — Gemma may wrap JSON in a fence.
Generation runs at temperature 0.7 (creative), unlike evaluation's 0.2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from . import formats
from .clients import gemma

log = logging.getLogger("vorch.storyboard")

GEN_TEMPERATURE = 0.7  # creative; evaluation uses 0.2 — don't mix them (NOTES §6)


class StoryboardError(RuntimeError):
    """Raised when Gemma returns unparseable or schema-invalid narrative JSON."""


class SceneBrief(BaseModel):
    scene: int
    description: str
    character: str = ""
    setting: str = ""
    mood: str = ""


class PromptPair(BaseModel):
    positive: str
    negative: str = ""


# --- Prompt context (format / purpose / style from the job's profile) ----------

PURPOSE_LINES = {
    "cinematic": "кинематографичен разказ: атмосфера, светлина, последователни кадри",
    "social_vertical": "вертикално социално видео (9:16): кратки, ударни сцени, субектът в центъра",
    "social_square": "квадратно социално видео (1:1): централна композиция, четимо на малък екран",
    "ad_broll": "рекламен b-roll: продукт/сцени без диалог, чисти кадри, място за текст",
    "stills": "самостоятелни кадри: всяка сцена е отделна снимка, не клип",
}
DEFAULT_KEYFRAME_NEGATIVE = "smooth, digital render, 3d, airbrushed"
DEFAULT_WAN_NEGATIVE = "distorted, morphing, jitter"


def motion_hint(clip_seconds: float) -> str:
    if clip_seconds <= 3.5:
        return "едно малко, локализирано движение (кратък клип)"
    if clip_seconds <= 6:
        return "едно доминиращо движение плюс бавно движение на камерата"
    return "едно непрекъснато движение, бавно еволюираща камера, без смяна на сцена"


@dataclass(frozen=True)
class PromptContext:
    """What the prompt writer must know about the delivery format and purpose."""

    purpose: str = "cinematic"
    kind: str = "video"
    aspect: str = "16:9"
    orientation: str = "landscape"
    framing_hint: str = formats.FRAMING_HINTS["landscape"]
    clip_seconds: float = 3.06
    style_positive: str = ""            # technique vocabulary that goes first in the prompt
    style_negative: str = ""            # Klein negative
    wan_positive: str = "cinematic"     # how the motion prompt should end
    wan_negative: str = ""              # Wan negative

    @classmethod
    def from_params(cls, params: Any) -> "PromptContext":
        return cls(
            purpose=params.purpose,
            kind=params.kind,
            aspect=params.aspect,
            orientation=params.orientation,
            framing_hint=formats.framing_hint(params.aspect),
            clip_seconds=params.clip_seconds,
            style_positive=params.style_prefix,
            style_negative=params.keyframe_negative,
            wan_positive=params.animation_suffix or "cinematic",
            wan_negative=params.animate_negative,
        )

    @property
    def motion_hint(self) -> str:
        return motion_hint(self.clip_seconds)

    def storyboard_lines(self) -> str:
        lines = [f"Предназначение: {PURPOSE_LINES.get(self.purpose, self.purpose)}",
                 f"Формат: {self.aspect} ({self.orientation}) — {self.framing_hint}"]
        if self.kind == "stills":
            lines.append("Всяка сцена е самостоятелен кадър (снимка), без движение.")
        else:
            lines.append(f"Всяка сцена е клип от ~{self.clip_seconds:g}s — едно действие на сцена.")
        return "\n".join(lines)


DEFAULT_CONTEXT = PromptContext()


def merge_negatives(*parts: str) -> str:
    """Comma-joined union of negative terms, first occurrence wins, order kept."""
    seen: list[str] = []
    for part in parts:
        for term in (part or "").split(","):
            t = term.strip()
            if t and t.lower() not in {x.lower() for x in seen}:
                seen.append(t)
    return ", ".join(seen)


# --- Storyboard --------------------------------------------------------------

_SYSTEM_STORYBOARD = "Ти си режисьор и сценарист. Връщаш само валиден JSON."


def generate_storyboard(
    idea: str,
    *,
    n_scenes: int | None = None,
    context: PromptContext | None = None,
    max_tokens: int = 1500,
    timeout: int = 300,
) -> list[SceneBrief]:
    """Break an idea into a list of scene briefs (description/character/setting/mood)."""
    ctx = context or DEFAULT_CONTEXT
    n = f"точно {n_scenes} сцени (не повече, не по-малко)" if n_scenes else "толкова сцени колкото историята изисква"
    what = "снимки" if ctx.kind == "stills" else "видео"
    user = (
        f'Идея за {what}: "{idea}"\n'
        f"{ctx.storyboard_lines()}\n"
        f"Разбий на {n}. Върни JSON масив, по един обект на сцена:\n"
        '[{"scene": 1, "description": "какво се случва (кратко, наративно)", '
        '"character": "кой герой е в кадъра", "setting": "локация", '
        '"mood": "емоция/тон"}]\n'
        "Само JSON, без друг текст."
    )
    raw = gemma.complete_text(
        _SYSTEM_STORYBOARD, user, max_tokens=max_tokens, temperature=GEN_TEMPERATURE, timeout=timeout
    )
    data = _parse(raw, "storyboard")
    if not isinstance(data, list) or not data:
        raise StoryboardError("storyboard: expected a non-empty JSON array of scenes")
    try:
        return [SceneBrief.model_validate(item) for item in data]
    except ValidationError as e:
        raise StoryboardError(f"storyboard: scene failed validation: {e}") from e


# --- Per-scene prompts -------------------------------------------------------

_SYSTEM_KEYFRAME = """Ти пишеш промпти за FLUX.2 Klein image модел. Правила:
- Техниката/стилът ОТПРЕД (напр. "impasto oil painting", "35mm film still", "phone photo, natural light")
- Конкретни физически детайли, НЕ абстракции
- Структура: техника → субект → действие (механично) → среда → светлина → камера/кадриране
- Опиши МЕХАНИКАТА на действието (hands pressing, arms bent), не само етикет
- Спазвай зададения формат/кадриране
- Течлива проза на английски. Връщаш само JSON."""

_SYSTEM_ANIMATION = """Ти пишеш промпти за Wan 2.2 image-to-video. Правила:
- Описвай ДВИЖЕНИЕ и КАМЕРА, не статичната сцена (тя вече е в keyframe-а)
- ЕДНО доминиращо движение на клип; количеството движение според дължината на клипа
- Завършвай с camera move + зададения завършек. Връщаш само JSON."""


def keyframe_prompt(
    scene: SceneBrief,
    *,
    style: str = "",
    context: PromptContext | None = None,
    timeout: int = 300,
) -> PromptPair:
    """Scene brief -> FLUX.2 Klein keyframe prompt (positive + negative).

    ``style`` (the job's free-text style) and the profile's ``style_positive``
    both go into the "technique first" slot; the profile negative is merged
    into whatever Gemma returns.
    """
    ctx = context or DEFAULT_CONTEXT
    style_line = ", ".join(x for x in (style, ctx.style_positive) if x) or "photorealistic, cinematic"
    negative = ctx.style_negative or DEFAULT_KEYFRAME_NEGATIVE
    user = (
        f"Сцена: {scene.model_dump_json()}\n"
        f"Стил/техника: {style_line}\n"
        f"Формат: {ctx.aspect} ({ctx.orientation}). Кадриране: {ctx.framing_hint}\n"
        f'Върни JSON: {{"positive": "...", "negative": "{negative}"}}'
    )
    pair = _prompt_pair(_SYSTEM_KEYFRAME, user, max_tokens=600, timeout=timeout, label="keyframe")
    return PromptPair(positive=pair.positive, negative=merge_negatives(pair.negative, ctx.style_negative))


def animation_prompt(
    scene: SceneBrief, *, context: PromptContext | None = None, timeout: int = 300
) -> PromptPair:
    """Scene brief -> Wan motion prompt (positive + negative)."""
    ctx = context or DEFAULT_CONTEXT
    negative = ctx.wan_negative or DEFAULT_WAN_NEGATIVE
    user = (
        f"Сцена: {scene.model_dump_json()}\n"
        f"Клип: ~{ctx.clip_seconds:g}s, {ctx.aspect} ({ctx.orientation}). Движение: {ctx.motion_hint}.\n"
        "Леко, локализирано движение работи най-добре на 16GB VRAM.\n"
        f"Завърши с camera move + \"{ctx.wan_positive}\".\n"
        f'Върни JSON: {{"positive": "...", "negative": "{negative}"}}'
    )
    pair = _prompt_pair(_SYSTEM_ANIMATION, user, max_tokens=500, timeout=timeout, label="animation")
    return PromptPair(positive=pair.positive, negative=merge_negatives(pair.negative, ctx.wan_negative))


# --- internals ---------------------------------------------------------------


def _prompt_pair(system: str, user: str, *, max_tokens: int, timeout: int, label: str) -> PromptPair:
    raw = gemma.complete_text(
        system, user, max_tokens=max_tokens, temperature=GEN_TEMPERATURE, timeout=timeout
    )
    data = _parse(raw, label)
    try:
        return PromptPair.model_validate(data)
    except ValidationError as e:
        raise StoryboardError(f"{label}: prompt failed validation: {e}") from e


def _parse(raw: str, label: str):
    try:
        return gemma.parse_json_lenient(raw)
    except (ValueError, IndexError) as e:
        raise StoryboardError(f"{label}: could not parse Gemma JSON: {e}\n---\n{raw[:300]}") from e
