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

from pydantic import BaseModel, ValidationError

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


# --- Storyboard --------------------------------------------------------------

_SYSTEM_STORYBOARD = "Ти си режисьор и сценарист. Връщаш само валиден JSON."


def generate_storyboard(
    idea: str,
    *,
    n_scenes: int | None = None,
    max_tokens: int = 1500,
    timeout: int = 300,
) -> list[SceneBrief]:
    """Break an idea into a list of scene briefs (description/character/setting/mood)."""
    n = f"около {n_scenes} сцени" if n_scenes else "толкова сцени колкото историята изисква"
    user = (
        f'Идея за видео: "{idea}"\n'
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
- Техниката/стилът ОТПРЕД (impasto, palette knife, visible brushstrokes за маслено)
- Конкретни физически детайли, НЕ абстракции
- Структура: техника → субект → действие (механично) → среда → светлина → камера
- Опиши МЕХАНИКАТА на действието (hands pressing, arms bent), не само етикет
- Течлива проза на английски. Връщаш само JSON."""

_SYSTEM_ANIMATION = """Ти пишеш промпти за Wan 2.2 image-to-video. Правила:
- Описвай ДВИЖЕНИЕ и КАМЕРА, не статичната сцена (тя вече е в keyframe-а)
- ЕДНО доминиращо движение на клип
- Завършвай с camera move + cinematic. Връщаш само JSON."""


def keyframe_prompt(
    scene: SceneBrief, *, style: str = "oil painting", timeout: int = 300
) -> PromptPair:
    """Scene brief -> FLUX.2 Klein keyframe prompt (positive + negative)."""
    user = (
        f"Сцена: {scene.model_dump_json()}\n"
        f"Стил: {style}\n"
        'Върни JSON: {"positive": "...", "negative": "smooth, digital render, 3d, airbrushed"}'
    )
    return _prompt_pair(_SYSTEM_KEYFRAME, user, max_tokens=600, timeout=timeout, label="keyframe")


def animation_prompt(scene: SceneBrief, *, timeout: int = 300) -> PromptPair:
    """Scene brief -> Wan motion prompt (positive + negative)."""
    user = (
        f"Сцена: {scene.model_dump_json()}\n"
        "Какво движение трябва (леко, локализирано работи най-добре на 16GB).\n"
        'Върни JSON: {"positive": "...", "negative": "distorted, morphing, jitter"}'
    )
    return _prompt_pair(_SYSTEM_ANIMATION, user, max_tokens=500, timeout=timeout, label="animation")


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
