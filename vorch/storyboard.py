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
    identity: str | None = None           # project asset id of the face for this scene (None = no swap)
    reference_ids: list[str] = []         # project asset ids the keyframe should follow (edit mode)


PROMPT_EXCLUDE = {"identity", "reference_ids"}  # asset ids never reach the image/motion prompts


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


def call_timeout(max_tokens: int, prompt: str = "") -> int:
    """HTTP timeout for one non-streaming Gemma call, sized to the work asked for.

    llama-server sends nothing until the answer is complete, so the read timeout must cover the
    whole generation. CPU Gemma measured ~15 tok/s generating and ~45 tok/s reading the prompt;
    budget half those speeds (a busy machine) plus two minutes. A 40-scene storyboard asks for
    ~6600 tokens — about seven minutes at 15 tok/s, far past a flat 300 s.
    """
    prompt_tokens = len(prompt) // 3
    return max(300, int(120 + max_tokens / 7.5 + prompt_tokens / 22))


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
    assets: tuple[tuple[str, str, str], ...] = ()   # (asset id, kind, label) the storyboard may assign
    edit_mode: bool = False             # job keyframes are reference-locked edits -> pick references too

    @classmethod
    def from_params(cls, params: Any, *, assets: tuple[tuple[str, str, str], ...] = (),
                    edit_mode: bool = False) -> "PromptContext":
        return cls(
            assets=tuple(assets),
            edit_mode=edit_mode,
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
        faces = [(i, lbl) for i, k, lbl in self.assets if k == "master_face"]
        refs = [(i, k, lbl) for i, k, lbl in self.assets if k != "master_face"]
        if faces:
            lines.append("Налични лица (identity) за face swap — id: описание: "
                         + "; ".join(f"{i}: {lbl}" for i, lbl in faces))
            lines.append('За всяка сцена посочи "identity": id-то на лицето, което съответства на героя и момента '
                         'от историята (възраст, период), или null ако в кадъра няма герой или е друг човек.')
        if refs and self.edit_mode:
            lines.append("Налични референции (композиция/продукти) — id: вид: описание: "
                         + "; ".join(f"{i}: {k}: {lbl}" for i, k, lbl in refs))
            lines.append('За всяка сцена посочи "reference_ids": списък от id-та на референциите, които кадърът да следва (може празен).')
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
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> list[SceneBrief]:
    """Break an idea into a list of scene briefs (description/character/setting/mood).

    The token budget grows with the scene count (~160 tokens a scene): a flat 1500 cut a
    40-scene storyboard off mid-array.
    """
    ctx = context or DEFAULT_CONTEXT
    if max_tokens is None:
        max_tokens = max(1500, 160 * n_scenes + 200) if n_scenes else 3000
    n = f"точно {n_scenes} сцени (не повече, не по-малко)" if n_scenes else "толкова сцени колкото историята изисква"
    what = "снимки" if ctx.kind == "stills" else "видео"
    user = (
        f'Идея за {what}: "{idea}"\n'
        f"{ctx.storyboard_lines()}\n"
        f"Разбий на {n}. Върни JSON масив, по един обект на сцена:\n"
        '[{"scene": 1, "description": "какво се случва (кратко, наративно)", '
        '"character": "кой герой е в кадъра", "setting": "локация", '
        '"mood": "емоция/тон"'
        + (', "identity": "id на лице или null"' if any(k == "master_face" for _, k, _ in ctx.assets) else "")
        + (', "reference_ids": ["id", ...]' if ctx.edit_mode and any(k != "master_face" for _, k, _ in ctx.assets) else "")
        + "}]\n"
        "Само JSON, без друг текст."
    )
    raw = gemma.complete_text(
        _SYSTEM_STORYBOARD, user, max_tokens=max_tokens, temperature=GEN_TEMPERATURE,
        timeout=timeout or call_timeout(max_tokens, _SYSTEM_STORYBOARD + user),
    )
    data = _parse(raw, "storyboard")
    if not isinstance(data, list) or not data:
        raise StoryboardError("storyboard: expected a non-empty JSON array of scenes")
    try:
        return [SceneBrief.model_validate(item) for item in data]
    except ValidationError as e:
        raise StoryboardError(f"storyboard: scene failed validation: {e}") from e


# --- Reference suggestions ----------------------------------------------------

FITS = ("same_shot", "motif_only")

_SYSTEM_SUGGEST = """Ти си фоторедактор. Съпоставяш сцени от сценарий със снимки от архива на автора.
- Предлагай снимка само ако показва същия субект/място като сцената; иначе null.
- "fit": "same_shot" — снимката е практически същият кадър (субект, гледна точка, кадриране);
  "motif_only" — същият субект, но различно кадриране или момент.
- Една снимка на сцена, най-подходящата. Не измисляй id-та. Връщаш само JSON."""


class ReferencePick(BaseModel):
    scene: int
    asset_id: str | None = None
    why: str = ""
    fit: str = "motif_only"


def suggest_references(
    briefs: list[SceneBrief],
    assets: tuple[tuple[str, str, str], ...],
    *,
    timeout: int | None = None,
) -> tuple[dict[int, ReferencePick], list[str]]:
    """Suggest, per scene, the project photo that best matches it — by the assets' labels.

    A separate pass after the storyboard, so the storyboard prompt stays small and a failure
    here costs only the hints. Returns ``({scene position (0-based): pick}, unknown ids)``;
    picks naming an id that is not in ``assets`` are dropped and reported, never trusted.
    Matching is a judgement, not creative writing: temperature 0.2, like evaluation.
    """
    refs = [(i, k, lbl) for i, k, lbl in assets if k != "master_face"]
    if not refs or not briefs:
        return {}, []
    catalogue = "\n".join(f"- {i} ({k}): {lbl}" for i, k, lbl in refs)
    scenes = "\n".join(
        f"{n}. {b.description}" + (f" | място: {b.setting}" if b.setting else "")
        for n, b in enumerate(briefs, start=1)
    )
    user = (
        f"Снимки (id (вид): описание):\n{catalogue}\n\n"
        f"Сцени:\n{scenes}\n\n"
        "За всяка сцена върни JSON масив, по един обект на сцена:\n"
        '[{"scene": 1, "asset_id": "id или null", "why": "кратко, на английски: какво съвпада", '
        '"fit": "same_shot | motif_only"}]\n'
        "Само JSON, без друг текст."
    )
    budget = 90 * len(briefs) + 200
    raw = gemma.complete_text(
        _SYSTEM_SUGGEST, user, max_tokens=budget, temperature=0.2,
        timeout=timeout or call_timeout(budget, _SYSTEM_SUGGEST + user),
    )
    data = _parse(raw, "reference suggestions")
    if not isinstance(data, list):
        raise StoryboardError("reference suggestions: expected a JSON array")
    known = {i for i, _, _ in refs}
    picks: dict[int, ReferencePick] = {}
    unknown: list[str] = []
    for item in data:
        try:
            pick = ReferencePick.model_validate(item)
        except ValidationError:
            continue  # one malformed entry should not cost the other scenes their hints
        pos = pick.scene - 1
        if not pick.asset_id or not 0 <= pos < len(briefs) or pos in picks:
            continue
        if pick.asset_id not in known:
            unknown.append(pick.asset_id)
            continue
        if pick.fit not in FITS:
            pick.fit = "motif_only"   # when unsure, the hint that does not lock the composition
        picks[pos] = pick
    return picks, unknown


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
        f"Сцена: {scene.model_dump_json(exclude=PROMPT_EXCLUDE)}\n"
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
        f"Сцена: {scene.model_dump_json(exclude=PROMPT_EXCLUDE)}\n"
        f"Клип: ~{ctx.clip_seconds:g}s, {ctx.aspect} ({ctx.orientation}). Движение: {ctx.motion_hint}.\n"
        "Леко, локализирано движение работи най-добре на 16GB VRAM.\n"
        f"Завърши с camera move + \"{ctx.wan_positive}\".\n"
        f'Върни JSON: {{"positive": "...", "negative": "{negative}"}}'
    )
    pair = _prompt_pair(_SYSTEM_ANIMATION, user, max_tokens=500, timeout=timeout, label="animation")
    return PromptPair(positive=pair.positive, negative=merge_negatives(pair.negative, ctx.wan_negative))


# A video model redraws every frame from an 8x-compressed latent: printed type below roughly
# 32 px of cap height has no representation there and comes back as a plausible smear. Asking
# the camera to *approach* a label therefore guarantees it will be destroyed — and left to
# itself Gemma writes "slow zoom-in towards the product label" for every product shot, which is
# the single most reliable way to ruin a brand name. Movement goes to the surroundings instead.
TEXT_SAFE_RULE = (
    "ВАЖНО: ако в кадъра има печатен текст (етикет, опаковка, надпис), той трябва да остане "
    "абсолютно неподвижен: без приближаване/отдалечаване на камерата към него, без въртене или "
    "мащабиране на този обект. Движението е в средата около него — листа, венчелистчета, прах, "
    "течност, светлина, сенки. Камерата е статична или се движи много леко странично.\n"
)


def motion_from_image(
    image_path: str | Path,
    *,
    direction: str = "",
    context: PromptContext | None = None,
    timeout: int | None = None,
) -> PromptPair:
    """Write a Wan motion prompt by **looking at the frame**, not at a description of it.

    For keyframes made elsewhere there is no storyboard to write the motion from, and a one-line
    caption loses what a clip needs: where the light falls, what could plausibly move, where the
    camera has room. ``direction`` is the job's one-line intent ("intimate, slow, warm"), which
    sets the tone without inventing content that is not in the picture.
    """
    ctx = context or DEFAULT_CONTEXT
    negative = ctx.wan_negative or DEFAULT_WAN_NEGATIVE
    user = (
        "Виж кадъра и опиши как да се анимира.\n"
        + (f"Обща посока от режисьора: {direction}\n" if direction.strip() else "")
        + f"Клип: ~{ctx.clip_seconds:g}s, {ctx.aspect} ({ctx.orientation}). Движение: {ctx.motion_hint}.\n"
        "Опиши САМО движение и камера — това, което вече е в кадъра, нищо ново. "
        "Леко, локализирано движение работи най-добре на 16GB VRAM.\n"
        + TEXT_SAFE_RULE
        + f'Завърши с camera move + "{ctx.wan_positive}".\n'
        f'Върни JSON: {{"positive": "...", "negative": "{negative}"}}'
    )
    raw = gemma.complete_vision(
        _SYSTEM_ANIMATION, user, [image_path], max_tokens=500, temperature=GEN_TEMPERATURE,
        timeout=timeout or call_timeout(500, _SYSTEM_ANIMATION + user),
    )
    data = _parse(raw, "animation (vision)")
    try:
        pair = PromptPair.model_validate(data)
    except ValidationError as e:
        raise StoryboardError(f"animation (vision): prompt failed validation: {e}") from e
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
