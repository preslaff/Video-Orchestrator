"""GPU-free checks for storyboard.py — Gemma text parsed into validated briefs/prompts.

gemma.complete_text is monkeypatched so we exercise JSON parsing (incl. fences),
schema validation, and error handling without a live model.
"""

from __future__ import annotations

import pytest

from vorch import storyboard
from vorch.storyboard import (
    PromptPair,
    SceneBrief,
    StoryboardError,
    animation_prompt,
    generate_storyboard,
    keyframe_prompt,
)

STORYBOARD_JSON = """
[
  {"scene": 1, "description": "woman kneads dough", "character": "Jena", "setting": "kitchen", "mood": "calm"},
  {"scene": 2, "description": "she shapes a loaf", "character": "Jena", "setting": "kitchen", "mood": "warm"}
]
"""


def test_generate_storyboard_parses_briefs(monkeypatch):
    monkeypatch.setattr(storyboard.gemma, "complete_text", lambda *a, **k: STORYBOARD_JSON)
    briefs = generate_storyboard("a woman baking bread", n_scenes=2)
    assert [b.scene for b in briefs] == [1, 2]
    assert briefs[0].character == "Jena" and briefs[0].setting == "kitchen"
    assert all(isinstance(b, SceneBrief) for b in briefs)


def test_generate_storyboard_tolerates_json_fence(monkeypatch):
    fenced = f"```json\n{STORYBOARD_JSON}\n```"
    monkeypatch.setattr(storyboard.gemma, "complete_text", lambda *a, **k: fenced)
    briefs = generate_storyboard("idea")
    assert len(briefs) == 2


def test_generate_storyboard_rejects_non_list(monkeypatch):
    monkeypatch.setattr(storyboard.gemma, "complete_text", lambda *a, **k: '{"oops": 1}')
    with pytest.raises(StoryboardError):
        generate_storyboard("idea")


def test_generate_storyboard_raises_on_unparseable(monkeypatch):
    monkeypatch.setattr(storyboard.gemma, "complete_text", lambda *a, **k: "not json")
    with pytest.raises(StoryboardError):
        generate_storyboard("idea")


def test_keyframe_prompt_returns_pair(monkeypatch):
    monkeypatch.setattr(
        storyboard.gemma, "complete_text",
        lambda *a, **k: '{"positive": "impasto oil painting, hands in dough", "negative": "smooth"}',
    )
    brief = SceneBrief(scene=1, description="kneading", character="Jena")
    pair = keyframe_prompt(brief, style="oil painting")
    assert isinstance(pair, PromptPair)
    assert "impasto" in pair.positive and pair.negative == "smooth"


def test_animation_prompt_returns_pair(monkeypatch):
    monkeypatch.setattr(
        storyboard.gemma, "complete_text",
        lambda *a, **k: '{"positive": "slow push-in, hands kneading", "negative": "jitter"}',
    )
    pair = animation_prompt(SceneBrief(scene=1, description="kneading"))
    assert "push-in" in pair.positive


def test_prompt_pair_validation_error(monkeypatch):
    # Missing required 'positive' field.
    monkeypatch.setattr(storyboard.gemma, "complete_text", lambda *a, **k: '{"negative": "x"}')
    with pytest.raises(StoryboardError):
        keyframe_prompt(SceneBrief(scene=1, description="d"))


# --- prompt context (format / purpose / negatives) ----------------------------


def _capture(monkeypatch, reply):
    calls = []

    def fake(system, user, **kw):
        calls.append((system, user))
        return reply

    monkeypatch.setattr(storyboard.gemma, "complete_text", fake)
    return calls


def test_context_from_params_and_motion_hint():
    from vorch.genparams import GenParams

    ctx = storyboard.PromptContext.from_params(
        GenParams.for_aspect("9:16", length=81, purpose="social_vertical",
                             style_prefix="phone photo", keyframe_negative="smooth",
                             animation_suffix="handheld", animate_negative="warped"))
    assert ctx.orientation == "portrait" and "vertical" in ctx.framing_hint
    assert ctx.clip_seconds == 5.06 and "доминиращо" in ctx.motion_hint
    assert storyboard.motion_hint(2.0).startswith("едно малко") and "непрекъснато" in storyboard.motion_hint(8)
    assert "9:16" in ctx.storyboard_lines() and "5.06s" in ctx.storyboard_lines()


def test_storyboard_prompt_carries_purpose_format_and_kind(monkeypatch):
    calls = _capture(monkeypatch, STORYBOARD_JSON)
    ctx = storyboard.PromptContext(purpose="ad_broll", kind="stills", aspect="1:1", orientation="square",
                                   framing_hint="centered square")
    generate_storyboard("a product", n_scenes=2, context=ctx)
    user = calls[0][1]
    assert "рекламен b-roll" in user and "1:1" in user and "снимки" in user and "самостоятелен кадър" in user


def test_keyframe_prompt_adds_framing_and_merges_negative(monkeypatch):
    calls = _capture(monkeypatch, '{"positive": "phone photo of a woman", "negative": "blurry, smooth"}')
    ctx = storyboard.PromptContext(aspect="9:16", orientation="portrait", framing_hint="headroom for captions",
                                   style_positive="phone photo, natural light", style_negative="smooth, airbrushed")
    pair = keyframe_prompt(SceneBrief(scene=1, description="d"), style="warm tones", context=ctx)
    user = calls[0][1]
    assert "warm tones, phone photo, natural light" in user and "headroom for captions" in user
    assert '"negative": "smooth, airbrushed"' in user
    assert pair.negative == "blurry, smooth, airbrushed"  # union, order kept, no duplicates


def test_animation_prompt_uses_clip_length_and_suffix(monkeypatch):
    calls = _capture(monkeypatch, '{"positive": "slow push-in", "negative": "jitter"}')
    ctx = storyboard.PromptContext(clip_seconds=7.56, wan_positive="handheld documentary", wan_negative="warped face")
    pair = animation_prompt(SceneBrief(scene=1, description="d"), context=ctx)
    user = calls[0][1]
    assert "7.56s" in user and "непрекъснато" in user and "handheld documentary" in user
    assert pair.negative == "jitter, warped face"


def test_default_context_keeps_legacy_prompts(monkeypatch):
    calls = _capture(monkeypatch, '{"positive": "p", "negative": "n"}')
    keyframe_prompt(SceneBrief(scene=1, description="d"))
    assert "photorealistic, cinematic" in calls[0][1] and "16:9" in calls[0][1]
    assert storyboard.merge_negatives("a, b", "B , c", "") == "a, b, c"
