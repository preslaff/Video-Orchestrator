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


# --- per-scene asset assignment (identity / references picked by the storyboard) -----------


def test_storyboard_lists_assets_and_parses_picks(monkeypatch):
    seen = {}

    def fake(system, user, **kw):
        seen["user"] = user
        return ('[{"scene": 1, "description": "child plays", "character": "Maria", "setting": "yard", '
                '"mood": "joy", "identity": "a1", "reference_ids": ["r1"]}, '
                '{"scene": 2, "description": "empty street", "identity": null}]')

    monkeypatch.setattr(storyboard.gemma, "complete_text", fake)
    ctx = storyboard.PromptContext(assets=(("a1", "master_face", "Maria, age 8"), ("a2", "master_face", "Maria, age 70"),
                                           ("r1", "reference", "yard.png")), edit_mode=True)
    briefs = storyboard.generate_storyboard("a life", n_scenes=2, context=ctx)
    assert briefs[0].identity == "a1" and briefs[0].reference_ids == ["r1"]
    assert briefs[1].identity is None and briefs[1].reference_ids == []
    assert "a1: Maria, age 8" in seen["user"] and "a2: Maria, age 70" in seen["user"] and "r1: reference: yard.png" in seen["user"]
    assert '"identity": "id на лице или null"' in seen["user"] and '"reference_ids"' in seen["user"]
    # without assets the prompt is unchanged (no identity field asked for)
    storyboard.generate_storyboard("a life", n_scenes=1, context=storyboard.PromptContext())
    assert "identity" not in seen["user"]
    # references are only offered for edit-mode jobs
    storyboard.generate_storyboard("x", n_scenes=1, context=storyboard.PromptContext(assets=(("r1", "reference", "yard"),)))
    assert "reference_ids" not in seen["user"]


def test_asset_ids_never_reach_the_image_prompts(monkeypatch):
    seen = []
    monkeypatch.setattr(storyboard.gemma, "complete_text", lambda s, u, **kw: seen.append(u) or '{"positive": "p", "negative": ""}')
    brief = storyboard.SceneBrief(scene=1, description="d", identity="a1", reference_ids=["r1"])
    storyboard.keyframe_prompt(brief)
    storyboard.animation_prompt(brief)
    assert all("a1" not in u and "reference_ids" not in u for u in seen)


# --- reference suggestions ---------------------------------------------------

from vorch.storyboard import suggest_references as _real_suggest  # noqa: E402  (imported before conftest patches it)

ASSETS = (
    ("face1", "master_face", "Bela"),
    ("stairs", "reference", "spiral stone staircase seen from below"),
    ("lens", "reference", "Fresnel lens in the lamp room"),
)
BRIEFS = [SceneBrief(scene=1, description="bike at a lighthouse"), SceneBrief(scene=2, description="climbing the spiral stairs"),
          SceneBrief(scene=3, description="the lens at dusk")]


def test_suggest_references_keeps_known_ids_and_reports_invented_ones(monkeypatch):
    seen = {}

    def fake(system, user, **kw):
        seen.update(user=user, **kw)
        return """```json
[{"scene": 1, "asset_id": null, "why": "", "fit": "motif_only"},
 {"scene": 2, "asset_id": "stairs", "why": "same staircase view", "fit": "same_shot"},
 {"scene": 3, "asset_id": "ghost", "why": "?", "fit": "same_shot"}]
```"""

    monkeypatch.setattr(storyboard.gemma, "complete_text", fake)
    picks, unknown = _real_suggest(BRIEFS, ASSETS)
    assert set(picks) == {1} and picks[1].asset_id == "stairs" and picks[1].fit == "same_shot"
    assert unknown == ["ghost"]
    assert "face1" not in seen["user"] and "stairs" in seen["user"]   # faces are not references
    assert seen["temperature"] == 0.2


def test_suggest_references_unsure_fit_falls_back_to_motif_only(monkeypatch):
    monkeypatch.setattr(storyboard.gemma, "complete_text",
                        lambda *a, **k: '[{"scene": 3, "asset_id": "lens", "why": "lens", "fit": "maybe"}, {"scene": 9, "asset_id": "lens"}, "junk"]')
    picks, unknown = _real_suggest(BRIEFS, ASSETS)
    assert list(picks) == [2] and picks[2].fit == "motif_only" and unknown == []


def test_suggest_references_skips_gemma_without_reference_assets(monkeypatch):
    monkeypatch.setattr(storyboard.gemma, "complete_text", lambda *a, **k: pytest.fail("no call expected"))
    assert _real_suggest(BRIEFS, (("face1", "master_face", "Bela"),)) == ({}, [])


def test_storyboard_budget_grows_with_scene_count(monkeypatch):
    seen = {}
    monkeypatch.setattr(storyboard.gemma, "complete_text", lambda *a, **k: seen.update(k) or STORYBOARD_JSON)
    generate_storyboard("idea", n_scenes=40)
    assert seen["max_tokens"] >= 160 * 40
