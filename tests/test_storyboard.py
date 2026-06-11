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
