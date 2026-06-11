"""GPU-free checks for the evaluator — Verdict validation, parsing, retry logic.

No live Gemma server: the network call (gemma.complete_vision) is monkeypatched
so we exercise schema validation, lenient repair, and the retry-with-fix loop.
"""

import json

import pytest
from pydantic import ValidationError

from vorch import evaluator
from vorch.evaluator import EvaluationError, Verdict, evaluate_image

ACCEPT_JSON = json.dumps(
    {
        "action_correct": True,
        "identity_match": True,
        "style_match": True,
        "artifacts": False,
        "composition_ok": True,
        "score": 8,
        "verdict": "accept",
        "fix_suggestion": "",
    }
)

RETRY_JSON = json.dumps(
    {
        "action_correct": False,
        "identity_match": True,
        "style_match": True,
        "artifacts": True,
        "composition_ok": True,
        "score": 3,
        "verdict": "retry",
        "fix_suggestion": "hands are deformed; describe fingers explicitly",
    }
)


def test_verdict_validates_good_json():
    v = Verdict.model_validate_json(ACCEPT_JSON)
    assert v.verdict == "accept" and v.score == 8 and v.artifacts is False


def test_verdict_rejects_out_of_range_score():
    bad = json.loads(ACCEPT_JSON) | {"score": 99}
    with pytest.raises(ValidationError):
        Verdict.model_validate(bad)


def test_verdict_rejects_bad_enum():
    bad = json.loads(ACCEPT_JSON) | {"verdict": "maybe"}
    with pytest.raises(ValidationError):
        Verdict.model_validate(bad)


def test_parse_verdict_tolerates_json_fence():
    fenced = f"```json\n{ACCEPT_JSON}\n```"
    v = evaluator._parse_verdict(fenced)
    assert v.verdict == "accept"


def test_response_format_is_well_formed():
    rf = evaluator._response_format()
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"]["properties"]["verdict"]  # schema embedded


def test_evaluate_image_retries_then_succeeds(monkeypatch):
    """First call returns garbage, second returns valid JSON -> Verdict."""
    outputs = iter(["not json at all", ACCEPT_JSON])
    monkeypatch.setattr(
        evaluator.gemma, "complete_vision", lambda *a, **k: next(outputs)
    )
    v = evaluate_image("fake.png", "woman kneading bread", attempts=2)
    assert v.verdict == "accept"


def test_evaluate_image_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(
        evaluator.gemma, "complete_vision", lambda *a, **k: "still not json"
    )
    with pytest.raises(EvaluationError):
        evaluate_image("fake.png", "task", attempts=2)


def test_keyframe_retry_loop_feeds_fix_then_accepts(monkeypatch):
    """generate_keyframe_evaluated: retry verdict -> regenerate with the fix -> accept."""
    from vorch import pipeline

    seen_prompts = []

    def fake_keyframe(prompt, *, seed):
        seen_prompts.append(prompt)
        return f"kf_{len(seen_prompts)}.png"

    verdicts = iter(
        [Verdict.model_validate_json(RETRY_JSON), Verdict.model_validate_json(ACCEPT_JSON)]
    )
    monkeypatch.setattr(pipeline, "keyframe", fake_keyframe)
    monkeypatch.setattr(pipeline, "evaluate_image", lambda path, task: next(verdicts))

    path, verdict = pipeline.generate_keyframe_evaluated(
        "a woman kneading bread", "woman kneading bread", seed=0, max_attempts=2
    )
    assert verdict.verdict == "accept"
    assert path == "kf_2.png"
    # The second attempt's prompt must carry the fix suggestion.
    assert "Correction:" in seen_prompts[1]
    assert "deformed" in seen_prompts[1]
