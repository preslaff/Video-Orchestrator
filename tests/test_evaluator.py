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


def test_objective_gate_overrides_inconsistent_accept():
    """All objective checks pass but score=1 & verdict=accept -> accept, flagged."""
    raw = json.loads(ACCEPT_JSON) | {"score": 1}
    v = Verdict.model_validate(raw)
    assert v.objective_accept is True
    assert v.decision == "accept"
    assert v.needs_review is True  # accepted but score below threshold
    assert v.disagreement is False  # gemma also said accept


def test_objective_gate_overrides_optimistic_accept():
    """Gemma says accept but there are artifacts -> objective gate forces retry."""
    raw = json.loads(ACCEPT_JSON) | {"artifacts": True}
    v = Verdict.model_validate(raw)
    assert v.objective_accept is False
    assert v.decision == "retry"
    assert v.disagreement is True  # gemma said accept, gate says retry
    assert v.needs_review is False


def test_decision_retry_when_action_wrong():
    v = Verdict.model_validate_json(RETRY_JSON)
    assert v.decision == "retry" and v.objective_accept is False


def test_evaluate_image_uses_objective_decision(monkeypatch):
    """A high score with artifacts must still come back as a retry decision."""
    bad = json.dumps(json.loads(ACCEPT_JSON) | {"artifacts": True, "score": 9})
    monkeypatch.setattr(evaluator.gemma, "complete_vision", lambda *a, **k: bad)
    v = evaluate_image("fake.png", "task", attempts=1)
    assert v.decision == "retry"


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

    def fake_keyframe(prompt, *, seed, **kw):
        seen_prompts.append(prompt)
        return f"kf_{len(seen_prompts)}.png"

    verdicts = iter(
        [Verdict.model_validate_json(RETRY_JSON), Verdict.model_validate_json(ACCEPT_JSON)]
    )
    monkeypatch.setattr(pipeline, "keyframe", fake_keyframe)
    monkeypatch.setattr(pipeline, "evaluate_image", lambda path, task, **kw: next(verdicts))

    path, verdict = pipeline.generate_keyframe_evaluated(
        "a woman kneading bread", "woman kneading bread", seed=0, max_attempts=2
    )
    assert verdict.verdict == "accept"
    assert path == "kf_2.png"
    # The second attempt's prompt must carry the fix suggestion.
    assert "Correction:" in seen_prompts[1]
    assert "deformed" in seen_prompts[1]


# --- framing_ok, per-profile threshold, media dispatch -------------------------


def test_framing_ok_false_fails_gate_none_is_ignored():
    base = json.loads(ACCEPT_JSON)
    assert Verdict.model_validate(base | {"framing_ok": None}).decision == "accept"
    assert Verdict.model_validate(base).framing_ok is None  # optional in the schema
    v = Verdict.model_validate(base | {"framing_ok": False})
    assert v.decision == "retry" and v.disagreement is True
    assert Verdict.model_validate(base | {"framing_ok": True}).decision == "accept"


def test_evaluate_image_asks_for_framing_and_applies_threshold(monkeypatch):
    seen = {}

    def fake(system, user, images, **kw):
        seen["user"] = user
        return json.dumps(json.loads(ACCEPT_JSON) | {"score": 5, "framing_ok": True})

    monkeypatch.setattr(evaluator.gemma, "complete_vision", fake)
    v = evaluate_image("k.png", "task", orientation="portrait", framing_hint="headroom for captions",
                       review_threshold=6)
    assert "portrait" in seen["user"] and "headroom for captions" in seen["user"]
    assert v.needs_review is True and v.review_threshold == 6  # 5 < 6
    v2 = evaluate_image("k.png", "task")
    assert "leave null" in seen["user"] and v2.needs_review is False  # 5 >= default 4


def test_evaluate_media_dispatches_by_suffix(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(evaluator, "evaluate_image", lambda p, t, **kw: calls.append(("image", kw)) or 1)
    monkeypatch.setattr(evaluator, "evaluate_video", lambda p, t, **kw: calls.append(("video", kw)) or 2)
    assert evaluator.evaluate_media("a.png", "t", orientation="square") == 1
    assert evaluator.evaluate_media("a.mp4", "t", frames=3) == 2
    assert calls == [("image", {"orientation": "square"}), ("video", {"frames": 3})]


def test_evaluate_video_samples_frames_and_judges_clip(monkeypatch, tmp_path):
    from vorch import media

    stills = [tmp_path / f"f{i}.png" for i in range(3)]
    monkeypatch.setattr(media, "sample_frames", lambda v, n, dest_dir=None: stills[:n])
    seen = {}

    def fake(system, user, images, **kw):
        seen["user"], seen["images"] = user, images
        return ACCEPT_JSON

    monkeypatch.setattr(evaluator.gemma, "complete_vision", fake)
    v = evaluator.evaluate_video(tmp_path / "c.mp4", "she waves", frames=3)
    assert v.decision == "accept" and seen["images"] == stills
    assert "3 images are frames" in seen["user"] and "morphing" in seen["user"]
