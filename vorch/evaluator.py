"""Vision evaluation — Gemma scores a generated result and decides accept/retry.

The evaluator is the heart of the self-evaluating loop (NOTES.md §6). It asks
Gemma (vision) to judge an image against the scene's intent and return a strict,
typed ``Verdict``. Output is constrained at the source via a json_schema
``response_format`` AND validated externally with Pydantic — because llama.cpp
grammar can "fail open" (DESIGN.md §11), the Pydantic check is the real guard.

Objective fields (action_correct / identity_match / artifacts) are reliable and
safe to automate; subjective ones (score, style) are weaker signals — treat as a
guide, flag for human review rather than hard-blocking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, ValidationError

import requests

from .clients import gemma

log = logging.getLogger("vorch.evaluator")

# An accepted frame whose (weak, subjective) score is below this is flagged for
# human review — not blocked. SKILL.md invariant 7 / Georgi's working style.
REVIEW_SCORE_THRESHOLD = 4


class Verdict(BaseModel):
    """Structured judgement of one generated image (NOTES.md §6).

    The ``verdict`` field is Gemma's *suggestion*; the authoritative accept/retry
    is ``decision``, derived from the reliable objective booleans (Gemma's verdict
    and score can contradict each other — e.g. all checks pass yet score=1). Score
    stays advisory: a low score on an accepted frame raises ``needs_review``, it
    does not block.
    """

    action_correct: bool = Field(description="Does the subject perform the described action?")
    identity_match: bool = Field(description="Correct character? true if identity is not applicable.")
    style_match: bool = Field(description="Does the style match (e.g. oil painting)?")
    artifacts: bool = Field(description="Are there deformations/morphing/extra fingers? true = BAD.")
    composition_ok: bool = Field(description="Is the framing/balance good?")
    framing_ok: bool | None = Field(
        default=None,
        description="Does the image respect the requested delivery format/framing? null if not asked.",
    )
    score: int = Field(ge=0, le=10, description="Overall quality 1-10 (10 = flawless, 5 = average). Never 0.")
    verdict: Literal["accept", "retry"]
    fix_suggestion: str = Field(default="", description="If retry: what to change in the prompt.")

    # Per-profile review threshold (set by evaluate_image; not part of the schema).
    _review_threshold: int = PrivateAttr(default=REVIEW_SCORE_THRESHOLD)
    # Nobody looked at this one: the evaluator was unreachable and the result was kept anyway.
    _unevaluated: str = PrivateAttr(default="")

    @classmethod
    def not_evaluated(cls, why: str) -> "Verdict":
        """A stand-in verdict for "the evaluator was down".

        Losing a rendered frame — ten minutes of GPU for a clip — because the *judge* is
        offline is the wrong trade: the objective checks pass by default, the result is kept,
        and ``needs_review`` flags it so a person looks at what nothing judged.
        """
        v = cls(action_correct=True, identity_match=True, style_match=True, artifacts=False,
                composition_ok=True, score=5, verdict="accept", fix_suggestion="")
        v._unevaluated = why[:200]
        return v

    @property
    def review_threshold(self) -> int:
        return self._review_threshold

    @property
    def objective_accept(self) -> bool:
        """The reliable signal: right action, right identity, framed, no artifacts.

        ``framing_ok`` counts only when it was asked for (False fails; None is ignored).
        """
        return (
            self.action_correct
            and self.identity_match
            and self.composition_ok
            and not self.artifacts
            and self.framing_ok is not False
        )

    @property
    def decision(self) -> Literal["accept", "retry"]:
        """Authoritative accept/retry — the objective gate, not Gemma's free verdict."""
        return "accept" if self.objective_accept else "retry"

    @property
    def unevaluated(self) -> str:
        """Why nothing judged this result, when nothing did ("" = it was evaluated)."""
        return self._unevaluated

    @property
    def needs_review(self) -> bool:
        """Accepted on objective grounds but the advisory score is low — or never judged."""
        if self._unevaluated:
            return True
        return self.objective_accept and self.score < self._review_threshold

    @property
    def disagreement(self) -> bool:
        """Gemma's own verdict contradicts the objective gate."""
        return self.verdict != self.decision


class EvaluationError(RuntimeError):
    """Raised when Gemma's output cannot be coerced into a valid Verdict."""


SYSTEM = (
    "You are a strict art director. Judge the image against the goal. "
    "Return ONLY valid JSON matching the schema — no prose, no markdown."
)


def _user_prompt(task_description: str, *, orientation: str | None = None, framing_hint: str = "",
                 with_identity: bool = False) -> str:
    text = (
        f'Goal of the frame: "{task_description}"\n\n'
        + ("The FIRST image is the generated frame; the SECOND image is the reference identity "
           "(the person who must appear). Set identity_match to true only if the person in the "
           "first image is recognisably the same person as in the second.\n\n" if with_identity else "")
        + "Evaluate it and return the JSON verdict. Set verdict to \"accept\" only if the "
        "action is correct, the right character is shown, and there are no artifacts; "
        "otherwise \"retry\" with a concrete fix_suggestion for the prompt.\n"
        "score: overall quality on a 1-10 scale (10 = flawless, 7 = usable, 5 = average, "
        "1-3 = poor). Always give a score; never 0."
    )
    if orientation:
        text += (
            f"\n\nDelivery format: {orientation} frame. Expected framing: {framing_hint or orientation}. "
            "Set framing_ok to true only if the subject placement and headroom suit that format "
            "(nothing essential cut off or crammed at the edges); otherwise false."
        )
    else:
        text += "\n\nframing_ok: leave null (no format requirement)."
    return text


def _response_format(*, require_framing: bool = False) -> dict:
    """llama.cpp-compatible json_schema response_format derived from Verdict.

    The grammar lets the model omit optional fields, so when a framing judgement
    is requested ``framing_ok`` is made a required boolean (no null branch);
    ``score`` keeps its 0-10 bounds for validation but the prompt asks for 1-10.
    """
    schema = Verdict.model_json_schema()
    if require_framing:
        schema["properties"]["framing_ok"] = {
            "type": "boolean",
            "description": "Does the image respect the requested delivery format/framing?",
        }
        required = list(schema.get("required", []))
        if "framing_ok" not in required:
            required.append("framing_ok")
        schema["required"] = required
    return {"type": "json_schema", "json_schema": {"name": "verdict", "schema": schema}}


def _parse_verdict(raw: str) -> Verdict:
    """Validate raw model output into a Verdict, tolerating ```json fences."""
    try:
        return Verdict.model_validate_json(raw)
    except ValidationError:
        # Grammar may have failed open or wrapped the JSON — clean and revalidate.
        return Verdict.model_validate(gemma.parse_json_lenient(raw))


def evaluate_image(
    image_path: str | Path,
    task_description: str,
    *,
    attempts: int = 2,
    orientation: str | None = None,
    framing_hint: str = "",
    review_threshold: int = REVIEW_SCORE_THRESHOLD,
    identity_image: str | Path | None = None,
) -> Verdict:
    """Evaluate one keyframe; return a validated Verdict (retries on bad JSON).

    ``orientation`` (landscape/portrait/square) + ``framing_hint`` make Gemma
    judge ``framing_ok`` for the delivery format; ``review_threshold`` is the
    profile's advisory score floor for ``needs_review``; ``identity_image`` (the
    scene's face) is attached so ``identity_match`` is judged against it.
    """
    images = [image_path] + ([identity_image] if identity_image else [])
    return _evaluate(
        images, _user_prompt(task_description, orientation=orientation, framing_hint=framing_hint,
                             with_identity=bool(identity_image)),
        attempts=attempts, review_threshold=review_threshold, label=Path(image_path).name,
        require_framing=orientation is not None,
    )


def evaluate_video(
    video_path: str | Path,
    task_description: str,
    *,
    frames: int = 5,
    attempts: int = 2,
    orientation: str | None = None,
    framing_hint: str = "",
    review_threshold: int = REVIEW_SCORE_THRESHOLD,
) -> Verdict:
    """Evaluate a clip from ``frames`` evenly sampled stills (experimental).

    Objective checks (action happened, identity kept, no morphing between the
    frames) are the reliable part; temporal smoothness is not judged reliably
    from stills — treat the score as advisory (skills/gemma-evaluate.md).
    """
    from .media import sample_frames

    stills = sample_frames(video_path, frames)
    prompt = _user_prompt(task_description, orientation=orientation, framing_hint=framing_hint)
    prompt += (
        f"\n\nThese {len(stills)} images are frames sampled in order from one short clip. "
        "Judge the clip: action_correct = the described motion visibly happens across the "
        "frames; artifacts = morphing, warping or identity drift between frames."
    )
    return _evaluate(stills, prompt, attempts=attempts, review_threshold=review_threshold,
                     label=Path(video_path).name, require_framing=orientation is not None)


def evaluate_media(path: str | Path, task_description: str, **kw) -> Verdict:
    """Dispatch on the file type: video clips go to ``evaluate_video``, stills to ``evaluate_image``."""
    from .registry import VIDEO_EXT

    if Path(path).suffix.lower() in VIDEO_EXT:
        return evaluate_video(path, task_description, **kw)
    return evaluate_image(path, task_description, **kw)


def _evaluate(
    images: list, prompt: str, *, attempts: int, review_threshold: int, label: str,
    require_framing: bool = False,
) -> Verdict:
    last_err: Exception | None = None
    rf = _response_format(require_framing=require_framing)
    for attempt in range(1, attempts + 1):
        try:
            raw = gemma.complete_vision(SYSTEM, prompt, images, response_format=rf)
        except (requests.RequestException, gemma.GemmaError) as e:
            # The judge is down (llama-server stopped, or restarting). Keep the generated
            # result, flag it for review, and let the run continue: a 40-scene job must not
            # fail — and lose the frame it just rendered — because the evaluator went away.
            log.warning("eval %s: evaluator unavailable (%s); keeping the result for review", label, e)
            return Verdict.not_evaluated(f"{type(e).__name__}: {e}")
        try:
            verdict = _parse_verdict(raw)
            verdict._review_threshold = review_threshold
            name = label
            log.info(
                "eval %s: %s (score=%s)%s",
                name,
                verdict.decision,
                verdict.score,
                "" if verdict.decision == "accept" else f" — fix: {verdict.fix_suggestion}",
            )
            if verdict.disagreement:
                log.info(
                    "eval %s: gemma said %r; objective gate -> %r",
                    name, verdict.verdict, verdict.decision,
                )
            if verdict.needs_review:
                log.warning(
                    "eval %s: accepted but score=%d (<%d) — flagged for review",
                    name, verdict.score, review_threshold,
                )
            return verdict
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            last_err = e
            log.warning("eval parse failed (attempt %d/%d): %s", attempt, attempts, e)
    raise EvaluationError(
        f"Gemma did not return a valid Verdict after {attempts} attempts"
    ) from last_err
