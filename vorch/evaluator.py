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

from pydantic import BaseModel, Field, ValidationError

from .clients import gemma

log = logging.getLogger("vorch.evaluator")


class Verdict(BaseModel):
    """Structured judgement of one generated image (NOTES.md §6)."""

    action_correct: bool = Field(description="Does the subject perform the described action?")
    identity_match: bool = Field(description="Correct character? true if identity is not applicable.")
    style_match: bool = Field(description="Does the style match (e.g. oil painting)?")
    artifacts: bool = Field(description="Are there deformations/morphing/extra fingers? true = BAD.")
    composition_ok: bool = Field(description="Is the framing/balance good?")
    score: int = Field(ge=0, le=10, description="Overall quality 0-10.")
    verdict: Literal["accept", "retry"]
    fix_suggestion: str = Field(default="", description="If retry: what to change in the prompt.")


class EvaluationError(RuntimeError):
    """Raised when Gemma's output cannot be coerced into a valid Verdict."""


SYSTEM = (
    "You are a strict art director. Judge the image against the goal. "
    "Return ONLY valid JSON matching the schema — no prose, no markdown."
)


def _user_prompt(task_description: str) -> str:
    return (
        f'Goal of the frame: "{task_description}"\n\n'
        "Evaluate it and return the JSON verdict. Set verdict to \"accept\" only if the "
        "action is correct, the right character is shown, and there are no artifacts; "
        "otherwise \"retry\" with a concrete fix_suggestion for the prompt."
    )


def _response_format() -> dict:
    """llama.cpp-compatible json_schema response_format derived from Verdict."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "schema": Verdict.model_json_schema()},
    }


def _parse_verdict(raw: str) -> Verdict:
    """Validate raw model output into a Verdict, tolerating ```json fences."""
    try:
        return Verdict.model_validate_json(raw)
    except ValidationError:
        # Grammar may have failed open or wrapped the JSON — clean and revalidate.
        return Verdict.model_validate(gemma.parse_json_lenient(raw))


def evaluate_image(
    image_path: str | Path, task_description: str, *, attempts: int = 2
) -> Verdict:
    """Evaluate one keyframe; return a validated Verdict (retries on bad JSON)."""
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        raw = gemma.complete_vision(
            SYSTEM,
            _user_prompt(task_description),
            [image_path],
            response_format=_response_format(),
        )
        try:
            verdict = _parse_verdict(raw)
            log.info(
                "eval %s: %s (score=%s)%s",
                Path(image_path).name,
                verdict.verdict,
                verdict.score,
                "" if verdict.verdict == "accept" else f" — fix: {verdict.fix_suggestion}",
            )
            return verdict
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            last_err = e
            log.warning("eval parse failed (attempt %d/%d): %s", attempt, attempts, e)
    raise EvaluationError(
        f"Gemma did not return a valid Verdict after {attempts} attempts"
    ) from last_err
