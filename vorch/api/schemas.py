"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel


class JobSpec(BaseModel):
    """Request body for creating a job (mirrors the YAML job spec / DESIGN §7)."""

    idea: str
    master_face: str = ""  # required only when face_swap is on
    style: str = ""
    autonomy: str = "semi"
    n_scenes: int | None = None
    seed: int = 0
    max_attempts: int = 2
    evaluate: bool = True
    face_swap: bool = True


class SceneEdit(BaseModel):
    """One scene's editable fields (storyboard gate review). Index is required."""

    index: int
    description: str | None = None
    keyframe_prompt: str | None = None
    animation_prompt: str | None = None


class SceneEdits(BaseModel):
    scenes: list[SceneEdit]
