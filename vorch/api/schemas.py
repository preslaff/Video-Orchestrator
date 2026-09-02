"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


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
    keyframe_negative: str | None = None
    animation_negative: str | None = None
    mode: Literal["t2i", "edit"] | None = None
    references: list[str] | None = None


class SceneEdits(BaseModel):
    scenes: list[SceneEdit]


# --- responses ---------------------------------------------------------------

Gate = Literal["storyboard", "keyframes"]


class VerdictOut(BaseModel):
    decision: str            # authoritative accept|retry (objective gate)
    needs_review: bool
    verdict: str             # the model's raw verdict (may disagree)
    score: int
    action_correct: bool
    identity_match: bool
    style_match: bool
    artifacts: bool
    composition_ok: bool
    framing_ok: bool | None = None
    fix_suggestion: str = ""


class VariantOut(BaseModel):
    path: str
    url: str | None = None
    seed: int | None = None
    verdict: VerdictOut | None = None
    swapped_path: str | None = None
    swapped_url: str | None = None
    post_paths: list[str] = []


class ScenePublic(BaseModel):
    index: int
    description: str = ""
    character: str = ""
    setting: str = ""
    mood: str = ""
    continuous: bool = False
    status: str
    keyframe_prompt: str = ""
    animation_prompt: str = ""
    keyframe_negative: str = ""
    animation_negative: str = ""
    mode: str = "t2i"
    references: list[str] = []
    keyframe_path: str | None = None
    keyframe_url: str | None = None
    keyframe_verdict: VerdictOut | None = None
    video_path: str | None = None
    video_url: str | None = None
    video_verdict: VerdictOut | None = None
    swapped_path: str | None = None
    swapped_url: str | None = None
    swap_verdict: VerdictOut | None = None
    variants: list[VariantOut] = []
    raw_video_path: str | None = None
    post_paths: list[str] = []


class JobPublic(BaseModel):
    id: str
    idea: str
    style: str
    master_face: str
    autonomy: str
    n_scenes: int | None
    seed: int
    max_attempts: int
    evaluate: bool
    face_swap: bool
    mode: str = "t2i"
    reference_images: list[str] = []
    project_id: str | None = None
    profile_id: str | None = None
    profile: dict | None = None
    kind: str = "video"
    export: dict | None = None
    status: str
    created_at: str
    updated_at: str
    scenes: list[ScenePublic]
    pending_gate: Gate | None = None


class JobSummaryOut(BaseModel):
    id: str
    status: str
    idea: str
    updated_at: str
    project_id: str | None = None
    kind: str = "video"
    profile_id: str | None = None


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class DecisionOut(BaseModel):
    scene_index: int | None
    phase: str
    attempt: int
    action: str
    score: int | None
    reason: str
    created_at: str


class ProvenanceOut(BaseModel):
    scene_index: int
    kind: Literal["keyframe", "swapped", "video"]
    filename: str
    url: str | None
    seed: int | None
    prompt: str | None
    template: str | None
    has_manifest: bool


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
