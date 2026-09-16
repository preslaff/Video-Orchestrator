"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class JobSpec(BaseModel):
    """Request body for creating a job.

    The profile (``profile_id`` → project default → built-in default) carries
    the generation settings; ``profile_overrides`` deep-merges on top; the
    legacy ``evaluate`` / ``face_swap`` / ``max_attempts`` knobs apply only when
    sent. ``force`` bypasses a failing environment check.
    """

    idea: str
    style: str = ""
    autonomy: Literal["semi", "full"] = "semi"
    n_scenes: int | None = None
    seed: int = 0
    project_id: str | None = None
    profile_id: str | None = None
    profile_overrides: dict[str, Any] | None = None
    mode: Literal["t2i", "edit"] = "t2i"
    reference_asset_ids: list[str] = []
    reference_images: list[str] = []
    master_face: str = ""          # explicit path; else the project's master_face asset
    max_attempts: int | None = None
    evaluate: bool | None = None
    face_swap: bool | None = None
    force: bool = False


class SceneEdit(BaseModel):
    """One scene's editable fields (storyboard gate review). Index is required."""

    index: int
    description: str | None = None
    keyframe_prompt: str | None = None
    animation_prompt: str | None = None
    keyframe_negative: str | None = None
    animation_negative: str | None = None
    mode: Literal["t2i", "edit"] | None = None
    references: list[str] | None = None            # explicit paths (CLI / no project)
    reference_asset_ids: list[str] | None = None   # project asset ids -> resolved into `references`
    identity_asset_id: str | None = None           # face for this scene's swap; "" clears it (no swap)
    chosen_variant: int | None = None      # legacy: pick a keyframe version by list position
    chosen_variant_id: str | None = None   # pick which keyframe version is active (stable id)
    chosen_clip_id: str | None = None      # pick which rendered clip is the deliverable


class SceneEdits(BaseModel):
    scenes: list[SceneEdit]


class SceneRegenerate(BaseModel):
    """Redo one scene: ``keyframe`` (everything from the keyframe on) or ``video`` (clip only).
    Optional prompt edits are applied first."""

    what: Literal["keyframe", "video"] = "keyframe"
    description: str | None = None
    keyframe_prompt: str | None = None
    keyframe_negative: str | None = None
    animation_prompt: str | None = None
    animation_negative: str | None = None
    mode: Literal["t2i", "edit"] | None = None
    references: list[str] | None = None
    reference_asset_ids: list[str] | None = None
    identity_asset_id: str | None = None           # "" = no swap for this scene
    apply_suggestion: bool = False   # append the evaluator's last fix suggestion to the keyframe prompt
    # Use this project asset as the keyframe instead of generating one (what="keyframe").
    keyframe_asset_id: str | None = None


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


class ClipOut(BaseModel):
    """One clip a scene has rendered; the active one is the scene's deliverable."""

    id: str
    video_path: str
    url: str | None = None
    raw_video_path: str | None = None
    raw_url: str | None = None
    post_paths: list[str] = []
    seed: int | None = None
    verdict: VerdictOut | None = None
    keyframe_path: str | None = None   # the frame it was animated from
    created_at: str = ""
    batch: int = 0
    active: bool = False


class VariantOut(BaseModel):
    path: str
    url: str | None = None
    seed: int | None = None
    verdict: VerdictOut | None = None
    swapped_path: str | None = None
    swapped_url: str | None = None
    post_paths: list[str] = []
    source: str = "generated"   # "user" = an image the user chose, not generated
    asset_id: str | None = None
    id: str = ""
    created_at: str = ""
    batch: int = 0              # which generation run made it
    active: bool = False        # the scene's current keyframe


class ReferenceSuggestionOut(BaseModel):
    asset_id: str
    why: str = ""
    fit: str = "motif_only"     # same_shot (edit mode keeps the scene) | motif_only (t2i stays better)


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
    reference_asset_ids: list[str] = []
    identity: str = ""                    # face path used by this scene's swap ("" = none)
    identity_asset_id: str | None = None
    identity_url: str | None = None       # the asset's file URL when it is a project asset
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
    clips: list[ClipOut] = []   # every clip ever rendered for this scene, oldest first
    # "user" when the chosen keyframe is an image the user picked instead of a generated one.
    keyframe_source: str = "generated"
    keyframe_asset_id: str | None = None
    reference_suggestion: ReferenceSuggestionOut | None = None   # a project photo the storyboard thinks fits


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
    assembly: dict | None = None
    status: str
    created_at: str
    updated_at: str
    scenes: list[ScenePublic]
    pending_gate: Gate | None = None
    # Live run state from this server: a segment in flight, and a stop requested but not
    # yet reached. status "running" with active false is an orphan (server restarted).
    active: bool = False
    stopping: bool = False
    # How many references the job's edit workflow uses; extra ones are ignored (0 = unknown).
    edit_reference_slots: int = 0


class JobSummaryOut(BaseModel):
    profile_name: str | None = None  # from the job's profile snapshot
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


# --- assembly ----------------------------------------------------------------


class AssemblyItemIn(BaseModel):
    scene_index: int
    source: Literal["final", "raw"] = "final"
    target_seconds: float | None = Field(
        None, gt=0, description="play the clip in N seconds (0.25x–4x of its length); omit for natural speed"
    )


class AssemblyCutIn(BaseModel):
    transition: str = "fade"
    duration: float = 0.5


class AssemblySpecIn(BaseModel):
    """All fields optional: omitted items = every clip in order, omitted cuts = fade 0.5s."""

    items: list[AssemblyItemIn] | None = None
    cuts: list[AssemblyCutIn] | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    crf: int = 18
    encoder: Literal["libx264", "h264_nvenc"] = "libx264"
    smooth: bool = False   # motion-interpolated slow motion for clips slower than 1x (much slower render)
    audio: bool = False    # keep the clips' audio (LTX clips have a track; Wan clips are silent -> ignored)
    delivery: str = "source"          # a target id from AssemblyStatus.deliveries; width/height/fps win
    max_mbps: float | None = None     # cap the bitrate for a platform ceiling
    grain: float = 0.0                # 0-3: light film grain (an upscale reads plastic without it)


class AssemblyClipOut(BaseModel):
    scene_index: int
    source: str = "final"
    path: str
    url: str | None = None
    duration: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    has_raw: bool = False
    error: str | None = None


class AssemblyRecordOut(BaseModel):
    path: str
    url: str | None = None
    created_at: str = ""
    duration: float = 0.0
    n_clips: int = 0
    spec: dict = {}
    ok: bool = True
    error: str = ""
    size_bytes: int = 0
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    exists: bool = False
    exported: bool = False
    version: int = 0
    clips: list[str] = []
    stale: bool = False   # a joined clip is no longer its scene's clip: assemble again


class AssemblyVersionOut(BaseModel):
    version: int
    path: str
    url: str | None = None
    created_at: str = ""
    duration: float = 0.0
    n_clips: int = 0
    spec: dict = {}
    size_bytes: int = 0
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    exists: bool = False
    current: bool = False


class DeliveryOut(BaseModel):
    """A delivery target for this job: what the assembly is rendered as."""

    id: str
    label: str
    width: int | None = None     # null = keep the clips' own size
    height: int | None = None
    fps: int | None = None
    note: str = ""


class AssemblyStatus(BaseModel):
    available: bool
    active: bool
    transitions: list[str]
    default_spec: dict
    clips: list[AssemblyClipOut]
    profile_fps: int | None = None
    deliveries: list[DeliveryOut] = []   # what this job can be mastered as
    delivery: str = "source"             # the preselected one
    record: AssemblyRecordOut | None = None
    versions: list[AssemblyVersionOut] = []
