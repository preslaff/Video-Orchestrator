"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

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
    # Animate images made elsewhere: one scene per asset, in this order, each installed as that
    # scene's keyframe. No storyboard is written; `idea` becomes the overall direction.
    keyframe_asset_ids: list[str] = []
    master_face: str = ""          # explicit path; else the project's master_face asset
    max_attempts: int | None = None
    evaluate: bool | None = None
    face_swap: bool | None = None
    # Ceiling in USD for hosted (API) models; null = uncapped. Estimated from each workflow's
    # declared `cost:`, retries included, and checked before every billed call.
    budget_usd: float | None = Field(default=None, ge=0)
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
    # Which edit workflow this scene uses ("" = the profile's); `edit_base="keyframe"` starts the
    # edit from the scene's own frame, "" from the references alone.
    edit_workflow: str | None = None
    keyframe_workflow: str | None = None
    edit_base: Literal["", "keyframe"] | None = None
    references: list[str] | None = None            # explicit paths (CLI / no project)
    reference_asset_ids: list[str] | None = None   # project asset ids -> resolved into `references`
    identity_asset_id: str | None = None           # face for this scene's swap; "" clears it (no swap)
    chosen_variant: int | None = None      # legacy: pick a keyframe version by list position
    chosen_variant_id: str | None = None   # pick which keyframe version is active (stable id)
    chosen_clip_id: str | None = None       # pick which rendered clip is the deliverable
    # The Bulgarian line spoken over this scene. Plain Cyrillic: the TTS verbalises numbers and
    # marks stress itself. Saving new text does not re-speak it — that is an explicit action.
    narration: str | None = None


class SceneInsert(BaseModel):
    """Add a scene at ``position`` (0 = first, len = last); Gemma writes its prompts."""

    position: int = Field(..., ge=0)
    description: str = Field("", max_length=2000)
    character: str = ""
    setting: str = ""
    mood: str = ""
    generate: bool = True      # False = an empty scene to write by hand


class SceneOrder(BaseModel):
    """The job's scenes in their new order — every scene's id, exactly once."""

    scene_ids: list[str]


class SaveSceneAsset(BaseModel):
    """Copy a scene's generated keyframe into the job's project, to reuse as a reference."""

    version_id: str | None = None                 # a keyframe version; default: the active one
    kind: Literal["reference", "product", "logo", "other"] = "reference"
    label: str = ""                               # given = a person named it; empty = Gemma describes it


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
    edit_workflow: str | None = None
    keyframe_workflow: str | None = None
    edit_base: Literal["", "keyframe"] | None = None
    references: list[str] | None = None
    reference_asset_ids: list[str] | None = None
    identity_asset_id: str | None = None           # "" = no swap for this scene
    apply_suggestion: bool = False   # append the evaluator's last fix suggestion to the keyframe prompt
    # Use this project asset as the keyframe instead of generating one (what="keyframe").
    keyframe_asset_id: str | None = None
    # Use this project VIDEO as the scene's clip instead of animating one (what="video"): it is
    # conformed to the job's size and fps, and the scene's keyframe becomes its first frame.
    clip_asset_id: str | None = None


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


class VersionDetail(BaseModel):
    """What made one keyframe or clip: the prompts recorded with it, plus the sidecar manifest.

    The prompts come from the version record (history that outlives the output folder); the
    manifest beside the file adds the rest — steps, cfg, model, backend — when it is still there.
    """

    version_id: str
    kind: Literal["keyframe", "clip"]
    path: str
    url: str | None = None
    created_at: str = ""
    batch: int = 0
    seed: int | None = None
    prompt: str = ""
    negative: str = ""
    source: str = "generated"          # "user" = an image the person chose, never generated
    # Where the prompt was found: "record" (saved with the version) or "manifest" (the sidecar,
    # for versions made before prompts were recorded) or "" when neither had one.
    prompt_from: str = ""
    params: dict[str, Any] = {}        # the manifest's params (steps, cfg, model, size, …)
    workflow: str = ""
    template: str = ""
    backend: str = ""
    has_manifest: bool = False
    # How it compares with what the scene would generate now.
    matches_scene_prompt: bool = True


class LockedRegion(BaseModel):
    """A rectangle of the frame, in fractions of its width/height, whose pixels come from
    the keyframe rather than from the model — a brand label, a monogram, an emblem.

    A video model redraws every frame from a latent compressed 8x in each direction, so type
    below roughly 32 px of cap height cannot survive it. Marking it takes it out of the
    model's hands entirely.
    """

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    # Lock the *shape* inside this rectangle rather than the rectangle itself — a label on a
    # curved bottle is not a rectangle. The matte is segmented from the clip's own frames and
    # tracked through it; the box still decides which keyframe pixels are used and at what scale.
    # Costs a SAM2 pass per box (~7.5 s for 81 frames), so saving is no longer instant.
    matte: bool = False


class ScenePublic(BaseModel):
    index: int                  # position in the job; renumbered when scenes are added/moved
    id: str = ""                # stable identity, survives a renumber (edits, reorder payloads)
    writing_prompts: bool = False   # Gemma is writing this scene's prompts right now
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
    edit_workflow: str = ""        # "" = the profile's edit workflow
    keyframe_workflow: str = ""    # "" = the profile's keyframe workflow
    edit_base: str = ""            # "keyframe" = the edit starts from this scene's own frame
    edit_base_path: str = ""       # that frame, as resolved when the edit was applied
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
    # Rectangles of the frame the animator must not redraw, restored from the keyframe.
    locked_regions: list[LockedRegion] = []
    narration: str = ""                  # the line to speak over this scene
    narration_url: str | None = None     # the voice, once spoken
    narration_seconds: float = 0.0
    narration_stale: bool = False        # the line has been reworded since it was spoken
    texts: list[TextItemIn] = []         # captions over this scene, in clip time


class SceneLocks(BaseModel):
    regions: list[LockedRegion] = []


class JobBudget(BaseModel):
    """Raise (or lift) a job's ceiling for hosted-model spending."""

    budget_usd: float | None = Field(default=None, ge=0)


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
    # Hosted-model spending: the ceiling, the running estimate, and every billed call.
    budget_usd: float | None = None
    spend_usd: float = 0.0             # worst case of what has been billed so far
    spend_calls: int = 0
    spend: list[dict] = []
    budget_reason: str = ""            # why the job is stopped on its budget, if it is
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
    scene_id: str | None = None   # set since scenes could be reordered; older rows have only the index
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


class TextItemIn(BaseModel):
    """One caption. Every appearance value lives on the item — there is no preset to look up at
    render time, so changing a preset later never moves text that already exists.

    Colours are ``RRGGBB`` as a person writes them. ASS wants those bytes reversed, and that
    conversion happens in exactly one place (``vorch.titles.ass_colour``), nowhere near this edge.
    """

    text: str = Field(description="what to draw; a newline becomes a hard line break")
    start: float = Field(0.0, ge=0, description="seconds — into the clip for a scene's caption, "
                                                "onto the finished film for the timeline's")
    duration: float = Field(2.0, gt=0, description="seconds on screen")
    effect: str = Field("fade", description="one of the assembly status's `text_effects`")
    effect_seconds: float = Field(0.35, ge=0, description="how long the effect takes")
    font: str = "DejaVu Sans"
    size: float = Field(0.048, gt=0, le=1, description="fraction of the frame HEIGHT, so 9:16 and "
                                                      "16:9 read the same")
    colour: str = Field("FFFFFF", description="RRGGBB")
    outline: str = Field("202020", description="RRGGBB")
    outline_width: float = Field(0.004, ge=0, le=0.1, description="fraction of the frame height")
    shadow: float = Field(0.0, ge=0, le=0.1)
    bold: bool = True
    italic: bool = False
    align: int = Field(2, ge=1, le=9, description="ASS numpad anchor: 1 bottom-left … 9 top-right")
    margin_x: float = Field(0.08, ge=0, lt=0.5, description="fraction of the frame width")
    margin_y: float = Field(0.08, ge=0, lt=0.5, description="fraction of the frame height")
    angle: float = 0.0
    opacity: float = Field(1.0, ge=0, le=1)


class SceneTexts(BaseModel):
    """The whole caption list for one scene — an empty list removes them all."""

    texts: list[TextItemIn] = []


class AssemblyItemIn(BaseModel):
    scene_index: int
    source: Literal["final", "raw"] = "final"
    target_seconds: float | None = Field(
        None, gt=0,
        description="play the KEPT part in N seconds (0.25x-4x of it); omit for natural speed. "
                    "Trim first, then retime: this measures what start/end kept, not the whole clip.",
    )
    clip_id: str | None = Field(
        None,
        description="which take from the scene's history; omit for whatever the scene points at now. "
                    "A scene may be listed more than once, so two takes can be joined into one longer shot.",
    )
    start: float | None = Field(
        None, ge=0,
        description="seconds into the clip to start at; omit for the beginning. Snapped to the clip's "
                    "own frame grid (at 16 fps one frame is 62.5 ms, so 1.234 becomes 1.25).",
    )
    end: float | None = Field(
        None, gt=0, description="seconds to end at; omit for the end. Snapped like `start`.",
    )

    @model_validator(mode="after")
    def _span_is_possible(self) -> "AssemblyItemIn":
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError(f"end ({self.end:g} s) must be after start ({self.start:g} s)")
        return self


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
    narration: bool = True            # lay each scene's spoken line over its clip, when it has one
    duck: float = Field(6.0, ge=1, le=20, description="how hard the clips duck under the voice")
    delivery: str = "source"          # a target id from AssemblyStatus.deliveries; width/height/fps win
    max_mbps: float | None = None     # cap the bitrate for a platform ceiling
    grain: float = 0.0                # 0-3: light film grain (an upscale reads plastic without it)
    texts: list[TextItemIn] | None = Field(
        None, description="captions on the finished film — a title card, an end card, credits. "
                          "Times are absolute, so these do not follow a reorder or a trim.")
    titles: bool = Field(True, description="draw captions at all; off renders a clean pass "
                                           "without deleting anything")


class AssemblyTakeOut(BaseModel):
    """One render this scene has had. Placing an older take beside the current one extends a shot
    with footage already paid for."""

    id: str
    path: str
    url: str | None = None
    created_at: str = ""
    batch: int = 0
    active: bool = False


class SaveCut(BaseModel):
    """Keep a timeline's trim as a real take of that scene."""

    clip_id: str | None = Field(None, description="which take to cut from; omit for the scene's current clip")
    start: float | None = Field(None, ge=0)
    end: float | None = Field(None, gt=0)
    make_active: bool = Field(
        False, description="also make the cut this scene's clip, so the export folder carries it")


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
    has_audio: bool = Field(
        False, description="the clip carries an audio track, so 'keep audio' has something to keep "
                           "— without it, narration replaces the clip's own sound instead of "
                           "sitting over it")
    takes: list[AssemblyTakeOut] = []
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
    text_effects: list[str] = []
    text_presets: dict[str, dict] = {}
    text_fonts: list[str] = []           # families installed here that can draw Cyrillic
