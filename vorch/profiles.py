"""Output profiles — the persisted, user-facing generation presets.

A profile says *what kind of raw material* a job produces: purpose, kind
(video | stills), aspect + sizes, clip length/fps, sampler settings, style
vocabulary + negatives, face-swap / evaluation settings, which registered
workflows each step uses, and optional post-steps. Built-ins ship as YAML in
``vorch/profiles_builtin/`` (read-only, versioned with the code); user
profiles live in the ``profiles`` table. At submit the chosen profile is
snapshotted onto the job (``Job.profile``) so later edits never change a run.
``genparams.GenParams.from_profile`` turns either into the flat params the
pipeline reads.
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from . import config, formats, registry
from .steps.animate import DEFAULT_NEGATIVE

BUILTIN_DIR = Path(__file__).parent / "profiles_builtin"
ALL_GATES = ("storyboard", "keyframes")

Purpose = Literal["cinematic", "social_vertical", "social_square", "ad_broll", "stills"]
Kind = Literal["video", "stills"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KeyframeSettings(BaseModel):
    workflow: str = "keyframe"
    width: int = config.KEYFRAME_WIDTH
    height: int = config.KEYFRAME_HEIGHT
    steps: int = Field(default=config.KEYFRAME_STEPS, ge=1)
    cfg: float = config.KEYFRAME_CFG
    unet: str | None = None

    @model_validator(mode="after")
    def _size(self) -> "KeyframeSettings":
        r = formats.keyframe_rules(self.workflow)
        formats.validate_size(self.width, self.height,
                              minimum=r.size_min, maximum=r.size_max, multiple=r.size_multiple)
        return self


class VideoSettings(BaseModel):
    workflow: str = "animate"
    width: int = config.WAN_WIDTH
    height: int = config.WAN_HEIGHT
    length: int = config.WAN_LENGTH
    fps: int = Field(default=config.WAN_FPS, ge=1, le=60)
    negative: str = ""  # "" -> style.animation_negative / DEFAULT_NEGATIVE

    @model_validator(mode="after")
    def _size_and_length(self) -> "VideoSettings":
        rules = formats.workflow_rules(self.workflow)  # Wan: 4n+1 / 16; LTX: 8n+1 / 32
        formats.validate_size(self.width, self.height, maximum=rules.size_max, multiple=rules.size_multiple)
        formats.validate_length(self.length, rules.length_constraint)
        return self


class StillsSettings(BaseModel):
    variants: int = Field(default=1, ge=1, le=8)


class StyleSettings(BaseModel):
    keyframe_prefix: str = ""
    keyframe_negative: str = "smooth, digital render, 3d, airbrushed"
    animation_suffix: str = "cinematic"
    animation_negative: str = DEFAULT_NEGATIVE


class FaceSwapSettings(BaseModel):
    enabled: bool = True
    workflow: str = "faceswap"
    restore_model: str = config.REACTOR_RESTORE_MODEL


class EvalSettings(BaseModel):
    enabled: bool = True
    max_attempts: int = Field(default=2, ge=1, le=6)
    review_threshold: int = Field(default=4, ge=0, le=10)


class PostStep(BaseModel):
    workflow: str
    params: dict[str, Any] = Field(default_factory=dict)


class OutputProfile(BaseModel):
    id: str
    name: str
    purpose: Purpose = "cinematic"
    kind: Kind = "video"
    aspect: str = "16:9"
    keyframe: KeyframeSettings = Field(default_factory=KeyframeSettings)
    video: VideoSettings | None = Field(default_factory=VideoSettings)
    stills: StillsSettings = Field(default_factory=StillsSettings)
    style: StyleSettings = Field(default_factory=StyleSettings)
    face_swap: FaceSwapSettings = Field(default_factory=FaceSwapSettings)
    evaluate: EvalSettings = Field(default_factory=EvalSettings)
    edit_workflow: str = "identity_edit"
    post_steps: list[PostStep] = Field(default_factory=list)
    builtin: bool = False
    created_at: str = ""
    updated_at: str = ""

    @field_validator("aspect")
    @classmethod
    def _aspect(cls, v: str) -> str:
        formats.parse_aspect(v)
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "OutputProfile":
        if self.kind == "video" and self.video is None:
            raise ValueError("kind='video' needs a `video` block")
        if self.kind == "stills":
            self.video = None
        kf = (self.keyframe.width, self.keyframe.height)
        aw, ah = formats.parse_aspect(self.aspect)
        if not formats.aspects_match(kf, (aw, ah)):
            raise ValueError(f"keyframe size {kf} does not match aspect {self.aspect}")
        if self.video is not None and not formats.aspects_match(kf, (self.video.width, self.video.height)):
            raise ValueError("keyframe and video sizes have different aspects")
        # An enum-size keyframe workflow renders only the aspects its from_aspect table lists.
        r = formats.keyframe_rules(self.keyframe.workflow)
        if r.from_aspect and self.aspect not in r.aspects:
            raise ValueError(
                f"keyframe workflow {self.keyframe.workflow!r} cannot render aspect "
                f"{self.aspect!r} (it offers {', '.join(r.aspects)})"
            )
        return self

    # -- helpers ---------------------------------------------------------------

    def snapshot(self) -> dict:
        """JSON-safe frozen copy for ``Job.profile``."""
        return self.model_dump(mode="json")

    def workflows_used(self) -> list[str]:
        names = [self.keyframe.workflow]
        if self.video is not None:
            names.append(self.video.workflow)
        if self.face_swap.enabled:
            names.append(self.face_swap.workflow)
        names += [s.workflow for s in self.post_steps]
        return list(dict.fromkeys(names))

    def gates(self) -> list[str]:
        return gates_for(self)


def gates_for(profile: OutputProfile | dict | None) -> list[str]:
    """Video jobs pause after the storyboard and before Wan; stills only after the storyboard."""
    kind = (profile.get("kind") if isinstance(profile, dict) else getattr(profile, "kind", "video")) if profile else "video"
    return ["storyboard"] if kind == "stills" else ["storyboard", "keyframes"]


# --- built-ins ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def builtin_profiles() -> dict[str, OutputProfile]:
    out: dict[str, OutputProfile] = {}
    for y in sorted(BUILTIN_DIR.glob("*.yaml")):
        with open(y, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        doc.setdefault("id", y.stem)
        doc["builtin"] = True
        p = OutputProfile.model_validate(doc)
        out[p.id] = p
    if "cinematic_16x9" not in out:
        raise RuntimeError("built-in profile cinematic_16x9 is missing")
    return out


DEFAULT_PROFILE_ID = "cinematic_16x9"


def default_profile() -> OutputProfile:
    return builtin_profiles()[DEFAULT_PROFILE_ID]


# --- overrides / cloning ----------------------------------------------------------------


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(profile: OutputProfile, overrides: dict | None) -> OutputProfile:
    """A revalidated copy with ``overrides`` deep-merged (dotted keys not supported)."""
    if not overrides:
        return profile
    return OutputProfile.model_validate(_deep_merge(profile.model_dump(mode="json"), overrides))


def new_user_id() -> str:
    return f"u_{uuid.uuid4().hex[:8]}"


def clone(profile: OutputProfile, name: str, *, overrides: dict | None = None) -> OutputProfile:
    d = _deep_merge(profile.model_dump(mode="json"), overrides or {})
    now = _now()
    d.update(id=new_user_id(), name=name, builtin=False, created_at=now, updated_at=now)
    return OutputProfile.model_validate(d)


# --- legacy mapping ------------------------------------------------------------------


def from_legacy_spec(spec: dict, store: Any | None = None) -> OutputProfile:
    """``job.yaml`` (pre-profile) -> a profile: cinematic_16x9 plus the old knobs.

    Recognised keys: ``profile`` (id — a built-in, or with ``store`` also a user
    profile such as an imported cloud one), ``profile_overrides``, ``evaluate``,
    ``face_swap``, ``retry.max_attempts``, ``video.length`` / ``video.fps``,
    ``aspect`` (picks sizes via formats when no profile id is given).
    """
    pid = spec.get("profile") or DEFAULT_PROFILE_ID
    base = builtin_profiles().get(pid)
    if base is None and store is not None:
        try:
            base = resolve(store, pid)
        except KeyError:
            base = None
    if base is None:
        raise KeyError(f"unknown profile {pid!r}; built-ins: {sorted(builtin_profiles())}"
                       + (" (user profiles: `vorch profiles`)" if store is not None else ""))
    ov: dict = {}
    if spec.get("aspect") and not spec.get("profile"):
        a = str(spec["aspect"])
        kw, kh = formats.keyframe_size(a)
        ww, wh = formats.wan_size(a)
        ov.update(aspect=a, keyframe={"width": kw, "height": kh}, video={"width": ww, "height": wh})
    if "evaluate" in spec:
        ov.setdefault("evaluate", {})["enabled"] = bool(spec["evaluate"])
    if "face_swap" in spec:
        ov.setdefault("face_swap", {})["enabled"] = bool(spec["face_swap"])
    retry = spec.get("retry") or {}
    if "max_attempts" in retry:
        ov.setdefault("evaluate", {})["max_attempts"] = int(retry["max_attempts"])
    video = spec.get("video") or {}
    for k in ("length", "fps"):
        if k in video and base.kind == "video":
            ov.setdefault("video", {})[k] = int(video[k])
    ov = _deep_merge(ov, spec.get("profile_overrides") or {})
    return apply_overrides(base, ov)


def legacy_profile_for_job(job: Any) -> OutputProfile:
    """For jobs persisted before profiles existed: defaults + the job's run-config."""
    return apply_overrides(default_profile(), {
        "evaluate": {"enabled": bool(getattr(job, "evaluate", True)),
                     "max_attempts": int(getattr(job, "max_attempts", 2))},
        "face_swap": {"enabled": bool(getattr(job, "face_swap", True))},
    })


# --- store access ------------------------------------------------------------------


def resolve(store: Any, profile_id: str | None) -> OutputProfile:
    """DB first, then built-ins. KeyError when unknown."""
    pid = profile_id or DEFAULT_PROFILE_ID
    if store is not None:
        row = store.load_profile(pid)
        if row is not None:
            return OutputProfile.model_validate(row)
    try:
        return builtin_profiles()[pid]
    except KeyError:
        raise KeyError(f"unknown profile {pid!r}") from None


def list_all(store: Any) -> list[OutputProfile]:
    out = list(builtin_profiles().values())
    if store is not None:
        out += [OutputProfile.model_validate(r) for r in store.list_profiles()]
    return out


def save(store: Any, profile: OutputProfile) -> OutputProfile:
    if profile.builtin:
        raise PermissionError(f"profile {profile.id!r} is built-in (clone it instead)")
    now = _now()
    if not profile.created_at:
        profile.created_at = now
    profile.updated_at = now
    store.save_profile(profile.model_dump(mode="json"))
    return profile
