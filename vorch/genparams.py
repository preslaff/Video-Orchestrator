"""GenParams — the flat generation settings every step reads.

An ``OutputProfile`` (profiles.py) is the user-facing, nested, persisted shape;
``GenParams`` is what the pipeline threads through ``keyframe`` / ``animate`` /
``faceswap`` / ``edit``: sizes, length, fps, sampler settings, negatives,
workflow names, variants, post-steps. ``DEFAULT`` reproduces the pre-profile
constants exactly, so every existing caller keeps its behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from . import config, formats
from .evaluator import REVIEW_SCORE_THRESHOLD
from .steps.animate import DEFAULT_NEGATIVE


@dataclass(frozen=True)
class PostStep:
    workflow: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenParams:
    profile_name: str = ""
    kind: str = "video"                      # video | stills
    aspect: str = "16:9"
    purpose: str = "cinematic"
    keyframe_workflow: str = "keyframe"
    animate_workflow: str = "animate"
    faceswap_workflow: str = "faceswap"
    edit_workflow: str = "identity_edit"
    keyframe_size: tuple[int, int] = (config.KEYFRAME_WIDTH, config.KEYFRAME_HEIGHT)
    wan_size: tuple[int, int] = (config.WAN_WIDTH, config.WAN_HEIGHT)
    length: int = config.WAN_LENGTH
    fps: int = config.WAN_FPS
    steps: int = config.KEYFRAME_STEPS
    cfg: float = config.KEYFRAME_CFG
    unet: str | None = None
    keyframe_negative: str = ""
    animate_negative: str = DEFAULT_NEGATIVE
    style_prefix: str = ""                   # e.g. "oil painting, impasto"
    animation_suffix: str = "cinematic"
    restore_model: str = config.REACTOR_RESTORE_MODEL
    face_swap: bool = True
    evaluate: bool = True
    max_attempts: int = 2
    variants_per_scene: int = 1
    review_threshold: int = REVIEW_SCORE_THRESHOLD
    post_steps: tuple[PostStep, ...] = ()
    edit_overrides: dict = field(default_factory=dict)  # explicit edit steps/cfg/unet only

    # -- derived -------------------------------------------------------------

    @property
    def clip_seconds(self) -> float:
        return formats.clip_seconds(self.length, self.fps)

    @property
    def orientation(self) -> str:
        return formats.orientation(self.aspect)

    def keyframe_kwargs(self) -> dict:
        kw: dict[str, Any] = {
            "workflow": self.keyframe_workflow,
            "width": self.keyframe_size[0],
            "height": self.keyframe_size[1],
            "steps": self.steps,
            "cfg": self.cfg,
        }
        if self.unet:
            kw["unet"] = self.unet
        if self.keyframe_negative:
            kw["negative"] = self.keyframe_negative
        return kw

    def animate_kwargs(self) -> dict:
        return {
            "workflow": self.animate_workflow,
            "width": self.wan_size[0],
            "height": self.wan_size[1],
            "length": self.length,
            "fps": self.fps,
            "negative": self.animate_negative,
        }

    def faceswap_kwargs(self) -> dict:
        return {"workflow": self.faceswap_workflow, "restore_model": self.restore_model}

    def edit_kwargs(self) -> dict:
        kw: dict[str, Any] = {
            "workflow": self.edit_workflow,
            "width": self.keyframe_size[0],
            "height": self.keyframe_size[1],
        }
        if self.keyframe_negative:
            kw["negative"] = self.keyframe_negative
        kw.update(self.edit_overrides)
        return kw

    # -- validation ------------------------------------------------------------

    def validate(self) -> list[str]:
        """Raise ValueError on hard errors; return soft warnings."""
        formats.validate_size(*self.keyframe_size)
        if self.kind == "video":
            rules = formats.workflow_rules(self.animate_workflow)
            formats.validate_size(*self.wan_size, maximum=rules.size_max, multiple=rules.size_multiple)
            formats.validate_length(self.length, rules.length_constraint)
            if self.fps < 1:
                raise ValueError("fps must be >= 1")
            if not formats.aspects_match(self.keyframe_size, self.wan_size):
                raise ValueError(
                    f"keyframe {self.keyframe_size} and video {self.wan_size} aspects differ; "
                    "Wan would centre-crop the keyframe"
                )
        elif self.kind != "stills":
            raise ValueError(f"kind must be video|stills, got {self.kind!r}")
        if not 1 <= self.variants_per_scene <= 8:
            raise ValueError("variants_per_scene must be 1..8")
        warnings: list[str] = []
        if self.kind == "video" and not formats.workflow_rules(self.animate_workflow).remote:
            warnings += formats.vram_warnings(*self.wan_size, self.length)
        return warnings

    # -- constructors ----------------------------------------------------------

    @classmethod
    def for_aspect(cls, aspect: str, **overrides: Any) -> "GenParams":
        """Defaults with sizes computed for ``aspect`` (16:9, 9:16, 1:1, ...)."""
        p = cls(
            aspect=aspect,
            keyframe_size=formats.keyframe_size(aspect),
            wan_size=formats.wan_size(aspect),
            **overrides,
        )
        p.validate()
        return p

    @classmethod
    def from_profile(cls, profile: Any) -> "GenParams":
        """Map a (nested) OutputProfile or its dict snapshot onto flat params."""
        if isinstance(profile, dict):
            profile = _NS.wrap(profile)
        g = lambda obj, name, default=None: getattr(obj, name, default) if obj is not None else default  # noqa: E731
        kf, vid, st, sw, ev, style = (g(profile, n) for n in ("keyframe", "video", "stills", "face_swap", "evaluate", "style"))
        kind = g(profile, "kind", "video")
        aspect = g(profile, "aspect", "16:9")
        keyframe_size = (g(kf, "width", config.KEYFRAME_WIDTH), g(kf, "height", config.KEYFRAME_HEIGHT))
        if vid is not None:
            wan_size = (g(vid, "width", config.WAN_WIDTH), g(vid, "height", config.WAN_HEIGHT))
        else:
            wan_size = formats.wan_size(aspect) if aspect in formats.WAN_PRESETS else (config.WAN_WIDTH, config.WAN_HEIGHT)
        post = tuple(
            PostStep(workflow=g(s, "workflow"), params=_as_dict(g(s, "params", {})))
            for s in (g(profile, "post_steps", []) or [])
        )
        p = cls(
            profile_name=g(profile, "id", "") or g(profile, "name", ""),
            kind=kind,
            aspect=aspect,
            purpose=g(profile, "purpose", "cinematic"),
            keyframe_workflow=g(kf, "workflow", "keyframe"),
            animate_workflow=g(vid, "workflow", "animate"),
            faceswap_workflow=g(sw, "workflow", "faceswap"),
            edit_workflow=g(profile, "edit_workflow", "identity_edit"),
            keyframe_size=keyframe_size,
            wan_size=wan_size,
            length=g(vid, "length", config.WAN_LENGTH),
            fps=g(vid, "fps", config.WAN_FPS),
            steps=g(kf, "steps", config.KEYFRAME_STEPS),
            cfg=g(kf, "cfg", config.KEYFRAME_CFG),
            unet=g(kf, "unet"),
            keyframe_negative=g(style, "keyframe_negative", ""),
            animate_negative=g(vid, "negative", None) or g(style, "animation_negative", "") or DEFAULT_NEGATIVE,
            style_prefix=g(style, "keyframe_prefix", ""),
            animation_suffix=g(style, "animation_suffix", "cinematic"),
            restore_model=g(sw, "restore_model", config.REACTOR_RESTORE_MODEL),
            face_swap=g(sw, "enabled", True),
            evaluate=g(ev, "enabled", True),
            max_attempts=g(ev, "max_attempts", 2),
            variants_per_scene=g(st, "variants", 1),
            review_threshold=g(ev, "review_threshold", REVIEW_SCORE_THRESHOLD),
            post_steps=post,
            edit_overrides=_as_dict(g(profile, "edit_overrides", {})),
        )
        p.validate()
        return p

    def with_(self, **changes: Any) -> "GenParams":
        return replace(self, **changes)


def _as_dict(v: Any) -> dict:
    if v is None:
        return {}
    if isinstance(v, _NS):
        return dict(v._d)
    if hasattr(v, "model_dump"):
        return dict(v.model_dump())
    return dict(v)


class _NS:
    """Attribute view over nested dicts (so from_profile can read snapshots)."""

    def __init__(self, d: dict):
        self._d = d

    @classmethod
    def wrap(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return cls(v)
        if isinstance(v, list):
            return [cls.wrap(x) for x in v]
        return v

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._d:
            raise AttributeError(name)
        return _NS.wrap(self._d[name])


DEFAULT = GenParams()
