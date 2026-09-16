"""Choosing among a scene's media versions.

A scene keeps every keyframe it has had (``Scene.variants``) and every clip it has
rendered (``Scene.clips``); the scene's scalar fields — ``keyframe_path``,
``swapped_path``, ``video_path``, ``raw_video_path``, ``post_paths`` — are the
**projection of the active pair**, so assembly, export and the API keep reading one
keyframe and one clip without knowing about history.

This module is the single place that moves those pointers: which version is active,
what that implies for what was built on top of it, and what removing a version does.
Files are never touched — deleting a version drops the reference only; the image or
clip stays on disk with its provenance sidecar.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .state import Clip, Scene, SceneStatus, Variant


def variant_by(scene: Scene, *, id: str | None = None, index: int | None = None) -> Variant:
    """One keyframe version by id (stable) or list index (legacy edits). Raises ValueError."""
    if id is not None:
        found = next((v for v in scene.variants if v.id == id), None)
        if found is None:
            raise ValueError(f"scene {scene.index}: no keyframe version {id!r}")
        return found
    if index is None:
        raise ValueError("give a version id or an index")
    if not 0 <= int(index) < len(scene.variants):
        raise ValueError(f"scene {scene.index}: no variant {index}")
    return scene.variants[int(index)]


def clip_by(scene: Scene, clip_id: str) -> Clip:
    found = next((c for c in scene.clips if c.id == clip_id), None)
    if found is None:
        raise ValueError(f"scene {scene.index}: no clip version {clip_id!r}")
    return found


def active_variant(scene: Scene) -> Variant | None:
    return next((v for v in scene.variants if v.path == scene.keyframe_path), None)


def active_clip(scene: Scene) -> Clip | None:
    return next((c for c in scene.clips if c.video_path == scene.video_path), None)


def _project_clip(scene: Scene, clip: Clip | None) -> None:
    """Point the scene's clip fields at ``clip`` (or clear them when there is none)."""
    scene.video_path = clip.video_path if clip else None
    scene.raw_video_path = clip.raw_video_path if clip else None
    scene.post_paths = list(clip.post_paths) if clip else []
    scene.video_verdict = clip.verdict if clip else None


def select_variant(scene: Scene, variant: Variant, *, kind: str) -> str:
    """Make ``variant`` the scene's keyframe. Returns what happened to the clip:

    ``"unchanged"`` (already active), ``"stills"`` (the scene now shows that variant's
    swap/post outputs), ``"clip_reused"`` (a clip animated from this keyframe was still in
    the history, so it becomes active again — nothing to re-render) or ``"clip_cleared"``
    (no such clip: the video phase will render one and the gate reopens).
    """
    if variant.path == scene.keyframe_path:
        return "unchanged"
    scene.keyframe_path, scene.keyframe_verdict = variant.path, variant.verdict
    if kind == "stills":
        # Stills finish every variant of a run; the scene shows the chosen one's outputs.
        scene.swapped_path = variant.swapped_path
        scene.post_paths = list(variant.post_paths)
        return "stills"
    scene.swapped_path = variant.swapped_path   # the swap belongs to the keyframe
    scene.swap_verdict = None
    made_from_it = [c for c in scene.clips if c.keyframe_path == variant.path]
    if made_from_it:
        _project_clip(scene, made_from_it[-1])   # the newest take of this keyframe
        scene.status = SceneStatus.COMPLETE
        return "clip_reused"
    _project_clip(scene, None)
    scene.status = SceneStatus.KEYFRAME
    return "clip_cleared"


def select_clip(scene: Scene, clip: Clip) -> bool:
    """Make ``clip`` the scene's deliverable (what assembly and export use). False if already."""
    if clip.video_path == scene.video_path:
        return False
    _project_clip(scene, clip)
    scene.status = SceneStatus.COMPLETE
    return True


def delete_version(scene: Scene, version_id: str, *, kind: str) -> str:
    """Drop one keyframe or clip version from the history; the file stays on disk.

    Removing the active one falls back to the newest remaining version of that kind (for a
    keyframe, via ``select_variant``, so the clip follows the same rules). Returns
    ``"keyframe"`` or ``"clip"``; raises ValueError when the id is unknown.
    """
    variant = next((v for v in scene.variants if v.id == version_id), None)
    if variant is not None:
        was_active = variant.path == scene.keyframe_path
        scene.variants.remove(variant)
        if was_active:
            if scene.variants:
                select_variant(scene, scene.variants[-1], kind=kind)
            else:
                scene.keyframe_path, scene.keyframe_verdict = None, None
                scene.swapped_path, scene.swap_verdict = None, None
                _project_clip(scene, None)
                scene.status = SceneStatus.PENDING
        return "keyframe"
    clip = next((c for c in scene.clips if c.id == version_id), None)
    if clip is None:
        raise ValueError(f"scene {scene.index}: no version {version_id!r}")
    was_active = clip.video_path == scene.video_path
    scene.clips.remove(clip)
    if was_active:
        _project_clip(scene, scene.clips[-1] if scene.clips else None)
        if not scene.clips:
            scene.status = SceneStatus.KEYFRAME if scene.keyframe_path else SceneStatus.PENDING
    return "clip"


def stable_id(path: str) -> str:
    """A version id derived from the file path.

    Used when adopting media a job had before versions existed: the id is then the same on
    every read, so the UI can select that version without the job having to be written first.
    """
    return hashlib.sha1(path.encode()).hexdigest()[:8]


def adopt_legacy(scene: Any) -> bool:
    """Give a scene saved before versioning its history: the keyframe and clip it already has.

    Jobs made before this feature have ``keyframe_path`` / ``video_path`` but empty lists, so
    their current media would not appear in the version strips. Ids are derived from the path
    (``stable_id``), which makes this safe to run on a read. Returns True if anything was added.
    """
    changed = False
    if scene.keyframe_path and not any(v.path == scene.keyframe_path for v in scene.variants):
        scene.variants.append(Variant(path=scene.keyframe_path, verdict=scene.keyframe_verdict,
                                      swapped_path=scene.swapped_path, id=stable_id(scene.keyframe_path)))
        changed = True
    if scene.video_path and not any(c.video_path == scene.video_path for c in scene.clips):
        scene.clips.append(Clip(video_path=scene.video_path, raw_video_path=scene.raw_video_path,
                                post_paths=list(scene.post_paths), verdict=scene.video_verdict,
                                keyframe_path=scene.keyframe_path, id=stable_id(scene.video_path)))
        changed = True
    return changed
