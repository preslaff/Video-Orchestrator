"""Structural edits to a job's storyboard: insert, remove, reorder — and renumber.

A storyboard is never right first time: Gemma collapses six climbs into one, writes filler in
the middle, and shows a face the plan wanted twice. This module is the small, pure layer that
lets a person fix that after generation; ``queue.JobManager`` owns the guards (nothing runs),
the persistence and the Gemma pass that writes an inserted scene's prompts.

Two identities, deliberately kept apart:

* ``Scene.index`` is the **position**. Export file names, assembly order and every "Scene 12"
  in the UI come from it, so it is renumbered after every structural change (``renumber``).
* ``Scene.id`` is the **identity**. It never changes, so drafts in the browser, the reorder
  payload and new decision-log rows survive a renumber that would otherwise re-attribute them.

Generated media keeps the file name it was born with (``s07_keyframe.png`` on a scene now
numbered 12): paths are stored on the scene and never re-derived, and the export folder renames
on copy, so the handoff is correct even when the working files are not.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from .state import Job, Scene, SceneStatus

log = logging.getLogger("vorch.scenes")


def by_id(job: Job, scene_id: str) -> Scene:
    """The scene with this stable id. Raises KeyError naming it."""
    found = next((s for s in job.scenes if s.id == scene_id), None)
    if found is None:
        raise KeyError(scene_id)
    return found


def insert(
    job: Job,
    position: int,
    *,
    description: str = "",
    character: str = "",
    setting: str = "",
    mood: str = "",
    writing_prompts: bool = False,
) -> Scene:
    """Add an empty scene at ``position`` (clamped to the ends) and renumber.

    It carries the job's default keyframe mode and no media, so the next run picks it up as the
    only scene that needs a keyframe (``Orchestrator._scene_needs_keyframes``).
    """
    scene = Scene(
        index=0,                       # renumber() below assigns the real one
        description=description.strip(),
        character=character.strip(),
        setting=setting.strip(),
        mood=mood.strip(),
        mode=job.mode,
        status=SceneStatus.PENDING,
        writing_prompts=writing_prompts,
    )
    at = max(0, min(int(position), len(job.scenes)))
    job.scenes.insert(at, scene)
    renumber(job)
    return scene


def remove(job: Job, scene_id: str, *, delete_files: bool = False) -> tuple[Scene, int]:
    """Drop a scene from the job and renumber; returns it and the position it held.

    ``delete_files`` also unlinks what it generated — every variant, clip and post-step output,
    including the ones no longer selected. Project assets are never touched: a user keyframe
    points at a copy under the ComfyUI output tree, not at the asset itself.
    """
    scene = by_id(job, scene_id)
    was = scene.index
    job.scenes.remove(scene)
    renumber(job)
    if delete_files:
        for path in _files_of(scene):
            try:
                Path(path).unlink()
            except OSError as e:          # already gone, or not ours to delete
                log.warning("scene %s: could not delete %s (%s)", scene_id, path, e)
    return scene, was


def _files_of(scene: Scene) -> list[str]:
    """Every file this scene generated, newest and oldest alike (no duplicates, order kept)."""
    out: list[str] = []
    for v in scene.variants:
        out += [v.path, v.swapped_path, *v.post_paths]
    for c in scene.clips:
        out += [c.video_path, c.raw_video_path, *c.post_paths]
    out += [scene.keyframe_path, scene.swapped_path, scene.video_path, scene.raw_video_path, *scene.post_paths]
    seen: list[str] = []
    for p in out:
        if p and p not in seen:
            seen.append(p)
    return seen


def reorder(job: Job, scene_ids: Iterable[str]) -> None:
    """Put the scenes in the given order and renumber.

    ``scene_ids`` must be exactly the job's scenes, each once: a partial order would silently
    drop or duplicate a scene's work, so it is a ValueError instead.
    """
    wanted = list(scene_ids)
    have = [s.id for s in job.scenes]
    if sorted(wanted) != sorted(have):
        missing = [i for i in have if i not in wanted]
        unknown = [i for i in wanted if i not in have]
        raise ValueError(
            "the order must list every scene exactly once"
            + (f"; missing {missing}" if missing else "")
            + (f"; unknown {unknown}" if unknown else "")
        )
    order = {sid: pos for pos, sid in enumerate(wanted)}
    job.scenes.sort(key=lambda s: order[s.id])
    renumber(job)


def renumber(job: Job) -> dict[int, int]:
    """``index = position`` for every scene; returns ``{old index: new index}``.

    Called after every structural change so the numbers a person sees, the export file names
    and the assembly order all agree with the list.
    """
    mapping: dict[int, int] = {}
    for pos, scene in enumerate(job.scenes):
        if scene.index != pos:
            mapping[scene.index] = pos
            scene.index = pos
    job.n_scenes = len(job.scenes)     # the storyboard-time cap would otherwise become a lie
    return mapping


def remap_assembly(job: Job, moved: dict[int, int], removed: Iterable[int] = ()) -> bool:
    """Point a stored assembly spec at the scenes it was actually built from.

    The spec addresses clips by ``scene_index``, so without this a reorder would silently join
    different clips, and a delete would make ``assembly.resolve`` raise "no scene N". Items for
    removed scenes are dropped, the rest follow their scene. Returns True when anything changed.
    """
    record = job.assembly
    if not record or not isinstance(record.get("spec"), dict):
        return False
    items = record["spec"].get("items")
    if not isinstance(items, list):
        return False
    gone = set(removed)
    kept: list[dict] = []
    changed = False
    for item in items:
        old = item.get("scene_index")
        if old in gone:
            changed = True
            continue
        if old in moved:
            item = {**item, "scene_index": moved[old]}
            changed = True
        kept.append(item)
    if changed:
        record["spec"] = {**record["spec"], "items": kept}
    return changed
