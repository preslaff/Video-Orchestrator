"""Per-scene reference assets: the face used by the swap and the edit-mode references.

Scenes store both the project **asset ids** (what the UI and the storyboard pick, what
provenance records) and the resolved **paths** (what the pipeline stages). This module
is the single place that turns an edit in asset ids into paths, used by the API scene
editor, by regenerate and by the storyboard assignment.

Edit semantics: ``identity_asset_id`` None = untouched, ``""`` = clear (no swap for the
scene), an id = set; ``reference_asset_ids`` None = untouched, a list = replace.
"""

from __future__ import annotations

from typing import Any

from . import projects as projects_mod

ASSET_EDIT_FIELDS = ("identity_asset_id", "reference_asset_ids")


def has_asset_edits(edits: dict | None) -> bool:
    return bool(edits) and any(edits.get(k) is not None for k in ASSET_EDIT_FIELDS)


def apply_asset_edits(scene: Any, edits: dict, project: Any | None) -> list[str]:
    """Apply asset-id edits to ``scene``; return the names of the fields changed.

    Raises ValueError when ids are given but the job has no project, and
    ValueError naming the id when it is not one of the project's assets.
    """
    changed: list[str] = []
    ident = edits.get("identity_asset_id")
    if ident is not None:
        if ident == "":
            scene.identity, scene.identity_asset_id = "", None
        else:
            scene.identity = _resolve(project, [ident])[0]
            scene.identity_asset_id = ident
        changed.append("identity")
    refs = edits.get("reference_asset_ids")
    if refs is not None:
        scene.references = _resolve(project, list(refs))
        scene.reference_asset_ids = list(refs)
        changed.append("references")
    return changed


def _resolve(project: Any | None, ids: list[str]) -> list[str]:
    try:
        return projects_mod.resolve_asset_paths(project, ids)
    except KeyError as e:
        if str(e).strip("'") == "no project":
            raise ValueError("the job has no project: pass reference paths instead of asset ids") from None
        raise ValueError(f"unknown project asset {str(e).strip(chr(39))!r}") from None


def identity_label(project: Any | None, scene: Any) -> str:
    """Human name of a scene's identity: the asset label/name, else the file name."""
    if project is not None and scene.identity_asset_id:
        a = project.asset(scene.identity_asset_id)
        if a is not None:
            return a.label or a.name
    return scene.identity.rsplit("/", 1)[-1] if scene.identity else ""


# --- validation & invalidation (edits at any stage) -----------------------------------

MODE_FIELDS = ("mode", "references", "reference_asset_ids", "edit_workflow", "edit_base")


def resolve_edit_base(scene: Any) -> None:
    """Turn ``edit_base="keyframe"`` into the file the scene edits from.

    Resolved when the edit is applied, not when it runs: regenerating a keyframe clears
    ``keyframe_path`` first (``Orchestrator.reopen_scene``), which is exactly when the base
    would otherwise disappear.
    """
    if getattr(scene, "edit_base", "") == "keyframe":
        current = scene.swapped_path or scene.keyframe_path
        if current:
            scene.edit_base_path = current
    else:
        scene.edit_base_path = ""


def validate_scene(scene: Any) -> None:
    """A scene's keyframe settings must be generatable; raise ValueError otherwise.

    Checked *before* any work starts: a bad edit used to surface only inside the run
    segment, turning a typo into a FAILED job instead of a 400.
    """
    from pathlib import Path

    if scene.mode not in ("t2i", "edit"):
        raise ValueError(f"scene {scene.index}: unknown mode {scene.mode!r} (t2i | edit)")
    if scene.mode == "edit":
        base = getattr(scene, "edit_base_path", "")
        if not scene.references and not base:
            raise ValueError(f"scene {scene.index}: edit mode needs at least one reference image")
        missing = [r for r in ([base] if base else []) + list(scene.references)
                   if not Path(r).expanduser().is_file()]
        if missing:
            raise ValueError(f"scene {scene.index}: reference image not found: {missing[0]}")
    # A scene may name its own edit or keyframe workflow; an unknown one must be a 400 here, not
    # a KeyError inside the run segment (which fails the whole job).
    for field, want in (("edit_workflow", "edit"), ("keyframe_workflow", "keyframe")):
        name = getattr(scene, field, "")
        if not name:
            continue
        from . import registry  # local: registry pulls in the workflow templates

        try:
            spec = registry.load_registry().get(name)
        except (KeyError, registry.RegistryError):
            raise ValueError(
                f"scene {scene.index}: unknown {want} workflow {name!r}") from None
        if spec.kind != want:
            raise ValueError(f"scene {scene.index}: workflow {name!r} is kind "
                             f"{spec.kind!r}, not {want!r}")


def invalidate_after_identity_change(scene: Any, kind: str) -> None:
    """Drop every output built on the scene's previous face, so the new one is applied.

    Video: the swap, the raw clip, the clip and its post-steps (the keyframe stays).
    Stills: every variant's swap and post-steps. The scene goes back to KEYFRAME so the
    next run redoes exactly that. Kept out of ``apply_asset_edits``, which the storyboard
    also uses on scenes that have no outputs yet.
    """
    from .state import SceneStatus

    scene.swapped_path = None
    scene.swap_verdict = None
    scene.post_paths = []
    if kind == "stills":
        for v in scene.variants:
            v.swapped_path = None
            v.post_paths = []
    else:
        scene.raw_video_path = None
        scene.video_path = None
        scene.video_verdict = None
    if scene.keyframe_path:
        scene.status = SceneStatus.KEYFRAME
