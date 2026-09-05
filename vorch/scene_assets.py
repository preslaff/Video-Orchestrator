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
