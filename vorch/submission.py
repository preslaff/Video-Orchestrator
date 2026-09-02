"""Resolve a job request into concrete inputs: profile, project, master face, references.

Shared by the HTTP API (``JobSpec``) and the CLI (``job.yaml``) so both apply
the same rules:

1. project: ``project_id`` (KeyError when unknown) or none;
2. profile: ``profile_id`` → the project's ``default_profile_id`` → the
   built-in default; then ``profile_overrides`` (deep-merged) and the legacy
   knobs (``evaluate`` / ``face_swap`` / ``max_attempts``) **only when given**;
3. master face: an explicit path, else the project's ``master_face`` asset;
4. references: project asset ids (KeyError when unknown) and/or explicit paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import profiles as profiles_mod
from . import projects as projects_mod
from .profiles import OutputProfile
from .projects import Project


@dataclass
class Submission:
    profile: OutputProfile
    project: Project | None
    master_face: str
    reference_images: list[str] = field(default_factory=list)

    @property
    def project_id(self) -> str | None:
        return self.project.id if self.project is not None else None

    @property
    def kind(self) -> str:
        return self.profile.kind


def resolve_submission(
    store: Any,
    *,
    project_id: str | None = None,
    profile_id: str | None = None,
    profile_overrides: dict | None = None,
    master_face: str = "",
    reference_asset_ids: Sequence[str] = (),
    reference_images: Sequence[str] = (),
    evaluate: bool | None = None,
    face_swap: bool | None = None,
    max_attempts: int | None = None,
) -> Submission:
    project = None
    if project_id:
        project = projects_mod.load(store, project_id)
        if project is None:
            raise KeyError(f"unknown project {project_id!r}")

    pid = profile_id or (project.default_profile_id if project else None)
    profile = profiles_mod.resolve(store, pid)
    profile = profiles_mod.apply_overrides(profile, profile_overrides)

    legacy: dict = {}
    if evaluate is not None:
        legacy.setdefault("evaluate", {})["enabled"] = bool(evaluate)
    if max_attempts is not None:
        legacy.setdefault("evaluate", {})["max_attempts"] = int(max_attempts)
    if face_swap is not None:
        legacy.setdefault("face_swap", {})["enabled"] = bool(face_swap)
    profile = profiles_mod.apply_overrides(profile, legacy)

    face = master_face or ""
    if not face and project is not None:
        asset = projects_mod.master_face(project)
        if asset is not None:
            face = asset.path

    refs: list[str] = []
    for aid in reference_asset_ids:
        if project is None:
            raise KeyError(f"reference asset {aid!r} needs a project")
        asset = project.asset(aid)
        if asset is None:
            raise KeyError(f"unknown asset {aid!r} in project {project.slug!r}")
        refs.append(asset.path)
    refs += [str(r) for r in reference_images]

    return Submission(profile=profile, project=project, master_face=face, reference_images=refs)
