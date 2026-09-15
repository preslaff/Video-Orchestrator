"""Projects / campaigns: CRUD, assets (multipart upload, download, delete), jobs."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator

from .. import media, profiles, projects
from ..describer import AssetDescriber
from ..projects import ASSET_KINDS, Project
from ..state import Store
from .deps import get_describer, get_store
from .errors import ApiError
from .schemas import JobSummaryOut

router = APIRouter(prefix="/api/projects", tags=["projects"])


class AssetOut(BaseModel):
    id: str
    kind: str
    name: str
    label: str = ""
    # "" never labelled | "auto" suggested by Gemma, awaiting review | "human" confirmed.
    # The UI flags "auto" so the reviewer can see which labels nobody has checked.
    label_source: str = ""
    # The one-sentence description behind the label — shown under the thumbnail in the
    # asset list and the per-scene pickers. The rest of the note stays server-side.
    note_caption: str = ""
    note_tags: list[str] = []    # tags from the description, for filtering pickers
    note_text: str = ""          # legible sign / plaque text, for search ("mizen")
    # Background description: "queued" | "describing" | "failed" | "" (idle), with the error.
    describe_state: str = ""
    describe_error: str = ""
    sha256: str = ""
    created_at: str = ""
    path: str
    url: str
    thumb_url: str               # small cached JPEG for pickers and grids


class ProjectOut(BaseModel):
    id: str
    slug: str
    name: str
    brand_notes: str = ""
    style_notes: str = ""
    default_profile_id: str
    export_root: str = ""
    export_root_effective: str
    describe_hint: str = ""
    assets: list[AssetOut] = []
    created_at: str = ""
    updated_at: str = ""
    n_jobs: int = 0


class ProjectSummaryOut(BaseModel):
    id: str
    slug: str
    name: str
    default_profile_id: str
    n_assets: int
    n_jobs: int
    updated_at: str
    last_job_at: str | None = None  # updated_at of the project's most recent job


class ProjectCreate(BaseModel):
    name: str
    slug: str | None = None
    brand_notes: str = ""
    style_notes: str = ""
    default_profile_id: str = profiles.DEFAULT_PROFILE_ID
    export_root: str = ""
    describe_hint: str = ""


class ProjectPatch(BaseModel):
    name: str | None = None
    brand_notes: str | None = None
    style_notes: str | None = None
    default_profile_id: str | None = None
    export_root: str | None = None
    describe_hint: str | None = None


def _asset_out(project: Project, a, describer: AssetDescriber | None = None) -> AssetOut:
    note = a.note or {}
    state, error = describer.state(a.id) if describer is not None else ("", "")
    base = f"/api/projects/{project.id}/assets/{a.id}"
    return AssetOut(id=a.id, kind=a.kind, name=a.name, label=a.label, label_source=a.label_source,
                    note_caption=str(note.get("caption", "")),
                    note_tags=[str(t) for t in note.get("tags", []) or []],
                    note_text=str(note.get("text_in_image", "")),
                    describe_state=state, describe_error=error,
                    sha256=a.sha256, created_at=a.created_at,
                    path=a.path, url=f"{base}/file", thumb_url=f"{base}/thumb")


def _out(store: Store, p: Project, describer: AssetDescriber | None = None) -> ProjectOut:
    return ProjectOut(
        id=p.id, slug=p.slug, name=p.name, brand_notes=p.brand_notes, style_notes=p.style_notes,
        default_profile_id=p.default_profile_id, export_root=p.export_root,
        export_root_effective=str(projects.export_root(p)), describe_hint=p.describe_hint,
        assets=[_asset_out(p, a, describer) for a in p.assets], created_at=p.created_at, updated_at=p.updated_at,
        n_jobs=store.count_jobs(project_id=p.id),
    )


def _load(store: Store, project_id: str) -> Project:
    p = projects.load(store, project_id)
    if p is None:
        raise ApiError(404, "not_found", f"no such project {project_id!r}")
    return p


def _check_profile(store: Store, profile_id: str) -> None:
    try:
        profiles.resolve(store, profile_id)
    except KeyError:
        raise ApiError(400, "bad_request", f"unknown default profile {profile_id!r}")


def _save(store: Store, p: Project) -> Project:
    try:
        return projects.save(store, p)
    except projects.SlugTaken as e:
        raise ApiError(409, "conflict", str(e))


@router.get("", response_model=list[ProjectSummaryOut])
async def list_projects(store: Store = Depends(get_store)) -> list[ProjectSummaryOut]:
    out = []
    for p in projects.list_all(store):
        latest = store.list_jobs(project_id=p.id, limit=1)
        out.append(ProjectSummaryOut(
            id=p.id, slug=p.slug, name=p.name, default_profile_id=p.default_profile_id,
            n_assets=len(p.assets), n_jobs=store.count_jobs(project_id=p.id), updated_at=p.updated_at,
            last_job_at=latest[0].updated_at if latest else None,
        ))
    return out


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(body: ProjectCreate, store: Store = Depends(get_store)) -> ProjectOut:
    _check_profile(store, body.default_profile_id)
    p = projects.new_project(body.name, slug=body.slug, brand_notes=body.brand_notes, style_notes=body.style_notes,
                             default_profile_id=body.default_profile_id, export_root=body.export_root,
                             describe_hint=body.describe_hint)
    return _out(store, _save(store, p))


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: str, store: Store = Depends(get_store), describer: AssetDescriber | None = Depends(get_describer),
) -> ProjectOut:
    return _out(store, _load(store, project_id), describer)


@router.put("/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: str, patch: ProjectPatch, store: Store = Depends(get_store),
    describer: AssetDescriber | None = Depends(get_describer),
) -> ProjectOut:
    if patch.default_profile_id is not None:
        _check_profile(store, patch.default_profile_id)
    with projects.editing(store, project_id) as p:
        if p is None:
            raise ApiError(404, "not_found", f"no such project {project_id!r}")
        for k, v in patch.model_dump(exclude_none=True).items():
            setattr(p, k, v)
        return _out(store, _save(store, p), describer)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str, delete_files: bool = Query(False), store: Store = Depends(get_store)
) -> None:
    p = _load(store, project_id)
    store.delete_project(project_id)
    if delete_files:
        import shutil

        shutil.rmtree(projects.project_dir(p), ignore_errors=True)


@router.get("/{project_id}/jobs", response_model=list[JobSummaryOut])
async def project_jobs(project_id: str, store: Store = Depends(get_store)) -> list[JobSummaryOut]:
    _load(store, project_id)
    return [
        JobSummaryOut(id=s.id, status=s.status.value, idea=s.idea, updated_at=s.updated_at,
                      project_id=s.project_id, kind=s.kind, profile_id=s.profile_id)
        for s in store.list_jobs(project_id=project_id)
    ]


# --- assets ---------------------------------------------------------------------------


@router.post("/{project_id}/assets", response_model=AssetOut, status_code=201)
async def upload_asset(
    project_id: str,
    file: UploadFile = File(...),
    kind: str = Form("other"),
    label: str = Form(""),
    store: Store = Depends(get_store),
    describer: AssetDescriber | None = Depends(get_describer),
) -> AssetOut:
    """Add a file to the project. Unless a label was given (a person already named it), the
    asset is queued for background description and gets a suggested label when Gemma is done."""
    if kind not in ASSET_KINDS:
        raise ApiError(422, "validation_error", f"kind must be one of {ASSET_KINDS}")
    data = await file.read()
    if not data:
        raise ApiError(400, "bad_request", "empty upload")
    with projects.editing(store, project_id) as p:
        if p is None:
            raise ApiError(404, "not_found", f"no such project {project_id!r}")
        a = projects.add_asset(p, data, kind=kind, name=file.filename or "asset", label=label)
        _save(store, p)
    if describer is not None and describer.auto and not a.label:
        describer.enqueue(p.id, [a.id])
    return _asset_out(p, a, describer)


class DescribeRequest(BaseModel):
    asset_ids: list[str] | None = None   # None = every asset in the project
    force: bool = False                  # also redo labels Gemma already suggested


class DescribeResult(BaseModel):
    queued: int


@router.post("/{project_id}/assets/describe", response_model=DescribeResult, status_code=202)
async def describe_assets(
    project_id: str, body: DescribeRequest, store: Store = Depends(get_store),
    describer: AssetDescriber | None = Depends(get_describer),
) -> DescribeResult:
    """Queue assets for background description: unlabelled ones by default, suggested (``auto``)
    ones too with ``force``. Labels a person wrote are never touched."""
    if describer is None:
        raise ApiError(409, "conflict", "background description is not available in this server")
    p = _load(store, project_id)
    wanted = set(body.asset_ids) if body.asset_ids is not None else None
    if wanted is not None and (unknown := wanted - {a.id for a in p.assets}):
        raise ApiError(404, "not_found", f"no such asset {sorted(unknown)[0]!r}")
    ids = [a.id for a in p.assets
           if (wanted is None or a.id in wanted) and a.label_source != "human" and (body.force or not a.label.strip())]
    return DescribeResult(queued=describer.enqueue(p.id, ids))


@router.get("/{project_id}/assets/{asset_id}/file")
async def asset_file(project_id: str, asset_id: str, store: Store = Depends(get_store)):
    p = _load(store, project_id)
    a = p.asset(asset_id)
    if a is None or not Path(a.path).is_file():
        raise ApiError(404, "not_found", "no such asset")
    return FileResponse(a.path, filename=a.name)


_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


@router.get("/{project_id}/assets/{asset_id}/thumb")
def asset_thumb(project_id: str, asset_id: str, store: Store = Depends(get_store)):
    """A small cached JPEG of the asset for pickers and grids (320 px long side).

    A plain ``def`` route: FastAPI runs it in its threadpool, so the one-time ffmpeg call
    never blocks the event loop. Built under a temporary name and renamed into place, so a
    burst of first requests never serves a half-written file. Assets never change (a new
    upload is a new id), so the response may be cached forever. Anything ffmpeg cannot
    shrink (already small, not an image) is served as the original file.
    """
    p = _load(store, project_id)
    a = p.asset(asset_id)
    if a is None or not Path(a.path).is_file():
        raise ApiError(404, "not_found", "no such asset")
    dest = projects.thumb_path(p, a.id)
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.stem}.{uuid.uuid4().hex}.jpg")
        try:
            out = media.downscale(a.path, max_side=projects.THUMB_SIDE, dest=tmp)
            if out == tmp:
                os.replace(tmp, dest)
        except (media.MediaError, OSError):
            pass
        finally:
            tmp.unlink(missing_ok=True)
    if dest.is_file():
        return FileResponse(dest, media_type="image/jpeg", headers=_IMMUTABLE)
    return FileResponse(a.path, filename=a.name, headers=_IMMUTABLE)


class AssetPatch(BaseModel):
    label: str | None = None
    kind: Literal["master_face", "reference", "product", "logo", "other"] | None = None

    @model_validator(mode="after")
    def _something(self) -> "AssetPatch":
        if self.label is None and self.kind is None:
            raise ValueError("give a label, a kind, or both")
        return self


@router.patch("/{project_id}/assets/{asset_id}", response_model=AssetOut)
async def update_asset(
    project_id: str, asset_id: str, body: AssetPatch, store: Store = Depends(get_store),
    describer: AssetDescriber | None = Depends(get_describer),
) -> AssetOut:
    """Rename an asset's label (e.g. a face as "Maria, age 8") and/or change its kind — labels
    and kinds are what the storyboard reads when it assigns identities and references to scenes.

    Saving a label marks it ``human``, which both clears the "AI" flag in the UI and makes the
    label immune to later descriptions. Confirming an unchanged suggestion is the same call.
    Changing the kind rebuilds a still-suggested label for that kind; scenes already using the
    asset are not changed."""
    with projects.editing(store, project_id) as p:
        if p is None:
            raise ApiError(404, "not_found", f"no such project {project_id!r}")
        try:
            a = None
            if body.kind is not None:
                a = projects.set_asset_kind(p, asset_id, body.kind)
            if body.label is not None:
                a = projects.set_asset_label(p, asset_id, body.label)
        except KeyError:
            raise ApiError(404, "not_found", "no such asset")
        _save(store, p)
        return _asset_out(p, a, describer)


@router.delete("/{project_id}/assets/{asset_id}", status_code=204)
async def delete_asset(project_id: str, asset_id: str, store: Store = Depends(get_store)) -> None:
    with projects.editing(store, project_id) as p:
        if p is None:
            raise ApiError(404, "not_found", f"no such project {project_id!r}")
        if not projects.remove_asset(p, asset_id):
            raise ApiError(404, "not_found", "no such asset")
        _save(store, p)
