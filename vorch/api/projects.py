"""Projects / campaigns: CRUD, assets (multipart upload, download, delete), jobs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import profiles, projects
from ..projects import ASSET_KINDS, Project
from ..state import Store
from .deps import get_store
from .errors import ApiError
from .schemas import JobSummaryOut

router = APIRouter(prefix="/api/projects", tags=["projects"])


class AssetOut(BaseModel):
    id: str
    kind: str
    name: str
    label: str = ""
    sha256: str = ""
    created_at: str = ""
    path: str
    url: str


class ProjectOut(BaseModel):
    id: str
    slug: str
    name: str
    brand_notes: str = ""
    style_notes: str = ""
    default_profile_id: str
    export_root: str = ""
    export_root_effective: str
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


class ProjectPatch(BaseModel):
    name: str | None = None
    brand_notes: str | None = None
    style_notes: str | None = None
    default_profile_id: str | None = None
    export_root: str | None = None


def _asset_out(project: Project, a) -> AssetOut:
    return AssetOut(id=a.id, kind=a.kind, name=a.name, label=a.label, sha256=a.sha256, created_at=a.created_at,
                    path=a.path, url=f"/api/projects/{project.id}/assets/{a.id}/file")


def _out(store: Store, p: Project) -> ProjectOut:
    return ProjectOut(
        id=p.id, slug=p.slug, name=p.name, brand_notes=p.brand_notes, style_notes=p.style_notes,
        default_profile_id=p.default_profile_id, export_root=p.export_root,
        export_root_effective=str(projects.export_root(p)),
        assets=[_asset_out(p, a) for a in p.assets], created_at=p.created_at, updated_at=p.updated_at,
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
                             default_profile_id=body.default_profile_id, export_root=body.export_root)
    return _out(store, _save(store, p))


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(project_id: str, store: Store = Depends(get_store)) -> ProjectOut:
    return _out(store, _load(store, project_id))


@router.put("/{project_id}", response_model=ProjectOut)
async def update_project(project_id: str, patch: ProjectPatch, store: Store = Depends(get_store)) -> ProjectOut:
    p = _load(store, project_id)
    if patch.default_profile_id is not None:
        _check_profile(store, patch.default_profile_id)
    for k, v in patch.model_dump(exclude_none=True).items():
        setattr(p, k, v)
    return _out(store, _save(store, p))


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
) -> AssetOut:
    p = _load(store, project_id)
    if kind not in ASSET_KINDS:
        raise ApiError(422, "validation_error", f"kind must be one of {ASSET_KINDS}")
    data = await file.read()
    if not data:
        raise ApiError(400, "bad_request", "empty upload")
    a = projects.add_asset(p, data, kind=kind, name=file.filename or "asset", label=label)
    _save(store, p)
    return _asset_out(p, a)


@router.get("/{project_id}/assets/{asset_id}/file")
async def asset_file(project_id: str, asset_id: str, store: Store = Depends(get_store)):
    p = _load(store, project_id)
    a = p.asset(asset_id)
    if a is None or not Path(a.path).is_file():
        raise ApiError(404, "not_found", "no such asset")
    return FileResponse(a.path, filename=a.name)


@router.delete("/{project_id}/assets/{asset_id}", status_code=204)
async def delete_asset(project_id: str, asset_id: str, store: Store = Depends(get_store)) -> None:
    p = _load(store, project_id)
    if not projects.remove_asset(p, asset_id):
        raise ApiError(404, "not_found", "no such asset")
    _save(store, p)
