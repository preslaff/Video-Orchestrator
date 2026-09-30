"""Job routes: create / list / get / delete / decisions / provenance / edit scenes / approve."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Response

from .. import media
from ..clients import tts
from .. import projects as projects_mod
from .. import provenance
from .. import versions as versions_mod
from ..describer import AssetDescriber
from ..envcheck import EnvironmentNotReady
from ..queue import JobManager
from ..state import JobStatus, Store
from .deps import get_describer, get_manager, get_store
from .errors import ApiError
from .public import job_to_public, media_url, provenance_rows
from .projects import AssetOut, asset_out
from .schemas import (
    DecisionOut, Gate, JobBudget, JobPublic, JobSpec, JobSummaryOut, Page, ProvenanceOut, SaveSceneAsset, SceneEdits,
    SaveCut, SceneInsert, SceneLocks, SceneOrder, SceneRegenerate, VersionDetail,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _load(store: Store, job_id: str):
    job = store.load_job(job_id)
    if job is None:
        raise ApiError(404, "not_found", "no such job")
    return job


@router.post("", response_model=JobPublic)
async def create_job(
    spec: JobSpec,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    try:
        job = await manager.submit(
            spec.idea,
            style=spec.style,
            master_face=spec.master_face,
            autonomy=spec.autonomy,
            n_scenes=spec.n_scenes,
            seed=spec.seed,
            max_attempts=spec.max_attempts,
            evaluate=spec.evaluate,
            face_swap=spec.face_swap,
            project_id=spec.project_id,
            profile_id=spec.profile_id,
            profile_overrides=spec.profile_overrides,
            mode=spec.mode,
            reference_asset_ids=spec.reference_asset_ids,
            reference_images=spec.reference_images,
            keyframe_asset_ids=spec.keyframe_asset_ids,
            budget_usd=spec.budget_usd,
            force=spec.force,
        )
    except EnvironmentNotReady as e:
        raise ApiError(409, "environment_not_ready", str(e), details=e.report.to_dict())
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise ApiError(400, "bad_request", str(e).strip("'"))
    return job_to_public(job, store, manager)


@router.get("", response_model=Page[JobSummaryOut])
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: JobStatus | None = None,
    project_id: str | None = None,
    unassigned: bool = Query(False, description="only jobs without a project"),
    store: Store = Depends(get_store),
) -> Page[JobSummaryOut]:
    if unassigned and project_id:
        raise ApiError(400, "bad_request", "project_id and unassigned are mutually exclusive")
    items = [
        JobSummaryOut(id=s.id, status=s.status.value, idea=s.idea, updated_at=s.updated_at,
                      project_id=s.project_id, kind=s.kind, profile_id=s.profile_id, profile_name=s.profile_name)
        for s in store.list_jobs(limit=limit, offset=offset, status=status, project_id=project_id, unassigned=unassigned)
    ]
    total = store.count_jobs(status=status, project_id=project_id, unassigned=unassigned)
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get("/{job_id}", response_model=JobPublic)
async def get_job(
    job_id: str,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    return job_to_public(_load(store, job_id), store, manager)


@router.delete("/{job_id}", status_code=204)
async def delete_job(
    job_id: str,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> None:
    _load(store, job_id)
    if manager.is_active(job_id):
        raise ApiError(409, "conflict", f"job {job_id} is running")
    store.delete_job(job_id)


@router.get("/{job_id}/decisions", response_model=list[DecisionOut])
async def get_decisions(job_id: str, store: Store = Depends(get_store)) -> list[DecisionOut]:
    _load(store, job_id)
    return [
        DecisionOut(
            scene_index=d.scene_index, phase=d.phase, attempt=d.attempt,
            action=d.action, score=d.score, reason=d.reason, created_at=d.created_at,
            scene_id=d.scene_id,
        )
        for d in store.decisions(job_id)
    ]


@router.get("/{job_id}/provenance", response_model=list[ProvenanceOut])
async def get_provenance(job_id: str, store: Store = Depends(get_store)) -> list[ProvenanceOut]:
    return provenance_rows(_load(store, job_id))


@router.put("/{job_id}/scenes", response_model=JobPublic)
async def update_scenes(
    job_id: str,
    edits: SceneEdits,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    try:
        job = manager.update_scenes(job_id, [e.model_dump() for e in edits.scenes])
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.get("/{job_id}/scenes/{index}/versions/{version_id}", response_model=VersionDetail)
async def version_detail(
    job_id: str, index: int, version_id: str, store: Store = Depends(get_store),
) -> VersionDetail:
    """What made this take: the prompts saved with it, plus whatever the sidecar manifest adds.

    Read on demand rather than on every job fetch — a 40-scene job has hundreds of versions, and
    this opens one file.
    """
    job = _load(store, job_id)
    scene = next((s for s in job.scenes if s.index == index), None)
    if scene is None:
        raise ApiError(404, "not_found", f"no scene {index}")
    versions_mod.adopt_legacy(scene)      # a pre-versions job still has its current media
    version = next((v for v in scene.variants if v.id == version_id), None)
    clip = None if version else next((c for c in scene.clips if c.id == version_id), None)
    if version is None and clip is None:
        raise ApiError(404, "not_found", f"no version {version_id!r} in scene {index + 1}")
    kind = "keyframe" if version is not None else "clip"
    path = version.path if version is not None else clip.video_path
    record = version if version is not None else clip
    manifest = provenance.read_manifest(path) or {}
    params = manifest.get("params", {}) if isinstance(manifest.get("params"), dict) else {}
    prompt, negative = record.prompt, record.negative
    source = "record" if prompt else ""
    if not prompt:                        # versions made before prompts were recorded
        prompt = str(params.get("positive", "") or "")
        negative = str(params.get("negative", "") or "")
        source = "manifest" if prompt else ""
    current = scene.keyframe_prompt if kind == "keyframe" else scene.animation_prompt
    return VersionDetail(
        version_id=version_id, kind=kind, path=path, url=media_url(path),
        created_at=record.created_at, batch=record.batch, seed=record.seed,
        prompt=prompt, negative=negative, prompt_from=source,
        source=getattr(record, "source", "generated"),
        params={k: v for k, v in params.items() if k not in ("positive", "negative")},
        workflow=str(manifest.get("workflow", "") or ""), template=str(manifest.get("template", "") or ""),
        backend=str(manifest.get("backend", "") or ""), has_manifest=bool(manifest),
        matches_scene_prompt=(prompt.strip() == (current or "").strip()),
    )


@router.post("/{job_id}/scenes", response_model=JobPublic, status_code=201)
async def add_scene(
    job_id: str,
    body: SceneInsert,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Insert a scene into the storyboard. With ``generate`` (the default) Gemma writes its
    keyframe and animation prompts in the background; the scene is returned at once, flagged
    ``writing_prompts``, and the next run renders only it."""
    try:
        job = manager.add_scene(job_id, body.position, body.model_dump(
            include={"description", "character", "setting", "mood"}), generate=body.generate)
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.delete("/{job_id}/scenes/{index}", response_model=JobPublic)
async def delete_scene(
    job_id: str,
    index: int,
    delete_files: bool = Query(False, description="also unlink the images and clips it generated"),
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Remove a scene and renumber the rest. Its generated files stay on disk unless asked."""
    try:
        job = manager.delete_scene(job_id, index, delete_files=delete_files)
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    return job_to_public(job, store, manager)


@router.put("/{job_id}/scenes/order", response_model=JobPublic)
async def reorder_scenes(
    job_id: str,
    body: SceneOrder,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Reorder the storyboard: the ids of every scene, in the order they should play."""
    try:
        job = manager.reorder_scenes(job_id, body.scene_ids)
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.post("/{job_id}/scenes/{index}/write-prompts", response_model=JobPublic, status_code=202)
async def write_prompts(
    job_id: str,
    index: int,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """(Re)write this scene's prompts from its description with Gemma — for a scene whose
    description changed, or whose first write failed."""
    try:
        job = manager.write_prompts(job_id, index)
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.post("/{job_id}/scenes/{index}/speak", response_model=JobPublic, status_code=201)
async def speak_scene(
    job_id: str,
    index: int,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Speak this scene's line in the cloned voice.

    Deliberate, never automatic: saving new text does not trigger it, you ask for it. It loads an
    11.3 GB model, so it waits for the GPU lock and the arbiter frees the card first. If a render
    is in flight this request waits for it rather than fighting it for VRAM.
    """
    try:
        job = await manager.speak_scene(job_id, index)
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except (ValueError, tts.TTSError) as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.delete("/{job_id}/scenes/{index}/narration", response_model=JobPublic)
async def remove_narration(
    job_id: str,
    index: int,
    keep_text: bool = True,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Throw away this scene's spoken take, so its clip is heard as it was recorded.

    Assembly uses whichever take exists, not the text, so emptying the words is not enough to
    silence a voice — this is. ``keep_text`` leaves the line in place to be re-spoken; pass
    ``keep_text=false`` to remove the line as well.
    """
    try:
        job = manager.clear_narration(job_id, index, keep_text=keep_text)
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    return job_to_public(job, store, manager)


@router.put("/{job_id}/budget", response_model=JobPublic)
async def set_budget(
    job_id: str,
    body: JobBudget,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Set the ceiling for hosted (API) model spending; null lifts it.

    A job paused on its budget needs this before its gate is approved — approving alone would
    stop it again at the very next billed call.
    """
    try:
        job = manager.set_budget(job_id, body.budget_usd)
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.put("/{job_id}/scenes/{index}/locks", response_model=JobPublic)
async def set_scene_locks(
    job_id: str,
    index: int,
    body: SceneLocks,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Mark the parts of this scene's frame the animator must not redraw — a brand label, a
    monogram — and restore them on the clip that already exists. A paste, not a render."""
    try:
        job = manager.set_locked_regions(job_id, index, [r.model_dump() for r in body.regions])
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.post("/{job_id}/scenes/{index}/cut", response_model=JobPublic, status_code=201)
async def save_cut(
    job_id: str,
    index: int,
    body: SaveCut,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Render a trimmed span to a file and keep it as one of the scene's takes.

    A trim in a timeline reaches nothing but `assembly.mp4`. Saved, the cut is a clip like any
    other — offered as a take, listed in the version strip, and with `make_active` the scene's
    deliverable, so the export folder carries the cut and not the whole render.
    """
    try:
        job = manager.save_cut(job_id, index, clip_id=body.clip_id, start=body.start,
                               end=body.end, make_active=body.make_active)
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except (ValueError, media.MediaError) as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.post("/{job_id}/scenes/{index}/regenerate", response_model=JobPublic)
async def regenerate_scene(
    job_id: str,
    index: int,
    body: SceneRegenerate,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    _load(store, job_id)
    edits = body.model_dump(exclude={"what"}, exclude_none=True)
    try:
        job = await manager.regenerate(job_id, index, what=body.what, edits=edits)
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.post("/{job_id}/scenes/{index}/save-asset", response_model=AssetOut, status_code=201)
async def save_scene_asset(
    job_id: str,
    index: int,
    body: SaveSceneAsset,
    response: Response,
    store: Store = Depends(get_store),
    describer: AssetDescriber | None = Depends(get_describer),
) -> AssetOut:
    """Copy a scene's keyframe into the job's project as an asset, so other scenes can use it
    as a reference (the picker only offers project assets).

    A copy, not a pointer: regenerating the source scene never changes a scene that uses it.
    Importing the same frame twice returns the asset already holding those bytes (200).
    """
    job = _load(store, job_id)
    if not job.project_id:
        raise ApiError(400, "bad_request", "the job has no project: save it to a project to reuse the image")
    scene = next((s for s in job.scenes if s.index == index), None)
    if scene is None:
        raise ApiError(404, "not_found", f"no scene {index}")
    if body.version_id:
        try:
            variant = versions_mod.variant_by(scene, id=body.version_id)
        except ValueError as e:
            raise ApiError(404, "not_found", str(e))
        source = variant.swapped_path or variant.path
    else:
        source = scene.swapped_path or scene.keyframe_path
    if not source or not Path(source).is_file():
        raise ApiError(404, "not_found", f"scene {index + 1} has no keyframe image yet")
    digest = projects_mod.file_sha256(source)
    with projects_mod.editing(store, job.project_id) as p:
        if p is None:
            raise ApiError(404, "not_found", "no such project")
        existing = projects_mod.find_asset_by_sha256(p, digest)
        a = existing or projects_mod.add_asset(
            p, source, kind=body.kind, name=f"scene-{index + 1}-keyframe{Path(source).suffix or '.png'}",
            label=body.label,
        )
        if existing is None:
            projects_mod.save(store, p)
    if existing is not None:
        response.status_code = 200
    elif describer is not None and describer.auto and not a.label:
        describer.enqueue(p.id, [a.id])       # a caption makes it findable and suggestible
    return asset_out(p, a, describer)


@router.post("/{job_id}/stop", response_model=JobPublic, status_code=202)
async def stop(
    job_id: str,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Ask a running job to stop at its next safe point.

    Accepted, not finished: the job keeps status ``running`` with ``stopping: true``
    until the worker reaches a checkpoint (a ComfyUI step is withdrawn within a poll
    interval; a Gemma call finishes first), then ends ``cancelled`` and can be edited
    and resumed with ``POST /retry``.
    """
    _load(store, job_id)
    try:
        job = manager.stop(job_id)
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    return job_to_public(job, store, manager)


@router.delete("/{job_id}/scenes/{index}/versions/{version_id}", response_model=JobPublic)
async def delete_scene_version(
    job_id: str,
    index: int,
    version_id: str,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Drop one keyframe or clip version from a scene's history.

    The generated file is kept on disk; only the reference is removed. Deleting the active
    version falls back to the newest remaining one (for a keyframe, its clip follows)."""
    _load(store, job_id)
    try:
        job = manager.delete_version(job_id, index, version_id)
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store, manager)


@router.post("/{job_id}/retry", response_model=JobPublic)
async def retry(
    job_id: str,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Resume a failed, stopped or orphaned job from where it stopped; finished work is kept."""
    _load(store, job_id)
    try:
        job = await manager.retry(job_id)
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    return job_to_public(job, store, manager)


@router.post("/{job_id}/approve/{gate}", response_model=JobPublic)
async def approve(
    job_id: str,
    gate: Gate,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    job = _load(store, job_id)
    if job.status is not JobStatus.NEEDS_APPROVAL:
        raise ApiError(409, "conflict", f"job is {job.status.value}, not awaiting approval")
    try:
        job = await manager.approve(job_id, gate)
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    return job_to_public(job, store, manager)
