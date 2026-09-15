"""Public (API) projections of persisted state.

``state.job_to_dict`` is the *storage* shape (it is what lands in the JSONB
column), so anything the UI needs beyond it — media URLs, the pending gate,
provenance rows — is added here and never written back.
"""

from __future__ import annotations

from pathlib import Path

from .. import config, provenance
from ..evaluator import Verdict
from ..orchestrator import Orchestrator
from ..steps.edit import reference_slots
from ..state import Job, Scene, Store, Variant, _scene_to_dict, job_to_dict
from .schemas import JobPublic, ProvenanceOut, ScenePublic, VariantOut, VerdictOut


def media_url(path: str | None) -> str | None:
    """``/media/<path relative to COMFY_OUTPUT>`` or None when outside it."""
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(config.COMFY_OUTPUT.resolve())
    except ValueError:
        return None
    return "/media/" + rel.as_posix()


def verdict_out(v: Verdict | None) -> VerdictOut | None:
    if v is None:
        return None
    return VerdictOut(decision=v.decision, needs_review=v.needs_review, **v.model_dump())


def variant_out(v: Variant) -> VariantOut:
    return VariantOut(
        path=v.path, url=media_url(v.path), seed=v.seed, verdict=verdict_out(v.verdict),
        swapped_path=v.swapped_path, swapped_url=media_url(v.swapped_path), post_paths=list(v.post_paths),
    )


def scene_to_public(s: Scene, project_id: str | None = None) -> ScenePublic:
    d = _scene_to_dict(s)
    d["variants"] = [variant_out(v) for v in s.variants]
    d.update(
        identity_url=(f"/api/projects/{project_id}/assets/{s.identity_asset_id}/file"
                      if project_id and s.identity_asset_id else None),
        keyframe_url=media_url(s.keyframe_path),
        video_url=media_url(s.video_path),
        swapped_url=media_url(s.swapped_path),
        keyframe_verdict=verdict_out(s.keyframe_verdict),
        video_verdict=verdict_out(s.video_verdict),
        swap_verdict=verdict_out(s.swap_verdict),
    )
    return ScenePublic.model_validate(d)


def _edit_reference_slots(job: Job, store: Store) -> int:
    """References the job's edit workflow uses (0 = unknown); the UI says "the first N are used"."""
    try:
        return reference_slots(Orchestrator.from_job(store, job).params.edit_workflow)
    except Exception:  # a job whose profile no longer validates must still be viewable
        return 0


def job_to_public(job: Job, store: Store, manager=None) -> JobPublic:
    """``manager`` (a ``JobManager``) adds the live run state: whether a segment is in
    flight in this server (``active``) and whether a stop is pending (``stopping``).
    A job persisted as running but not active here is an orphan left by a restart."""
    d = job_to_dict(job)
    d["scenes"] = [scene_to_public(s, job.project_id) for s in job.scenes]
    d["pending_gate"] = Orchestrator.pending_gate(job, store)
    d["edit_reference_slots"] = _edit_reference_slots(job, store)
    if manager is not None:
        d["active"] = manager.is_active(job.id)
        d["stopping"] = manager.is_stopping(job.id)
    return JobPublic.model_validate(d)


def provenance_rows(job: Job) -> list[ProvenanceOut]:
    """Seed + prompt per artifact, read from the sidecar manifests."""
    rows: list[ProvenanceOut] = []
    for s in job.scenes:
        for kind, path in (
            ("keyframe", s.keyframe_path),
            ("swapped", s.swapped_path),
            ("video", s.video_path),
        ):
            if not path:
                continue
            m = provenance.read_manifest(path) or {}
            p = m.get("params", {})
            rows.append(
                ProvenanceOut(
                    scene_index=s.index,
                    kind=kind,
                    filename=Path(path).name,
                    url=media_url(path),
                    seed=p.get("seed", p.get("seed_high")),
                    prompt=p.get("positive"),
                    template=m.get("template"),
                    has_manifest=bool(m),
                )
            )
    return rows
