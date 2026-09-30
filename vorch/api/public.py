"""Public (API) projections of persisted state.

``state.job_to_dict`` is the *storage* shape (it is what lands in the JSONB
column), so anything the UI needs beyond it — media URLs, the pending gate,
provenance rows — is added here and never written back.
"""

from __future__ import annotations

from pathlib import Path

from .. import budget, config, provenance
from .. import versions as versions_mod
from ..evaluator import Verdict
from ..orchestrator import Orchestrator
from ..steps.edit import reference_slots
from ..state import Clip, Job, Scene, Store, Variant, _scene_to_dict, job_to_dict
from .schemas import ClipOut, JobPublic, ProvenanceOut, ScenePublic, VariantOut, VerdictOut


def media_url(path: str | None, *, bust: bool = False) -> str | None:
    """``/media/<path relative to COMFY_OUTPUT>`` or None when outside it.

    ``bust`` appends ``?v=<mtime><size>`` so a browser refetches after the file is rewritten. Most
    generated media does not need it — ComfyUI numbers every output, so a regenerated keyframe
    arrives under a new name and the URL changes by itself. Narration is the exception: a scene's
    take is deliberately written to one path, so re-speaking a line left the URL identical and the
    browser went on playing the previous take from cache (and a cached byte range against a file of
    a different length answers 416).
    """
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(config.COMFY_OUTPUT.resolve())
    except ValueError:
        return None
    url = "/media/" + rel.as_posix()
    if bust:
        try:
            st = Path(path).stat()
            return f"{url}?v={int(st.st_mtime)}{st.st_size}"
        except OSError:
            pass
    return url


def verdict_out(v: Verdict | None) -> VerdictOut | None:
    if v is None:
        return None
    return VerdictOut(decision=v.decision, needs_review=v.needs_review, **v.model_dump())


def variant_out(v: Variant, *, active: bool = False) -> VariantOut:
    return VariantOut(
        path=v.path, url=media_url(v.path), seed=v.seed, verdict=verdict_out(v.verdict),
        swapped_path=v.swapped_path, swapped_url=media_url(v.swapped_path), post_paths=list(v.post_paths),
        source=v.source, asset_id=v.asset_id, id=v.id, created_at=v.created_at, batch=v.batch, active=active,
    )


def clip_out(c: Clip, *, active: bool = False) -> ClipOut:
    return ClipOut(
        id=c.id, video_path=c.video_path, url=media_url(c.video_path),
        raw_video_path=c.raw_video_path, raw_url=media_url(c.raw_video_path),
        post_paths=list(c.post_paths), seed=c.seed, verdict=verdict_out(c.verdict),
        keyframe_path=c.keyframe_path, created_at=c.created_at, batch=c.batch, active=active,
    )


def scene_to_public(s: Scene, project_id: str | None = None) -> ScenePublic:
    versions_mod.adopt_legacy(s)  # jobs from before versions still show their media in the strips
    d = _scene_to_dict(s)
    d["variants"] = [variant_out(v, active=v.path == s.keyframe_path) for v in s.variants]
    d["clips"] = [clip_out(c, active=c.video_path == s.video_path) for c in s.clips]
    chosen = next((v for v in s.variants if v.path == s.keyframe_path), None)
    if chosen is not None and chosen.source == "user":
        d["keyframe_source"], d["keyframe_asset_id"] = "user", chosen.asset_id
    d.update(
        identity_url=(f"/api/projects/{project_id}/assets/{s.identity_asset_id}/file"
                      if project_id and s.identity_asset_id else None),
        keyframe_url=media_url(s.keyframe_path),
        narration_url=media_url(s.narration_path, bust=True),
        narration_stale=bool(s.narration.strip()) and s.narration.strip() != s.narration_said.strip(),
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
    _spend = budget.spent(job)
    d["spend_usd"] = _spend.max_usd          # the worst case, which is what the ceiling reserves
    d["spend_calls"] = _spend.calls
    d["budget_reason"] = Orchestrator.budget_blocked(job, store) or ""
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
