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


def scene_to_public(s: Scene) -> ScenePublic:
    d = _scene_to_dict(s)
    d["variants"] = [variant_out(v) for v in s.variants]
    d.update(
        keyframe_url=media_url(s.keyframe_path),
        video_url=media_url(s.video_path),
        swapped_url=media_url(s.swapped_path),
        keyframe_verdict=verdict_out(s.keyframe_verdict),
        video_verdict=verdict_out(s.video_verdict),
        swap_verdict=verdict_out(s.swap_verdict),
    )
    return ScenePublic.model_validate(d)


def job_to_public(job: Job, store: Store) -> JobPublic:
    d = job_to_dict(job)
    d["scenes"] = [scene_to_public(s) for s in job.scenes]
    d["pending_gate"] = Orchestrator.pending_gate(job, store)
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
