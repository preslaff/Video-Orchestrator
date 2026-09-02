"""GPU-free checks for state.py — job persistence + decision log.

Exercises the SQLite store with a temp DB file: Job/Scene/Verdict roundtrip,
updates, listing, cascade delete, and the append-only decision log. Persistence
across reopen proves the data is on disk, not just in the connection.
"""

from __future__ import annotations

import threading

import pytest

from vorch.evaluator import Verdict
from vorch.state import (
    Decision,
    Job,
    JobStatus,
    Scene,
    SceneStatus,
    Store,
    new_job,
)


def _verdict(verdict="accept", score=8, fix="") -> Verdict:
    # Keep objective fields consistent with the intended decision so the
    # objective gate (evaluator.Verdict.decision) yields `verdict`.
    ok = verdict == "accept"
    return Verdict(
        action_correct=ok,
        identity_match=True,
        style_match=True,
        artifacts=not ok,
        composition_ok=True,
        score=score,
        verdict=verdict,
        fix_suggestion=fix,
    )


def test_new_job_has_id_and_timestamps():
    j = new_job(idea="a woman kneading bread", style="oil painting")
    assert j.id and j.created_at and j.updated_at == j.created_at
    assert j.status is JobStatus.PENDING


def test_job_roundtrip_preserves_scene_and_verdict(store):
    job = new_job(idea="bread", style="oil", master_face="face.jpg")
    job.scenes = [
        Scene(
            index=0,
            description="woman kneading dough",
            keyframe_prompt="oil painting, kneading bread",
            keyframe_path="kf0.png",
            keyframe_verdict=_verdict("retry", 3, "fix the hands"),
            status=SceneStatus.KEYFRAME,
        )
    ]
    store.save_job(job)

    loaded = store.load_job(job.id)
    assert loaded is not None
    assert loaded.idea == "bread" and loaded.master_face == "face.jpg"
    s = loaded.scenes[0]
    assert s.description == "woman kneading dough"
    assert s.status is SceneStatus.KEYFRAME
    assert s.keyframe_verdict is not None
    assert s.keyframe_verdict.verdict == "retry"
    assert s.keyframe_verdict.fix_suggestion == "fix the hands"


def test_run_config_persists(store):
    """seed/max_attempts/evaluate/face_swap/n_scenes survive a save/load round-trip."""
    job = new_job(idea="x", n_scenes=3, seed=11, max_attempts=4, evaluate=False, face_swap=False)
    store.save_job(job)
    loaded = store.load_job(job.id)
    assert loaded.n_scenes == 3 and loaded.seed == 11
    assert loaded.max_attempts == 4 and loaded.evaluate is False
    assert loaded.face_swap is False


def test_save_updates_in_place_and_bumps_updated_at(store):
    job = new_job(idea="x")
    store.save_job(job)
    first = job.updated_at

    job.status = JobStatus.RUNNING
    job.scenes.append(Scene(index=0, video_path="v0.mp4", status=SceneStatus.VIDEO))
    store.save_job(job)

    loaded = store.load_job(job.id)
    assert loaded.status is JobStatus.RUNNING
    assert loaded.scenes[0].video_path == "v0.mp4"
    assert loaded.updated_at >= first  # re-stamped on save
    # Still a single row — update, not insert.
    assert len(store.list_jobs()) == 1


def test_list_jobs_returns_summaries(store):
    a = new_job(idea="first")
    b = new_job(idea="second")
    store.save_job(a)
    store.save_job(b)
    summaries = store.list_jobs()
    assert {s.idea for s in summaries} == {"first", "second"}
    assert all(s.status is JobStatus.PENDING for s in summaries)


def test_load_missing_job_returns_none(store):
    assert store.load_job("nope") is None


def test_decision_log_is_ordered_and_ided(store):
    job = new_job()
    store.save_job(job)
    d1 = store.log_decision(
        Decision(job_id=job.id, scene_index=0, phase="keyframe", attempt=1, action="retry")
    )
    d2 = store.log_decision(
        Decision(job_id=job.id, scene_index=0, phase="keyframe", attempt=2, action="accept")
    )
    assert d1.id is not None and d2.id is not None and d2.id > d1.id

    log = store.decisions(job.id)
    assert [d.action for d in log] == ["retry", "accept"]
    assert [d.attempt for d in log] == [1, 2]


def test_record_verdict_maps_fields(store):
    job = new_job()
    store.save_job(job)
    store.record_verdict(job.id, 0, "keyframe", _verdict("retry", 4, "hands deformed"), attempt=1)
    (d,) = store.decisions(job.id)
    assert d.action == "retry" and d.score == 4 and d.reason == "hands deformed"


def test_delete_job_cascades_decisions(store):
    job = new_job()
    store.save_job(job)
    store.record_verdict(job.id, 0, "keyframe", _verdict())
    assert store.decisions(job.id)

    store.delete_job(job.id)
    assert store.load_job(job.id) is None
    assert store.decisions(job.id) == []  # cascaded away


def test_concurrent_readers_and_writers_do_not_misuse(store):
    """Writer (executor) + reader (UI) threads share one pooled Store safely."""
    from vorch.state import Decision

    job = new_job(idea="x")
    store.save_job(job)
    errors: list[Exception] = []

    def writer():
        try:
            for i in range(60):
                store.log_decision(
                    Decision(job_id=job.id, scene_index=0, phase="keyframe",
                             attempt=i, action="generated")
                )
                store.save_job(job)
        except Exception as e:  # noqa: BLE001 - capturing for the assertion
            errors.append(e)

    def reader():
        try:
            for _ in range(60):
                store.decisions(job.id)
                store.load_job(job.id)
                store.list_jobs()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors


def test_persists_across_reopen(make_store):
    with make_store() as s:
        job = new_job(idea="durable")
        job.scenes = [Scene(index=0, keyframe_verdict=_verdict("accept", 9))]
        s.save_job(job)
        s.record_verdict(job.id, 0, "keyframe", _verdict("accept", 9))
        job_id = job.id

    # Fresh Store (new pool), same schema — data must still be there.
    with make_store() as s2:
        loaded = s2.load_job(job_id)
        assert loaded is not None and loaded.idea == "durable"
        assert loaded.scenes[0].keyframe_verdict.score == 9
        assert len(s2.decisions(job_id)) == 1


# --- PostgreSQL specifics -----------------------------------------------------


def test_migrations_applied_once_and_idempotent(make_store):
    from vorch import db

    s1 = make_store()
    assert s1.schema_version() == db.latest_version() >= 1
    # A second Store on the same schema finds nothing pending and does not fail.
    s2 = make_store()
    assert s2.schema_version() == s1.schema_version()
    assert len(s2.decisions("nope")) == 0


def test_jsonb_roundtrip_unicode(store):
    job = new_job(idea="жена меси хляб — „impasto“", style="маслени бои")
    job.scenes = [Scene(index=0, description="кухня, топла светлина", keyframe_prompt="x")]
    store.save_job(job)
    loaded = store.load_job(job.id)
    assert loaded.idea == job.idea and loaded.style == job.style
    assert loaded.scenes[0].description == "кухня, топла светлина"


def test_timestamps_roundtrip_iso(store):
    from datetime import datetime

    job = store.save_job(new_job(idea="t"))
    summary = store.list_jobs()[0]
    # Stored as timestamptz, read back as the same ISO-8601 UTC string.
    assert summary.updated_at == job.updated_at
    assert datetime.fromisoformat(summary.updated_at).utcoffset().total_seconds() == 0
    d = store.log_decision(Decision(job_id=job.id, scene_index=None, phase="p", attempt=1, action="a"))
    assert store.decisions(job.id)[0].created_at == d.created_at
