"""GPU-free checks for state.py — job persistence + decision log.

Exercises the SQLite store with a temp DB file: Job/Scene/Verdict roundtrip,
updates, listing, cascade delete, and the append-only decision log. Persistence
across reopen proves the data is on disk, not just in the connection.
"""

from __future__ import annotations

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
    return Verdict(
        action_correct=True,
        identity_match=True,
        style_match=True,
        artifacts=False,
        composition_ok=True,
        score=score,
        verdict=verdict,
        fix_suggestion=fix,
    )


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "vorch.db") as s:
        yield s


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


def test_persists_across_reopen(tmp_path):
    db = tmp_path / "persist.db"
    with Store(db) as s:
        job = new_job(idea="durable")
        job.scenes = [Scene(index=0, keyframe_verdict=_verdict("accept", 9))]
        s.save_job(job)
        s.record_verdict(job.id, 0, "keyframe", _verdict("accept", 9))
        job_id = job.id

    # Fresh Store, same file — data must still be there.
    with Store(db) as s2:
        loaded = s2.load_job(job_id)
        assert loaded is not None and loaded.idea == "durable"
        assert loaded.scenes[0].keyframe_verdict.score == 9
        assert len(s2.decisions(job_id)) == 1
