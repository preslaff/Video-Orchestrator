"""GPU-free checks for queue.py — the async job manager + event broker.

No web framework and no real generation: a fake Orchestrator factory drives the
manager. Async is exercised via asyncio.run (no pytest-asyncio dependency).
"""

from __future__ import annotations

import asyncio

import pytest

from vorch.queue import Broker, JobManager
from vorch.state import JobStatus, Store, new_job


class FakeOrch:
    """Stand-in Orchestrator: emits a couple events and finishes the job."""

    def __init__(self, store, job):
        self.store = store
        self.emit = lambda event: None

    def run(self, job):
        self.emit({"job_id": job.id, "type": "status", "status": "running"})
        job.status = JobStatus.DONE
        self.store.save_job(job)
        self.emit({"job_id": job.id, "type": "status", "status": "done"})

    def approve(self, job, gate):
        self.emit({"job_id": job.id, "type": "approved", "gate": gate})
        job.status = JobStatus.DONE
        self.store.save_job(job)


class BoomOrch(FakeOrch):
    def run(self, job):
        raise RuntimeError("boom")


async def _drain(q, timeout=2.0):
    """Collect events until the terminal {'type': 'end'}."""
    events = []
    while True:
        e = await asyncio.wait_for(q.get(), timeout)
        events.append(e)
        if e["type"] == "end":
            return events


def test_broker_fans_out_and_unsubscribes():
    async def go():
        b = Broker()
        q = b.subscribe("j1")
        b.publish("j1", {"a": 1})
        b.publish("other", {"b": 2})  # no subscriber -> dropped
        got = q.get_nowait()
        b.unsubscribe("j1", q)
        b.publish("j1", {"a": 2})  # no subscribers now
        return got, q.empty()

    got, empty_after = asyncio.run(go())
    assert got["a"] == 1 and got["id"] == 1 and got["job_id"] == "j1"
    assert empty_after is True


def test_execute_streams_events_and_completes(tmp_path, make_store):
    async def go():
        store = make_store()
        job = new_job(idea="x")
        store.save_job(job)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        q = mgr.events(job.id)
        mgr._spawn(mgr._execute(job, lambda o, j: o.run(j)))
        events = await _drain(q)
        final = store.load_job(job.id).status
        store.close()
        return events, final

    events, final = asyncio.run(go())
    assert final is JobStatus.DONE
    statuses = [e.get("status") for e in events if e["type"] == "status"]
    assert statuses == ["running", "done"]
    assert events[-1]["type"] == "end"


def test_execute_reports_failure_as_state_and_event(tmp_path, make_store):
    async def go():
        store = make_store()
        job = new_job(idea="x")
        store.save_job(job)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: BoomOrch(s, j))
        q = mgr.events(job.id)
        mgr._spawn(mgr._execute(job, lambda o, j: o.run(j)))
        events = await _drain(q)
        final = store.load_job(job.id).status
        store.close()
        return events, final

    events, final = asyncio.run(go())
    assert final is JobStatus.FAILED
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["error"] == "boom"
    assert events[-1]["type"] == "end"


def test_submit_creates_validates_and_runs(tmp_path, make_store):
    master = tmp_path / "m.jpg"
    master.write_bytes(b"\xff\xd8\xff")

    async def go():
        store = make_store()
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        job = await mgr.submit("idea", master_face=str(master), autonomy="full", n_scenes=1)
        await asyncio.gather(*list(mgr._tasks))  # let the background run finish
        final = store.load_job(job.id).status
        store.close()
        return job, final

    job, final = asyncio.run(go())
    assert job.master_face.endswith("m.jpg") and job.n_scenes == 1
    assert final is JobStatus.DONE


def test_submit_no_master_ok_when_face_swap_off(tmp_path, make_store):
    async def go():
        store = make_store()
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        job = await mgr.submit("a landscape", autonomy="full", face_swap=False)  # no master
        await asyncio.gather(*list(mgr._tasks))
        final = store.load_job(job.id).status
        store.close()
        return job, final

    job, final = asyncio.run(go())
    assert job.face_swap is False and job.master_face == ""
    assert final is JobStatus.DONE


def test_submit_rejects_missing_master(tmp_path, make_store):
    async def go():
        store = make_store()
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        try:
            with pytest.raises(FileNotFoundError):
                await mgr.submit("idea", master_face="/no/such/face.jpg")
        finally:
            store.close()

    asyncio.run(go())


def test_update_scenes_applies_edits(tmp_path, make_store):
    from vorch.state import Scene

    async def go():
        store = make_store()
        job = new_job(idea="x")
        job.status = JobStatus.NEEDS_APPROVAL
        job.scenes = [Scene(index=0, keyframe_prompt="old kf", animation_prompt="old an")]
        store.save_job(job)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        mgr.update_scenes(job.id, [{"index": 0, "keyframe_prompt": "new kf", "description": "d"}])
        reloaded = store.load_job(job.id)
        store.close()
        return reloaded

    j = asyncio.run(go())
    assert j.scenes[0].keyframe_prompt == "new kf"
    assert j.scenes[0].description == "d"
    assert j.scenes[0].animation_prompt == "old an"  # untouched (not in edit)


def test_update_scenes_rejected_while_active(tmp_path, make_store):
    async def go():
        store = make_store()
        job = new_job(idea="x")
        store.save_job(job)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        mgr._active.add(job.id)  # simulate a run in flight
        try:
            with pytest.raises(RuntimeError):
                mgr.update_scenes(job.id, [{"index": 0, "keyframe_prompt": "x"}])
        finally:
            store.close()

    asyncio.run(go())


def test_duplicate_approve_ignored_while_active(tmp_path, make_store):
    """A second approve while a run segment is in flight is a no-op."""
    calls = {"n": 0}

    class CountingOrch(FakeOrch):
        def approve(self, job, gate):
            calls["n"] += 1

    async def go():
        store = make_store()
        job = new_job(idea="x")
        job.status = JobStatus.NEEDS_APPROVAL
        store.save_job(job)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: CountingOrch(s, j))
        mgr._active.add(job.id)  # simulate an in-flight run segment
        await mgr.approve(job.id, "storyboard")  # must be ignored
        store.close()
        return calls["n"], mgr

    n, mgr = asyncio.run(go())
    assert n == 0 and not mgr._tasks  # nothing spawned


def test_approve_loads_and_runs(tmp_path, make_store):
    async def go():
        store = make_store()
        job = new_job(idea="x", autonomy="semi")
        job.status = JobStatus.NEEDS_APPROVAL
        store.save_job(job)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        q = mgr.events(job.id)
        await mgr.approve(job.id, "storyboard")
        events = await _drain(q)
        final = store.load_job(job.id).status
        store.close()
        return events, final

    events, final = asyncio.run(go())
    assert any(e["type"] == "approved" and e["gate"] == "storyboard" for e in events)
    assert final is JobStatus.DONE


# --- broker ids / replay / global --------------------------------------------


def test_broker_ids_monotonic_per_job():
    b = Broker()
    b.publish("a", {"type": "x"})
    b.publish("b", {"type": "x"})
    b.publish("a", {"type": "y"})
    assert (b.last_id("a"), b.last_id("b"), b.last_id("zzz")) == (2, 1, 0)


def test_broker_replays_after_id():
    async def go():
        b = Broker()
        for i in range(4):
            b.publish("j", {"type": "status", "n": i})
        q = b.subscribe("j", after=2)
        replayed = [q.get_nowait()["id"] for _ in range(q.qsize())]
        fresh = b.subscribe("j")  # no replay by default
        return replayed, fresh.qsize()

    replayed, fresh_size = asyncio.run(go())
    assert replayed == [3, 4] and fresh_size == 0


def test_broker_ring_buffer_bounded():
    async def go():
        b = Broker(history=3)
        for i in range(10):
            b.publish("j", {"type": "status", "n": i})
        q = b.subscribe("j", after=0)
        return [q.get_nowait()["id"] for _ in range(q.qsize())]

    assert asyncio.run(go()) == [8, 9, 10]


def test_broker_global_subscriber_sees_all_jobs():
    async def go():
        b = Broker()
        g = b.subscribe_all()
        b.publish("a", {"type": "x"})
        b.publish("b", {"type": "y"})
        got = [g.get_nowait()["job_id"] for _ in range(g.qsize())]
        b.unsubscribe_all(g)
        b.publish("a", {"type": "z"})
        return got, g.qsize()

    got, after = asyncio.run(go())
    assert got == ["a", "b"] and after == 0


# --- submission: profile snapshot, env gate, auto-export -------------------------


def test_submit_snapshots_profile_and_project(tmp_path, make_store, monkeypatch):
    from vorch import config, projects

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(config, "EXPORT_ROOT", tmp_path / "export")

    async def go():
        store = make_store()
        proj = projects.save(store, projects.new_project("Snap", default_profile_id="social_9x16"))
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        job = await mgr.submit("idea", project_id=proj.id, face_swap=False,
                               profile_overrides={"video": {"length": 81}})
        await asyncio.gather(*list(mgr._tasks))
        return store.load_job(job.id)

    job = asyncio.run(go())
    assert job.profile_id == "social_9x16" and job.profile["video"]["length"] == 81
    assert job.profile["face_swap"]["enabled"] is False and job.kind == "video"
    assert job.project_id and job.status is JobStatus.DONE
    assert job.export and job.export["ok"] is True and job.export["files"] == []  # nothing to deliver, recorded


def test_env_gate_blocks_on_fail_and_force_bypasses(tmp_path, make_store):
    from vorch.envcheck import CheckResult, EnvReport, EnvironmentNotReady

    report = EnvReport([CheckResult("comfy", "fail", "down")], "now", 1)
    seen = []

    def gate(profile, project):
        seen.append(profile.id)
        raise EnvironmentNotReady(report)

    async def go():
        store = make_store()
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j),
                         env_check=gate, auto_export=False)
        with pytest.raises(EnvironmentNotReady) as ei:
            await mgr.submit("idea", face_swap=False)
        assert ei.value.report is report and seen == ["cinematic_16x9"]
        job = await mgr.submit("idea", face_swap=False, force=True)  # bypass
        await asyncio.gather(*list(mgr._tasks))
        return store.load_job(job.id).status

    assert asyncio.run(go()) is JobStatus.DONE


def test_auto_export_on_done_emits_event(tmp_path, make_store, monkeypatch):
    from vorch import export as export_mod

    rec = export_mod.ExportRecord(path=str(tmp_path / "exp"), exported_at="t", files=["s01_clip.mp4"])
    monkeypatch.setattr(export_mod, "export_job", lambda store, job, project: rec)

    async def go():
        store = make_store()
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        job = await mgr.submit("idea", face_swap=False)
        q = mgr.events(job.id, after=0)
        events = await _drain(q)
        return [e["type"] for e in events], store.load_job(job.id).export

    types, saved = asyncio.run(go())
    assert types[-2:] == ["export", "end"] and saved["files"] == ["s01_clip.mp4"]


def test_export_now_requires_done_unless_forced(tmp_path, make_store, monkeypatch):
    from vorch import export as export_mod
    from vorch.state import new_job

    monkeypatch.setattr(export_mod, "export_job", lambda store, job, project: export_mod.ExportRecord("p", "t"))
    store = make_store()
    mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
    job = store.save_job(new_job(idea="x"))
    with pytest.raises(RuntimeError):
        mgr.export_now(job.id)
    assert mgr.export_now(job.id, force=True).path == "p"
    assert store.load_job(job.id).export["path"] == "p"
    with pytest.raises(KeyError):
        mgr.export_now("nope")


def test_segment_is_marked_running_before_the_worker_starts(tmp_path, make_store):
    """The request returns (and persists) RUNNING immediately, so a client refetch
    right after approve/regenerate sees an active job and keeps its SSE stream."""
    from vorch.state import Scene, new_job

    class SlowOrch(FakeOrch):
        def approve(self, job, gate):
            import time
            time.sleep(0.2)
            super().approve(job, gate)

    async def go():
        store = make_store()
        job = new_job(idea="x")
        job.status = JobStatus.NEEDS_APPROVAL
        job.scenes = [Scene(index=0)]
        store.save_job(job)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: SlowOrch(s, j), auto_export=False)
        q = mgr.events(job.id)
        returned = await mgr.approve(job.id, "storyboard")
        ret_status = returned.status  # read now: the worker mutates the same object later
        persisted = store.load_job(job.id).status
        first = await asyncio.wait_for(q.get(), 1)
        await asyncio.gather(*list(mgr._tasks))
        return ret_status, persisted, first["type"], first.get("status"), store.load_job(job.id).status

    ret, persisted, ev_type, ev_status, final = asyncio.run(go())
    assert ret is JobStatus.RUNNING and persisted is JobStatus.RUNNING
    assert (ev_type, ev_status) == ("status", "running")
    assert final is JobStatus.DONE


# --- assembly task --------------------------------------------------------------------


def _done_job_with_clips(store, tmp_path, n=2):
    from vorch.state import Scene, new_job

    job = new_job(idea="asm", kind="video")
    job.status = JobStatus.DONE
    scenes = []
    for i in range(n):
        p = tmp_path / f"c{i}.mp4"
        p.write_bytes(b"v")
        scenes.append(Scene(index=i, keyframe_path=f"/k{i}.png", video_path=str(p)))
    job.scenes = scenes
    store.save_job(job)
    return job


@pytest.fixture
def fake_assembly(monkeypatch):
    from vorch import assembly as asm
    from vorch.media import Cut

    def resolve(job, spec):
        s = asm.AssemblySpec(items=[asm.AssemblyItem(x.index) for x in job.scenes], cuts=[Cut() for _ in job.scenes[1:]])
        return s, [x.video_path for x in job.scenes]

    calls = {"run": 0}

    def run(job, spec, paths, *, emit=None):
        calls["run"] += 1
        if emit:
            emit({"job_id": job.id, "type": "step", "step": "assemble", "n_clips": len(paths), "seconds": 5.6})
        ok = calls.get("fail") is None
        return asm.AssemblyRecord(path="/out/assembly_v1.mp4", created_at="t", duration=5.6, n_clips=len(paths),
                                  spec=spec.to_dict(), ok=ok, error="" if ok else "ffmpeg died", version=1)

    monkeypatch.setattr(asm, "resolve", resolve)
    monkeypatch.setattr(asm, "run", run)
    return calls


def test_assemble_task_emits_step_assembly_end_and_records(tmp_path, make_store, fake_assembly, monkeypatch):
    from vorch import export as export_mod

    monkeypatch.setattr(export_mod, "export_job", lambda store, job, project: export_mod.ExportRecord("/exp", "t", files=["assembly.mp4"]))

    async def go():
        store = make_store()
        job = _done_job_with_clips(store, tmp_path)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        q = mgr.events(job.id)
        await mgr.assemble(job.id, {"cuts": [{"transition": "fade", "duration": 0.5}]})
        assert mgr.is_active(job.id) and store.load_job(job.id).status is JobStatus.DONE  # status untouched
        events = await _drain(q)
        return [e["type"] for e in events], store.load_job(job.id)

    types, job = asyncio.run(go())
    assert types == ["step", "assembly", "export", "end"]
    assert job.assembly["ok"] and job.assembly["duration"] == 5.6 and job.status is JobStatus.DONE
    assert [v["version"] for v in job.assembly["versions"]] == [1]
    assert job.export["files"] == ["assembly.mp4"]


def test_assemble_refused_while_active_and_when_not_done(tmp_path, make_store, fake_assembly):
    async def go():
        store = make_store()
        job = _done_job_with_clips(store, tmp_path)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j), auto_export=False)
        mgr._active.add(job.id)
        with pytest.raises(RuntimeError, match="running"):
            await mgr.assemble(job.id)
        mgr._active.discard(job.id)
        job.status = JobStatus.NEEDS_APPROVAL
        store.save_job(job)
        with pytest.raises(RuntimeError, match="not done"):
            await mgr.assemble(job.id)
        await mgr.assemble(job.id, force=True)  # allowed
        await asyncio.gather(*list(mgr._tasks))
        with pytest.raises(KeyError):
            await mgr.assemble("nope")
        return store.load_job(job.id)

    job = asyncio.run(go())
    assert job.assembly["ok"] and job.status is JobStatus.NEEDS_APPROVAL


def test_assemble_failure_recorded_without_failed_status(tmp_path, make_store, fake_assembly):
    fake_assembly["fail"] = True

    async def go():
        store = make_store()
        job = _done_job_with_clips(store, tmp_path)
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j), auto_export=False)
        q = mgr.events(job.id)
        await mgr.assemble(job.id)
        events = await _drain(q)
        return events, store.load_job(job.id), mgr.is_active(job.id)

    events, job, active = asyncio.run(go())
    assert [e["type"] for e in events] == ["step", "assembly", "end"]
    assert events[1]["ok"] is False and "ffmpeg died" in events[1]["error"]
    assert job.status is JobStatus.DONE and job.assembly["ok"] is False and not active
