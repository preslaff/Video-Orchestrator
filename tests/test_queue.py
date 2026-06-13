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
    assert got == {"a": 1} and empty_after is True


def test_execute_streams_events_and_completes(tmp_path):
    async def go():
        store = Store(str(tmp_path / "q.db"))
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


def test_execute_reports_failure_as_state_and_event(tmp_path):
    async def go():
        store = Store(str(tmp_path / "q.db"))
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


def test_submit_creates_validates_and_runs(tmp_path):
    master = tmp_path / "m.jpg"
    master.write_bytes(b"\xff\xd8\xff")

    async def go():
        store = Store(str(tmp_path / "q.db"))
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        job = await mgr.submit("idea", master_face=str(master), autonomy="full", n_scenes=1)
        await asyncio.gather(*list(mgr._tasks))  # let the background run finish
        final = store.load_job(job.id).status
        store.close()
        return job, final

    job, final = asyncio.run(go())
    assert job.master_face.endswith("m.jpg") and job.n_scenes == 1
    assert final is JobStatus.DONE


def test_submit_rejects_missing_master(tmp_path):
    async def go():
        store = Store(str(tmp_path / "q.db"))
        mgr = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        try:
            with pytest.raises(FileNotFoundError):
                await mgr.submit("idea", master_face="/no/such/face.jpg")
        finally:
            store.close()

    asyncio.run(go())


def test_duplicate_approve_ignored_while_active(tmp_path):
    """A second approve while a run segment is in flight is a no-op."""
    calls = {"n": 0}

    class CountingOrch(FakeOrch):
        def approve(self, job, gate):
            calls["n"] += 1

    async def go():
        store = Store(str(tmp_path / "q.db"))
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


def test_approve_loads_and_runs(tmp_path):
    async def go():
        store = Store(str(tmp_path / "q.db"))
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
