"""GPU-free checks for server.py — the HTTP API routes.

A fake Orchestrator factory keeps the JobManager from doing real generation, so
the routes are exercised against TestClient without Gemma/ComfyUI. The async
streaming/queue behaviour itself is covered by test_queue.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vorch.queue import JobManager
from vorch.server import create_app
from vorch.state import JobStatus, Store


class FakeOrch:
    def __init__(self, store, job):
        self.store = store
        self.emit = lambda event: None

    def run(self, job):
        job.status = JobStatus.DONE
        self.store.save_job(job)

    def approve(self, job, gate):
        job.status = JobStatus.DONE
        self.store.save_job(job)


@pytest.fixture
def client(tmp_path):
    store = Store(str(tmp_path / "srv.db"))
    manager = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
    app = create_app(store=store, manager=manager)
    master = tmp_path / "m.jpg"
    master.write_bytes(b"\xff\xd8\xff")
    with TestClient(app) as c:
        c.master = str(master)  # stash for tests
        yield c
    store.close()


def test_create_job_returns_job(client):
    r = client.post("/api/jobs", json={"idea": "a cat", "master_face": client.master, "n_scenes": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] and body["idea"] == "a cat" and body["n_scenes"] == 1


def test_create_job_bad_master_is_400(client):
    r = client.post("/api/jobs", json={"idea": "x", "master_face": "/no/such.jpg"})
    assert r.status_code == 400


def test_create_job_no_master_ok_when_face_swap_off(client):
    r = client.post("/api/jobs", json={"idea": "a landscape", "face_swap": False})
    assert r.status_code == 200
    assert r.json()["face_swap"] is False


def test_create_then_list_and_get(client):
    jid = client.post(
        "/api/jobs", json={"idea": "bread", "master_face": client.master}
    ).json()["id"]

    listed = client.get("/api/jobs").json()
    assert any(j["id"] == jid for j in listed)

    got = client.get(f"/api/jobs/{jid}").json()
    assert got["id"] == jid and "scenes" in got


def test_get_missing_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_approve_missing_job_404(client):
    assert client.post("/api/jobs/nope/approve/storyboard").status_code == 404


def test_decisions_endpoint_returns_list(client):
    jid = client.post(
        "/api/jobs", json={"idea": "x", "master_face": client.master}
    ).json()["id"]
    r = client.get(f"/api/jobs/{jid}/decisions")
    assert r.status_code == 200 and isinstance(r.json(), list)
