"""GPU-free checks for server.py — the HTTP API routes.

A fake Orchestrator factory keeps the JobManager from doing real generation, so
the routes are exercised against TestClient without Gemma/ComfyUI. The async
streaming/queue behaviour itself is covered by test_queue.py.
"""

from __future__ import annotations

import asyncio

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
def client(tmp_path, make_store, monkeypatch):
    from vorch import config

    store = make_store()
    manager = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(config, "COMFY_OUTPUT", media)
    app = create_app(store=store, manager=manager, media_dir=media)
    master = tmp_path / "m.jpg"
    master.write_bytes(b"\xff\xd8\xff")
    with TestClient(app) as c:
        c.master = str(master)  # stash for tests
        c.store = store
        c.media = media
        yield c


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
    assert any(j["id"] == jid for j in listed["items"]) and listed["total"] >= 1

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


def _paused_job_with_scene(store):
    from vorch.state import Scene, new_job

    job = new_job(idea="edit me")
    job.status = JobStatus.NEEDS_APPROVAL
    job.scenes = [Scene(index=0, keyframe_prompt="old kf", animation_prompt="old an")]
    store.save_job(job)
    return job


def test_update_scenes_applies_edits(client):
    job = _paused_job_with_scene(client.store)
    r = client.put(
        f"/api/jobs/{job.id}/scenes",
        json={"scenes": [{"index": 0, "keyframe_prompt": "new kf", "description": "d"}]},
    )
    assert r.status_code == 200
    scene = r.json()["scenes"][0]
    assert scene["keyframe_prompt"] == "new kf"
    assert scene["description"] == "d"
    assert scene["animation_prompt"] == "old an"  # untouched


def test_update_scenes_missing_job_404(client):
    r = client.put("/api/jobs/nope/scenes", json={"scenes": [{"index": 0, "description": "x"}]})
    assert r.status_code == 404


def test_update_scenes_while_running_409(client):
    job = _paused_job_with_scene(client.store)
    client.app.state.manager._active.add(job.id)  # simulate a run segment in flight
    r = client.put(f"/api/jobs/{job.id}/scenes", json={"scenes": [{"index": 0, "description": "x"}]})
    assert r.status_code == 409


# --- commit 2: envelope, gate validation, public projection -------------------


def test_error_envelope_shape_on_404(client):
    r = client.get("/api/jobs/nope")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "not_found" and body["error"]["message"]


def test_approve_invalid_gate_422(client):
    job = _paused_job_with_scene(client.store)
    r = client.post(f"/api/jobs/{job.id}/approve/bogus")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_approve_not_paused_409(client):
    jid = client.post("/api/jobs", json={"idea": "x", "face_swap": False}).json()["id"]
    # FakeOrch finishes synchronously-ish; wait for the run to settle.
    import time
    for _ in range(50):
        if client.get(f"/api/jobs/{jid}").json()["status"] == "done":
            break
        time.sleep(0.02)
    r = client.post(f"/api/jobs/{jid}/approve/storyboard")
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"


def test_job_detail_reports_pending_gate(client):
    from vorch.state import Decision

    job = _paused_job_with_scene(client.store)
    assert client.get(f"/api/jobs/{job.id}").json()["pending_gate"] == "storyboard"
    client.store.log_decision(Decision(job_id=job.id, scene_index=None, phase="storyboard",
                                       attempt=1, action="approved"))
    assert client.get(f"/api/jobs/{job.id}").json()["pending_gate"] == "keyframes"
    job.status = JobStatus.DONE
    client.store.save_job(job)
    assert client.get(f"/api/jobs/{job.id}").json()["pending_gate"] is None


def test_job_public_has_media_urls(client, tmp_path, monkeypatch):
    from vorch import config
    from vorch.state import Scene, new_job

    out = tmp_path / "comfy_out"
    (out / "video").mkdir(parents=True)
    monkeypatch.setattr(config, "COMFY_OUTPUT", out)
    kf = out / "Flux2-Klein_00001_.png"
    kf.write_bytes(b"\x89PNG")
    vid = out / "video" / "clip.mp4"
    vid.write_bytes(b"\x00")

    job = new_job(idea="m", master_face=client.master)
    job.scenes = [Scene(index=0, keyframe_path=str(kf), video_path=str(vid))]
    client.store.save_job(job)

    s = client.get(f"/api/jobs/{job.id}").json()["scenes"][0]
    assert s["keyframe_url"] == "/media/Flux2-Klein_00001_.png"
    assert s["video_url"] == "/media/video/clip.mp4"
    assert s["swapped_url"] is None
    # local paths are still present for the export step
    assert s["keyframe_path"] == str(kf)


def test_delete_job_204_409_404(client):
    job = _paused_job_with_scene(client.store)
    client.app.state.manager._active.add(job.id)
    assert client.delete(f"/api/jobs/{job.id}").status_code == 409
    client.app.state.manager._active.discard(job.id)
    assert client.delete(f"/api/jobs/{job.id}").status_code == 204
    assert client.get(f"/api/jobs/{job.id}").status_code == 404
    assert client.delete(f"/api/jobs/{job.id}").status_code == 404


def test_decisions_missing_job_404(client):
    assert client.get("/api/jobs/nope/decisions").status_code == 404


def test_provenance_endpoint(client, tmp_path, monkeypatch):
    from vorch import config, provenance
    from vorch.state import Scene, new_job

    out = tmp_path / "o"
    out.mkdir()
    monkeypatch.setattr(config, "COMFY_OUTPUT", out)
    kf = out / "k.png"
    kf.write_bytes(b"\x89PNG")
    provenance.write_manifest(kf, "keyframe", {"positive": "a cat", "seed": 42})
    sw = out / "s.png"
    sw.write_bytes(b"\x89PNG")  # no manifest

    job = new_job(idea="p")
    job.scenes = [Scene(index=0, keyframe_path=str(kf), swapped_path=str(sw))]
    client.store.save_job(job)

    rows = client.get(f"/api/jobs/{job.id}/provenance").json()
    assert [(r["kind"], r["has_manifest"]) for r in rows] == [("keyframe", True), ("swapped", False)]
    assert rows[0]["seed"] == 42 and rows[0]["prompt"] == "a cat" and rows[0]["url"] == "/media/k.png"


# --- commit 3: /media static mount --------------------------------------------


def test_media_serves_file_with_mime(client):
    (client.media / "k.png").write_bytes(b"\x89PNG\r\n")
    r = client.get("/media/k.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    (client.media / "video").mkdir()
    (client.media / "video" / "c.mp4").write_bytes(b"\x00" * 16)
    r = client.get("/media/video/c.mp4")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"


def test_media_traversal_rejected(client, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    for path in ("/media/../secret.txt", "/media/%2e%2e/secret.txt", "/media/..%2fsecret.txt"):
        r = client.get(path)
        assert r.status_code == 404, path
        assert "nope" not in r.text


def test_media_range_request_206(client):
    (client.media / "c.mp4").write_bytes(bytes(range(64)))
    r = client.get("/media/c.mp4", headers={"Range": "bytes=8-15"})
    assert r.status_code == 206
    assert r.content == bytes(range(8, 16))
    assert r.headers["content-range"] == "bytes 8-15/64"
    assert client.get("/media/c.mp4").headers["accept-ranges"] == "bytes"


def test_media_missing_is_404(client):
    assert client.get("/media/does-not-exist.png").status_code == 404


# --- commit 4: SSE named events, ids, replay ---------------------------------


def _sse_events(text: str) -> list[dict]:
    """Parse an SSE body into [{event, id, data}] (ignores comments/pings)."""
    out, cur = [], {}
    for line in text.splitlines():
        if not line.strip():
            if cur:
                out.append(cur)
                cur = {}
            continue
        if line.startswith(":"):
            continue
        key, _, val = line.partition(":")
        cur[key] = val.strip()
    if cur:
        out.append(cur)
    return out


def test_sse_named_events_and_ids(client):
    jid = client.post("/api/jobs", json={"idea": "x", "face_swap": False}).json()["id"]
    # FakeOrch already ran; the ring buffer has status + end, replayed from after=0.
    with client.stream("GET", f"/api/jobs/{jid}/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    evs = _sse_events(body)
    assert [e["event"] for e in evs][-1] == "end"
    assert [int(e["id"]) for e in evs] == list(range(1, len(evs) + 1))
    assert all("job_id" in e["data"] for e in evs)


def test_sse_replays_after_last_event_id(client):
    from vorch.state import new_job

    job = new_job(idea="r")
    client.store.save_job(job)
    broker = client.app.state.manager.broker
    broker.publish(job.id, {"type": "status", "status": "running"})
    broker.publish(job.id, {"type": "keyframe", "index": 0})
    broker.publish(job.id, {"type": "end"})
    with client.stream("GET", f"/api/jobs/{job.id}/events", headers={"Last-Event-ID": "1"}) as r:
        body = "".join(r.iter_text())
    assert [(e["event"], e["id"]) for e in _sse_events(body)] == [("keyframe", "2"), ("end", "3")]
    with client.stream("GET", f"/api/jobs/{job.id}/events?after=2") as r:
        body = "".join(r.iter_text())
    assert [e["event"] for e in _sse_events(body)] == ["end"]


def test_sse_terminal_job_without_history_ends(client):
    job = _paused_job_with_scene(client.store)
    job.status = JobStatus.DONE
    client.store.save_job(job)
    with client.stream("GET", f"/api/jobs/{job.id}/events") as r:
        body = "".join(r.iter_text())
    evs = _sse_events(body)
    assert len(evs) == 1 and evs[0]["event"] == "end"


def test_sse_missing_job_404_and_bad_after_400(client):
    assert client.get("/api/jobs/nope/events").status_code == 404
    job = _paused_job_with_scene(client.store)
    assert client.get(f"/api/jobs/{job.id}/events?after=x").status_code == 400


def test_global_events_stream_carries_job_id(store):
    """/api/events is endless, so drive the route's generator directly instead of
    through TestClient (which cannot close an open stream)."""
    import json as _json

    from vorch.api.events import all_events

    async def go():
        manager = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        resp = await all_events(manager=manager)
        it = resp.body_iterator.__aiter__()
        manager.broker.publish("job-1", {"type": "status", "status": "running"})
        ev = await asyncio.wait_for(it.__anext__(), 2)
        await it.aclose()
        return ev, len(manager.broker._all)

    ev, subs_left = asyncio.run(go())
    assert ev.event == "status" and _json.loads(ev.data)["job_id"] == "job-1"
    assert subs_left == 0  # unsubscribed on close


# --- commit 5: pagination + filters -------------------------------------------


def test_list_jobs_pagination_and_total(client):
    from vorch.state import new_job

    for i in range(5):
        client.store.save_job(new_job(idea=f"j{i}"))
    page = client.get("/api/jobs?limit=2&offset=0").json()
    assert page["total"] == 5 and page["limit"] == 2 and page["offset"] == 0
    assert len(page["items"]) == 2 and page["items"][0]["idea"] == "j4"  # newest first
    page2 = client.get("/api/jobs?limit=2&offset=4").json()
    assert [j["idea"] for j in page2["items"]] == ["j0"]
    assert client.get("/api/jobs?limit=0").status_code == 422
    assert client.get("/api/jobs?limit=500").status_code == 422


def test_list_jobs_status_filter(client):
    from vorch.state import new_job

    a = new_job(idea="paused")
    a.status = JobStatus.NEEDS_APPROVAL
    client.store.save_job(a)
    client.store.save_job(new_job(idea="fresh"))
    page = client.get("/api/jobs?status=needs_approval").json()
    assert page["total"] == 1 and page["items"][0]["idea"] == "paused"
    assert client.get("/api/jobs?status=bogus").status_code == 422
