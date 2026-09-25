"""GPU-free checks for server.py — the HTTP API routes.

A fake Orchestrator factory keeps the JobManager from doing real generation, so
the routes are exercised against TestClient without Gemma/ComfyUI. The async
streaming/queue behaviour itself is covered by test_queue.py.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vorch.queue import JobManager
from vorch.api import create_app
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
    manager = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j), auto_export=False)
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
    assert evs[0]["event"] == "open" and "id" not in evs[0]   # the watermark, not a buffered event
    rest = evs[1:]
    assert [e["event"] for e in rest][-1] == "end"
    assert [int(e["id"]) for e in rest] == list(range(1, len(rest) + 1))
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
    assert [(e["event"], e.get("id")) for e in _sse_events(body)] == [
        ("open", None), ("keyframe", "2"), ("end", "3")]
    with client.stream("GET", f"/api/jobs/{job.id}/events?after=2") as r:
        body = "".join(r.iter_text())
    assert [e["event"] for e in _sse_events(body)] == ["open", "end"]


def test_sse_terminal_job_without_history_ends(client):
    job = _paused_job_with_scene(client.store)
    job.status = JobStatus.DONE
    client.store.save_job(job)
    with client.stream("GET", f"/api/jobs/{job.id}/events") as r:
        body = "".join(r.iter_text())
    evs = _sse_events(body)
    assert [e["event"] for e in evs] == ["open", "end"]
    import json as _json

    assert _json.loads(evs[1]["data"])["final"] is True


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


# --- commit 6: CORS, root hint, OpenAPI snapshot ------------------------------


def test_cors_preflight_from_next_origin(client):
    r = client.options(
        "/api/jobs",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    # An unknown origin gets no CORS grant.
    r = client.get("/api/jobs", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_root_hint(client):
    body = client.get("/").json()
    assert body["api"] == "/api" and body["ui"].startswith("http")


def test_openapi_snapshot_matches_frontend():
    import json
    from pathlib import Path

    from vorch.api import openapi_schema

    snap = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"
    assert snap.exists(), "frontend/openapi.json missing — run `make api-types`"
    assert openapi_schema() == json.loads(snap.read_text()), (
        "API schema changed — run `make api-types` and commit frontend/openapi.json"
    )


def test_list_jobs_project_filter_and_summary_fields(client):
    from vorch.state import new_job

    client.store.save_job(new_job(idea="s", kind="stills", profile_id="stills_16x9"))
    page = client.get("/api/jobs").json()
    assert page["items"][0]["kind"] == "stills" and page["items"][0]["profile_id"] == "stills_16x9"
    assert client.get("/api/jobs?project_id=nope").json()["total"] == 0


def test_create_job_env_fail_409_carries_report(tmp_path, make_store):
    from vorch.envcheck import CheckResult, EnvReport, EnvironmentNotReady

    store = make_store()
    report = EnvReport([CheckResult("gemma", "fail", "unreachable")], "now", 2)

    def gate(profile, project):
        raise EnvironmentNotReady(report)

    manager = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j), env_check=gate, auto_export=False)
    app = create_app(store=store, manager=manager, media_dir=tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/jobs", json={"idea": "x", "face_swap": False})
        assert r.status_code == 409
        body = r.json()["error"]
        assert body["code"] == "environment_not_ready" and body["details"]["checks"][0]["name"] == "gemma"
        assert c.post("/api/jobs", json={"idea": "x", "face_swap": False, "force": True}).status_code == 200
        assert c.post("/api/jobs", json={"idea": "x", "profile_id": "nope", "force": True}).status_code == 400


def test_regenerate_scene_route(client):
    from vorch.state import Scene, new_job

    job = new_job(idea="r")
    job.status = JobStatus.DONE
    job.scenes = [Scene(index=0, keyframe_path="/k.png", video_path="/v.mp4")]
    client.store.save_job(job)
    r = client.post(f"/api/jobs/{job.id}/scenes/0/regenerate", json={"what": "keyframe", "keyframe_prompt": "new"})
    assert r.status_code == 200
    assert client.post(f"/api/jobs/{job.id}/scenes/7/regenerate", json={}).status_code == 404
    assert client.post("/api/jobs/nope/scenes/0/regenerate", json={}).status_code == 404
    assert client.post(f"/api/jobs/{job.id}/scenes/0/regenerate", json={"what": "audio"}).status_code == 422


def test_scene_locks_route(client, monkeypatch):
    """Marking a label restores it on the clip that already exists — a paste, not a render."""
    from vorch import media
    from vorch.state import Scene, new_job

    monkeypatch.setattr(media, "lock_regions", lambda vid, src, regions, dest, **kw: Path(dest))
    job = new_job(idea="r")
    job.status = JobStatus.DONE
    job.scenes = [Scene(index=0, keyframe_path="/k.png", video_path="/v.mp4", raw_video_path="/v.mp4")]
    client.store.save_job(job)

    label = {"x": 0.41, "y": 0.35, "w": 0.2, "h": 0.1}
    r = client.put(f"/api/jobs/{job.id}/scenes/0/locks", json={"regions": [label]})
    assert r.status_code == 200
    scene = r.json()["scenes"][0]
    assert scene["locked_regions"] == [label]
    assert scene["video_path"] == "/v_locked.mp4"

    assert client.put(f"/api/jobs/{job.id}/scenes/0/locks", json={"regions": []}).status_code == 200
    assert client.put(f"/api/jobs/{job.id}/scenes/7/locks", json={"regions": []}).status_code == 404
    assert client.put("/api/jobs/nope/scenes/0/locks", json={"regions": []}).status_code == 404
    off = {"x": 0.9, "y": 0.1, "w": 0.3, "h": 0.1}     # a rectangle that leaves the frame
    assert client.put(f"/api/jobs/{job.id}/scenes/0/locks", json={"regions": [off]}).status_code == 400


def test_retry_resumes_a_failed_job(client):
    """A failed job (e.g. the API server lacked a workflow) is resumed from its state."""
    import time

    from vorch.state import Scene, new_job

    job = new_job(idea="r")
    job.status = JobStatus.FAILED
    job.scenes = [Scene(index=0, keyframe_path="/k.png")]
    client.store.save_job(job)
    r = client.post(f"/api/jobs/{job.id}/retry")
    assert r.status_code == 200 and r.json()["status"] in ("running", "done")
    for _ in range(50):
        if client.get(f"/api/jobs/{job.id}").json()["status"] == "done":
            break
        time.sleep(0.02)
    assert client.get(f"/api/jobs/{job.id}").json()["status"] == "done"
    actions = [d["action"] for d in client.get(f"/api/jobs/{job.id}/decisions").json()]
    assert "retry" in actions
    r = client.post(f"/api/jobs/{job.id}/retry")                       # not failed any more
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    assert client.post("/api/jobs/nope/retry").status_code == 404


def test_stop_route_and_live_run_flags(client):
    """POST /stop is 202 on a stoppable segment, 409 otherwise; the job carries
    `active`/`stopping` so a reload still knows a stop is pending."""
    import threading

    from vorch.state import new_job

    job = new_job(idea="s")
    client.store.save_job(job)
    got = client.get(f"/api/jobs/{job.id}").json()
    assert got["active"] is False and got["stopping"] is False

    assert client.post(f"/api/jobs/{job.id}/stop").status_code == 409  # nothing running
    assert client.post("/api/jobs/nope/stop").status_code == 404

    mgr = client.app.state.manager
    mgr._active.add(job.id)
    mgr._stop[job.id] = threading.Event()  # a segment in flight
    try:
        r = client.post(f"/api/jobs/{job.id}/stop")
        assert r.status_code == 202 and r.json()["active"] is True and r.json()["stopping"] is True
        assert client.get(f"/api/jobs/{job.id}").json()["stopping"] is True
    finally:
        mgr._active.discard(job.id)
        mgr._stop.pop(job.id, None)


def test_job_listing_carries_the_profile_name(client):
    from vorch import profiles
    from vorch.state import new_job

    prof = profiles.clone(profiles.default_profile(), "Cloud social 9:16 test")
    job = new_job(idea="named", profile_id=prof.id, profile=prof.snapshot())
    client.store.save_job(job)
    rows = {r["id"]: r for r in client.get("/api/jobs").json()["items"]}
    assert rows[job.id]["profile_name"] == "Cloud social 9:16 test" and rows[job.id]["profile_id"] == prof.id
    assert client.get(f"/api/jobs/{job.id}").json()["profile"]["name"] == "Cloud social 9:16 test"


def test_unassigned_jobs_filter_and_project_last_job(client):
    from vorch import projects as projects_mod
    from vorch.state import new_job

    proj = projects_mod.save(client.store, projects_mod.new_project(name="Filter Co"))
    a = new_job(idea="in project", project_id=proj.id)
    b = new_job(idea="no project")
    client.store.save_job(a)
    client.store.save_job(b)
    un = client.get("/api/jobs?unassigned=1").json()
    assert [j["idea"] for j in un["items"] if j["id"] in (a.id, b.id)] == ["no project"] and un["total"] >= 1
    assert all(j["project_id"] is None for j in un["items"])
    r = client.get(f"/api/jobs?unassigned=1&project_id={proj.id}")
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    rows = {p["id"]: p for p in client.get("/api/projects").json()}
    assert rows[proj.id]["n_jobs"] == 1 and rows[proj.id]["last_job_at"] == client.get(f"/api/jobs/{a.id}").json()["updated_at"]
    empty = projects_mod.save(client.store, projects_mod.new_project(name="Empty Co"))
    assert {p["id"]: p for p in client.get("/api/projects").json()}[empty.id]["last_job_at"] is None


def _project_with_faces(store):
    from vorch import projects as projects_mod

    p = projects_mod.save(store, projects_mod.new_project(name="Faces Co"))
    young = projects_mod.add_asset(p, b"y", kind="master_face", name="young.png", label="Maria, age 8")
    old = projects_mod.add_asset(p, b"o", kind="master_face", name="old.png", label="Maria, age 70")
    ref = projects_mod.add_asset(p, b"r", kind="reference", name="room.png")
    projects_mod.save(store, p)
    return p, young, old, ref


def test_asset_label_patch(client):
    p, young, _old, _ref = _project_with_faces(client.store)
    r = client.patch(f"/api/projects/{p.id}/assets/{young.id}", json={"label": "Maria, age 9"})
    assert r.status_code == 200 and r.json()["label"] == "Maria, age 9"
    assert client.patch(f"/api/projects/{p.id}/assets/nope", json={"label": "x"}).status_code == 404
    got = client.get(f"/api/projects/{p.id}").json()
    assert {a["id"]: a["label"] for a in got["assets"]}[young.id] == "Maria, age 9"


def test_asset_out_carries_the_suggestion_and_patching_confirms_it(client):
    """The review loop: Gemma suggests ("auto"), the UI flags it, saving makes it "human"."""
    from vorch import projects as projects_mod

    p, young, _old, _ref = _project_with_faces(client.store)
    projects_mod.set_asset_note(p, young.id, {"caption": "a girl with dark hair"},
                                label="girl, dark hair")
    projects_mod.save(client.store, p)

    assets = {a["id"]: a for a in client.get(f"/api/projects/{p.id}").json()["assets"]}
    assert assets[young.id]["label_source"] == "auto"
    assert assets[young.id]["note_caption"] == "a girl with dark hair"
    # An asset nobody has described carries empty strings, not nulls.
    assert assets[_ref.id]["label_source"] == "" and assets[_ref.id]["note_caption"] == ""

    r = client.patch(f"/api/projects/{p.id}/assets/{young.id}", json={"label": "Maria, age 8"})
    assert r.status_code == 200 and r.json()["label_source"] == "human"
    # ...and the caption survives, so the reviewer keeps their context.
    assert r.json()["note_caption"] == "a girl with dark hair"


def test_update_scenes_assigns_identity_and_references_by_asset_id(client):
    from vorch.state import Scene, new_job

    p, young, old, ref = _project_with_faces(client.store)
    job = new_job(idea="ages", project_id=p.id)
    job.status = JobStatus.NEEDS_APPROVAL
    job.scenes = [Scene(index=0), Scene(index=1)]
    client.store.save_job(job)
    r = client.put(f"/api/jobs/{job.id}/scenes", json={"scenes": [
        {"index": 0, "identity_asset_id": young.id, "reference_asset_ids": [ref.id], "mode": "edit"},
        {"index": 1, "identity_asset_id": old.id},
    ]})
    assert r.status_code == 200
    s0, s1 = r.json()["scenes"]
    assert s0["identity"] == young.path and s0["identity_asset_id"] == young.id
    assert s0["identity_url"] == f"/api/projects/{p.id}/assets/{young.id}/file"
    assert s0["references"] == [ref.path] and s0["reference_asset_ids"] == [ref.id]
    assert s1["identity_asset_id"] == old.id
    # clear the identity -> no swap for that scene
    r = client.put(f"/api/jobs/{job.id}/scenes", json={"scenes": [{"index": 1, "identity_asset_id": ""}]})
    assert r.json()["scenes"][1]["identity"] == "" and r.json()["scenes"][1]["identity_asset_id"] is None
    # unknown asset -> 400; asset ids without a project -> 400
    r = client.put(f"/api/jobs/{job.id}/scenes", json={"scenes": [{"index": 0, "identity_asset_id": "ghost"}]})
    assert r.status_code == 400 and "ghost" in r.json()["error"]["message"]
    orphan = new_job(idea="no project")
    orphan.status = JobStatus.NEEDS_APPROVAL
    orphan.scenes = [Scene(index=0)]
    client.store.save_job(orphan)
    r = client.put(f"/api/jobs/{orphan.id}/scenes", json={"scenes": [{"index": 0, "reference_asset_ids": [ref.id]}]})
    assert r.status_code == 400 and "no project" in r.json()["error"]["message"]


# --- assembly routes -------------------------------------------------------------


def _done_video_job(client, n=2, **extra):
    from vorch.state import Scene, new_job

    extra.setdefault("profile", {"video": {"width": 832, "height": 480, "fps": 16, "length": 49}})
    job = new_job(idea="asm", kind="video", **extra)
    job.status = JobStatus.DONE
    job.scenes = []
    for i in range(n):
        p = client.media / f"c{i}.mp4"
        p.write_bytes(b"v")
        job.scenes.append(Scene(index=i, keyframe_path=f"/k{i}.png", video_path=str(p)))
    client.store.save_job(job)
    return job


@pytest.fixture
def stub_media(monkeypatch):
    from vorch import media
    from vorch.media import MediaInfo

    monkeypatch.setattr(media, "transitions", lambda: ("none", "fade", "wipeleft"))
    monkeypatch.setattr(media, "probe", lambda p: MediaInfo(832, 480, 16.0, 3.0625, 49, "h264", False))


def test_assembly_get_default_spec_and_transitions(client, stub_media):
    job = _done_video_job(client)
    st = client.get(f"/api/jobs/{job.id}/assemble").json()
    assert st["available"] and st["active"] is False and st["transitions"][0] == "none" and "fade" in st["transitions"]
    assert [i["scene_index"] for i in st["default_spec"]["items"]] == [0, 1]
    assert st["default_spec"]["cuts"] == [{"transition": "fade", "duration": 0.5}]
    assert st["clips"][0]["url"] == "/media/c0.mp4" and st["clips"][0]["duration"] == 3.062
    assert st["profile_fps"] == 16 and st["record"] is None
    assert client.get("/api/jobs/nope/assemble").status_code == 404


def test_assembly_post_400_on_bad_spec(client, stub_media):
    job = _done_video_job(client)
    r = client.post(f"/api/jobs/{job.id}/assemble", json={"cuts": [{"transition": "swoosh"}]})
    assert r.status_code == 400 and "unknown transition" in r.json()["error"]["message"]
    r = client.post(f"/api/jobs/{job.id}/assemble", json={"items": [{"scene_index": 7}]})
    assert r.status_code == 400 and "no scene 8" in r.json()["error"]["message"]
    assert client.post(f"/api/jobs/{job.id}/assemble", json={"encoder": "magic"}).status_code == 422


def test_assembly_post_409_not_done_and_while_active(client, stub_media):
    job = _done_video_job(client)
    job.status = JobStatus.NEEDS_APPROVAL
    client.store.save_job(job)
    assert client.post(f"/api/jobs/{job.id}/assemble").status_code == 409
    client.app.state.manager._active.add(job.id)
    assert client.post(f"/api/jobs/{job.id}/assemble?force=1").status_code == 409
    client.app.state.manager._active.discard(job.id)


def test_assembly_post_200_runs_and_records(client, stub_media, monkeypatch):
    import time

    from vorch import assembly as asm

    def fake_run(job, spec, paths, *, emit=None):
        v = asm.next_version(job)
        out = client.media / "vorch" / job.id[:8] / f"assembly_v{v}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return asm.AssemblyRecord(path=str(out), created_at="t", duration=5.62, n_clips=len(paths), spec=spec.to_dict(),
                                  ok=True, size_bytes=3, version=v)

    monkeypatch.setattr(asm, "run", fake_run)
    job = _done_video_job(client)
    r = client.post(f"/api/jobs/{job.id}/assemble", json={"cuts": [{"transition": "wipeleft", "duration": 0.3}], "fps": 16})
    assert r.status_code == 200 and r.json()["active"] is True
    for _ in range(100):
        st = client.get(f"/api/jobs/{job.id}/assemble").json()
        if st["record"] and not st["active"]:
            break
        time.sleep(0.02)
    assert st["record"]["ok"] and st["record"]["duration"] == 5.62 and st["record"]["exists"] is True
    assert st["record"]["url"].startswith(f"/media/vorch/{job.id[:8]}/assembly_v1.mp4?v=")  # cache-busted per assembly
    assert [v["version"] for v in st["versions"]] == [1] and st["versions"][0]["current"]
    assert st["record"]["spec"]["cuts"][0]["transition"] == "wipeleft"
    assert client.get(f"/api/jobs/{job.id}").json()["assembly"]["ok"] is True


def test_media_is_served_with_no_cache(client):
    (client.media / "k.png").write_bytes(b"\x89PNG")
    r = client.get("/media/k.png")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    r = client.get("/media/k.png", headers={"Range": "bytes=0-1"})
    assert r.status_code == 206 and r.headers["cache-control"] == "no-cache"


def test_assembly_versions_select_and_delete_routes(client, stub_media, tmp_path):
    from vorch import assembly as asm

    job = _done_video_job(client)
    for n in (1, 2):
        f = client.media / "vorch" / job.id[:8] / f"assembly_v{n}.mp4"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(f"v{n}".encode())
        asm.register_version(job, asm.AssemblyRecord(path=str(f), created_at=f"2026-09-0{n}T00:00:00+00:00",
                                                     duration=5.0 + n, n_clips=2, version=n))
    client.store.save_job(job)
    st = client.get(f"/api/jobs/{job.id}/assemble").json()
    assert [v["version"] for v in st["versions"]] == [1, 2] and st["record"]["version"] == 2
    assert st["versions"][0]["url"].startswith(f"/media/vorch/{job.id[:8]}/assembly_v1.mp4?v=")

    r = client.post(f"/api/jobs/{job.id}/assemble/versions/1/select")
    assert r.status_code == 200 and r.json()["record"]["version"] == 1 and r.json()["versions"][0]["current"]
    assert (client.media / "vorch" / job.id[:8] / "assembly.mp4").read_bytes() == b"v1"
    assert client.post(f"/api/jobs/{job.id}/assemble/versions/9/select").status_code == 404

    r = client.delete(f"/api/jobs/{job.id}/assemble/versions/2")
    assert r.status_code == 200 and [v["version"] for v in r.json()["versions"]] == [1]
    client.app.state.manager._active.add(job.id)
    assert client.delete(f"/api/jobs/{job.id}/assemble/versions/1").status_code == 409
    client.app.state.manager._active.discard(job.id)
    assert client.delete(f"/api/jobs/{job.id}/assemble/versions/1").status_code == 200
    assert client.get(f"/api/jobs/{job.id}/assemble").json()["record"] is None


def test_assembly_speed_validation_routes(client, stub_media):
    job = _done_video_job(client)
    r = client.post(f"/api/jobs/{job.id}/assemble", json={"items": [{"scene_index": 0, "target_seconds": 99}, {"scene_index": 1}]})
    assert r.status_code == 400 and "outside" in r.json()["error"]["message"]
    assert client.post(f"/api/jobs/{job.id}/assemble", json={"items": [{"scene_index": 0, "target_seconds": -1}]}).status_code == 422
    st = client.get(f"/api/jobs/{job.id}/assemble").json()
    assert st["default_spec"]["items"][0]["target_seconds"] is None and st["default_spec"]["smooth"] is False


def test_job_reports_how_many_references_its_edit_workflow_uses(client):
    job = _paused_job_with_scene(client.store)
    assert client.get(f"/api/jobs/{job.id}").json()["edit_reference_slots"] == 2  # identity_edit: two images


# --- asset thumbnails, kind changes ---------------------------------------------------


def _project_with_image(client, tmp_path, size="1200x800"):
    import shutil
    import subprocess

    from vorch import projects as projects_mod

    p = projects_mod.save(client.store, projects_mod.new_project(name="Thumbs"))
    src = tmp_path / "big.jpg"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=teal:s={size}", "-frames:v", "1", str(src)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        src.write_bytes(b"\xff\xd8\xff")
    a = projects_mod.add_asset(p, src, kind="reference", name="big.jpg")
    junk = projects_mod.add_asset(p, b"not an image", kind="other", name="notes.txt")
    projects_mod.save(client.store, p)
    return p, a, junk


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
def test_asset_thumbnail_is_small_cached_and_immutable(client, tmp_path, monkeypatch):
    import subprocess

    from vorch import config, media, projects as projects_mod

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    p, a, junk = _project_with_image(client, tmp_path)
    got = client.get(f"/api/projects/{p.id}").json()["assets"][0]
    assert got["thumb_url"] == f"/api/projects/{p.id}/assets/{a.id}/thumb"

    r = client.get(got["thumb_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert "immutable" in r.headers["cache-control"]
    thumb = projects_mod.thumb_path(p, a.id)
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                                   "-of", "csv=p=0", str(thumb)]).decode().strip()
    assert out == "320,213"

    calls = []
    monkeypatch.setattr(media, "downscale", lambda *a_, **k: calls.append(1))
    assert client.get(got["thumb_url"]).content == thumb.read_bytes() and calls == []  # served from cache

    fallback = client.get(f"/api/projects/{p.id}/assets/{junk.id}/thumb")
    assert fallback.status_code == 200 and fallback.content == b"not an image"  # not shrinkable: the original
    assert client.get(f"/api/projects/{p.id}/assets/nope/thumb").status_code == 404
    assert not list(thumb.parent.glob(".*"))  # no temp files left behind


def test_patch_asset_kind_label_or_both(client, tmp_path, monkeypatch):
    from vorch import config, projects as projects_mod

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    p = projects_mod.save(client.store, projects_mod.new_project(name="Patch"))
    a = projects_mod.add_asset(p, b"x", kind="reference", name="x.jpg")
    projects_mod.set_asset_note(p, a.id, {"caption": "a smiling woman", "setting": "beach", "text_in_image": "MIZEN HEAD",
                                          "tags": ["portrait"], "is_portrait": True}, label="beach — a smiling woman")
    projects_mod.save(client.store, p)
    url = f"/api/projects/{p.id}/assets/{a.id}"

    r = client.patch(url, json={"kind": "master_face"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "master_face" and body["label"] == "a smiling woman" and body["label_source"] == "auto"
    assert body["note_tags"] == ["portrait"] and body["note_text"] == "MIZEN HEAD"

    r = client.patch(url, json={"kind": "reference", "label": "Bela on the beach"})
    assert r.json()["kind"] == "reference" and r.json()["label"] == "Bela on the beach" and r.json()["label_source"] == "human"
    assert client.patch(url, json={}).status_code == 422
    assert client.patch(url, json={"kind": "poster"}).status_code == 422
    assert client.patch(f"/api/projects/{p.id}/assets/nope", json={"kind": "logo"}).status_code == 404


def test_project_describe_hint_is_editable(client):
    pid = client.post("/api/projects", json={"name": "H", "describe_hint": "lighthouses"}).json()["id"]
    assert client.get(f"/api/projects/{pid}").json()["describe_hint"] == "lighthouses"
    r = client.put(f"/api/projects/{pid}", json={"describe_hint": "lighthouses; tag as fog_horn"})
    assert r.json()["describe_hint"] == "lighthouses; tag as fog_horn"


def test_scene_versions_are_public_and_deletable(client):
    from vorch.state import Clip, Scene, Variant, new_job

    job = new_job(idea="versions", kind="video")
    job.status = JobStatus.DONE
    job.scenes = [Scene(index=0, keyframe_path="/kf2.png", video_path="/clip2.mp4",
                        variants=[Variant("/kf1.png", id="k1"), Variant("/kf2.png", id="k2")],
                        clips=[Clip("/clip1.mp4", id="c1", keyframe_path="/kf1.png"),
                               Clip("/clip2.mp4", id="c2", keyframe_path="/kf2.png")])]
    client.store.save_job(job)

    s = client.get(f"/api/jobs/{job.id}").json()["scenes"][0]
    assert [(v["id"], v["active"]) for v in s["variants"]] == [("k1", False), ("k2", True)]
    assert [(c["id"], c["active"]) for c in s["clips"]] == [("c1", False), ("c2", True)]

    r = client.put(f"/api/jobs/{job.id}/scenes", json={"scenes": [{"index": 0, "chosen_variant_id": "k1"}]})
    assert r.status_code == 200
    s = r.json()["scenes"][0]
    assert s["keyframe_path"] == "/kf1.png" and s["video_path"] == "/clip1.mp4"  # its clip came back

    r = client.delete(f"/api/jobs/{job.id}/scenes/0/versions/c1")
    assert r.status_code == 200 and [c["id"] for c in r.json()["scenes"][0]["clips"]] == ["c2"]
    assert client.delete(f"/api/jobs/{job.id}/scenes/0/versions/nope").status_code == 400
    assert client.delete(f"/api/jobs/{job.id}/scenes/9/versions/k1").status_code == 404
    assert client.delete(f"/api/jobs/nope/scenes/0/versions/k1").status_code == 404


def test_assembly_status_offers_delivery_targets_and_accepts_one(client, stub_media, monkeypatch):
    from vorch import assembly as assembly_mod

    job = _done_video_job(client, profile={"purpose": "social_vertical", "aspect": "9:16",
                                           "video": {"width": 480, "height": 832, "fps": 16, "length": 49}})
    st = client.get(f"/api/jobs/{job.id}/assemble").json()
    by_id = {d["id"]: d for d in st["deliveries"]}
    assert st["delivery"] == "social_1080"                       # a vertical social job masters for a feed
    assert (by_id["social_1080"]["width"], by_id["social_1080"]["height"]) == (1080, 1920)
    assert by_id["social_1080"]["fps"] == 30 and by_id["source"]["width"] is None

    import time

    seen = {}
    monkeypatch.setattr(assembly_mod, "run", lambda j, s, p, **kw: seen.update(spec=s) or assembly_mod.AssemblyRecord(
        path="/out/assembly_v1.mp4", created_at="t", duration=6.0, n_clips=2, spec=s.to_dict(),
        fps=30, width=1080, height=1920, version=1))
    r = client.post(f"/api/jobs/{job.id}/assemble", json={"delivery": "social_1080", "grain": 1})
    assert r.status_code == 200
    for _ in range(50):          # the render runs in the background
        if "spec" in seen:
            break
        time.sleep(0.02)
    assert seen["spec"].delivery == "social_1080" and seen["spec"].grain == 1

    for _ in range(50):          # seeing the spec is not the task finishing: wait out the busy 409
        bad = client.post(f"/api/jobs/{job.id}/assemble", json={"delivery": "imax"})
        if bad.status_code != 409:
            break
        time.sleep(0.02)
    assert bad.status_code == 400 and "unknown delivery" in bad.json()["error"]["message"]
    assert client.post(f"/api/jobs/{job.id}/assemble", json={"grain": 9}).status_code == 400


# --- save a generated keyframe into the project (reuse it as a reference) --------------


def _job_with_keyframe(client, tmp_path, monkeypatch, *, with_project=True):
    from vorch import config, projects as projects_mod
    from vorch.state import Scene, Variant, new_job

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    p = projects_mod.save(client.store, projects_mod.new_project(name="Coast Co")) if with_project else None
    kf = client.media / "s00_keyframe.png"
    kf.write_bytes(b"tower-bytes")
    other = client.media / "s00_keyframe_v2.png"
    other.write_bytes(b"second-candidate")
    job = new_job(idea="towers", project_id=p.id if p else None)
    job.status = JobStatus.NEEDS_APPROVAL
    job.scenes = [Scene(index=0, keyframe_path=str(kf),
                        variants=[Variant(path=str(kf), id="v1"), Variant(path=str(other), id="v2")]),
                  Scene(index=1)]
    client.store.save_job(job)
    return job, p


def test_save_scene_keyframe_as_project_asset(client, tmp_path, monkeypatch):
    job, p = _job_with_keyframe(client, tmp_path, monkeypatch)
    r = client.post(f"/api/jobs/{job.id}/scenes/0/save-asset", json={})
    assert r.status_code == 201
    a = r.json()
    assert a["kind"] == "reference" and a["name"] == "scene-1-keyframe.png"
    assert a["label_source"] == "" and a["label"] == ""          # Gemma describes it in the background
    assert Path(a["path"]).read_bytes() == b"tower-bytes"        # a copy, not a pointer
    assert str(tmp_path / "projects") in a["path"]
    ids = [x["id"] for x in client.get(f"/api/projects/{p.id}").json()["assets"]]
    assert ids == [a["id"]]

    # The same frame again: the asset already holding those bytes, and no second copy.
    again = client.post(f"/api/jobs/{job.id}/scenes/0/save-asset", json={})
    assert again.status_code == 200 and again.json()["id"] == a["id"]
    assert len(client.get(f"/api/projects/{p.id}").json()["assets"]) == 1

    # A named version saves that candidate, labelled by the person who named it.
    v2 = client.post(f"/api/jobs/{job.id}/scenes/0/save-asset",
                     json={"version_id": "v2", "label": "striped tower"})
    assert v2.status_code == 201 and v2.json()["label_source"] == "human"
    assert Path(v2.json()["path"]).read_bytes() == b"second-candidate"


def test_save_scene_asset_rejects_what_it_cannot_save(client, tmp_path, monkeypatch):
    job, _ = _job_with_keyframe(client, tmp_path, monkeypatch)
    assert client.post(f"/api/jobs/{job.id}/scenes/9/save-asset", json={}).status_code == 404
    assert client.post(f"/api/jobs/{job.id}/scenes/1/save-asset", json={}).status_code == 404   # no keyframe yet
    bad = client.post(f"/api/jobs/{job.id}/scenes/0/save-asset", json={"version_id": "nope"})
    assert bad.status_code == 404 and "no keyframe version" in bad.json()["error"]["message"]
    assert client.post(f"/api/jobs/nope/scenes/0/save-asset", json={}).status_code == 404

    free, _ = _job_with_keyframe(client, tmp_path, monkeypatch, with_project=False)
    r = client.post(f"/api/jobs/{free.id}/scenes/0/save-asset", json={})
    assert r.status_code == 400 and "no project" in r.json()["error"]["message"]


def test_sse_keeps_listening_past_an_earlier_segment_end(store):
    """A job's ring buffer holds every segment, so a replay can contain an old `end`.

    While a segment is running that `end` is history: stopping there is what used to leave a
    regenerated keyframe invisible until the page was reloaded by hand. Driven directly, like
    the global stream test, because this stream deliberately does not close.
    """
    import json as _json

    from vorch.api.events import job_events
    from vorch.state import new_job

    job = new_job(idea="many segments")
    store.save_job(job)

    class _Req:
        headers: dict = {}
        query_params: dict = {"after": "0"}

    async def go():
        manager = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j))
        for ev in ({"type": "status", "status": "running"}, {"type": "keyframe", "index": 0}, {"type": "end"}):
            manager.broker.publish(job.id, ev)      # segment 1, already finished
        manager._active.add(job.id)                 # segment 2 is in flight
        resp = await job_events(job.id, _Req(), manager=manager, store=store)
        it = resp.body_iterator.__aiter__()
        seen = [await asyncio.wait_for(it.__anext__(), 2) for _ in range(4)]   # open + the replay
        manager.broker.publish(job.id, {"type": "keyframe", "index": 1})       # a live event after it
        seen.append(await asyncio.wait_for(it.__anext__(), 2))
        await it.aclose()
        return seen

    seen = asyncio.run(go())
    assert [e.event for e in seen] == ["open", "status", "keyframe", "end", "keyframe"]
    opening = _json.loads(seen[0].data)
    assert opening["last_id"] == 3 and opening["active"] is True
    assert _json.loads(seen[-1].data)["index"] == 1     # the live event the old client never saw


# --- structural scene edits over HTTP ---------------------------------------------------


def _storyboard_job(client, n=3):
    from vorch.state import Scene, new_job

    job = new_job(idea="coast", kind="video", n_scenes=n)
    job.status = JobStatus.NEEDS_APPROVAL
    job.scenes = [Scene(index=i, description=f"scene {i}") for i in range(n)]
    client.store.save_job(job)
    return job


def test_scene_routes_add_delete_and_reorder(client, monkeypatch):
    from vorch import storyboard
    from vorch.storyboard import PromptPair

    monkeypatch.setattr(storyboard, "keyframe_prompt", lambda b, **kw: PromptPair(positive="kf", negative=""))
    monkeypatch.setattr(storyboard, "animation_prompt", lambda b, **kw: PromptPair(positive="an", negative=""))
    job = _storyboard_job(client)

    r = client.post(f"/api/jobs/{job.id}/scenes", json={"position": 1, "description": "the lamp at night"})
    assert r.status_code == 201
    body = r.json()
    assert [s["description"] for s in body["scenes"]][1] == "the lamp at night"
    assert [s["index"] for s in body["scenes"]] == [0, 1, 2, 3] and body["n_scenes"] == 4
    ids = [s["id"] for s in body["scenes"]]
    assert all(ids) and len(set(ids)) == 4          # every scene has its own stable id

    order = [ids[3], ids[2], ids[1], ids[0]]
    r = client.put(f"/api/jobs/{job.id}/scenes/order", json={"scene_ids": order})
    assert r.status_code == 200 and [s["id"] for s in r.json()["scenes"]] == order
    assert [s["index"] for s in r.json()["scenes"]] == [0, 1, 2, 3]

    bad = client.put(f"/api/jobs/{job.id}/scenes/order", json={"scene_ids": order[:2]})
    assert bad.status_code == 400 and "every scene exactly once" in bad.json()["error"]["message"]

    r = client.delete(f"/api/jobs/{job.id}/scenes/0")
    assert r.status_code == 200 and [s["id"] for s in r.json()["scenes"]] == order[1:]
    assert client.delete(f"/api/jobs/{job.id}/scenes/9").status_code == 404
    assert client.post("/api/jobs/nope/scenes", json={"position": 0}).status_code == 404


def test_write_prompts_route_flags_the_scene_and_needs_a_description(client, monkeypatch):
    from vorch import storyboard
    from vorch.storyboard import PromptPair

    monkeypatch.setattr(storyboard, "keyframe_prompt", lambda b, **kw: PromptPair(positive="kf", negative=""))
    monkeypatch.setattr(storyboard, "animation_prompt", lambda b, **kw: PromptPair(positive="an", negative=""))
    job = _storyboard_job(client)
    r = client.post(f"/api/jobs/{job.id}/scenes/0/write-prompts")
    assert r.status_code == 202 and r.json()["scenes"][0]["writing_prompts"] is True

    blank = _storyboard_job(client)
    blank.scenes[0].description = ""
    client.store.save_job(blank)
    bad = client.post(f"/api/jobs/{blank.id}/scenes/0/write-prompts")
    assert bad.status_code == 400 and "no description" in bad.json()["error"]["message"]
    assert client.post(f"/api/jobs/{job.id}/scenes/9/write-prompts").status_code == 404


def test_version_detail_reads_the_record_then_the_sidecar(client, tmp_path, monkeypatch):
    """A version's prompt is history: saved with it, and read from the sidecar for old takes."""
    import json as _json

    from vorch import provenance
    from vorch.state import Clip, Scene, Variant, new_job

    kf, old, clip = client.media / "k2.png", client.media / "k1.png", client.media / "c1.mp4"
    for f in (kf, old, clip):
        f.write_bytes(b"x")
    provenance.manifest_path(old).write_text(_json.dumps({
        "workflow": "keyframe", "template": "image_flux2_klein_text_to_image.json", "backend": "local",
        "params": {"positive": "the old prompt", "negative": "smooth", "steps": 24, "cfg": 6.0, "seed": 42},
    }))
    job = new_job(idea="towers", kind="video")
    job.scenes = [Scene(
        index=0, keyframe_prompt="the current prompt", animation_prompt="slow pan",
        keyframe_path=str(kf), video_path=str(clip),
        variants=[Variant(path=str(old), id="v1", seed=42),                       # before prompts were saved
                  Variant(path=str(kf), id="v2", seed=53, prompt="the prompt that made it",
                          negative="waxy", created_at="2026-09-19T08:46:57+00:00", batch=1)],
        clips=[Clip(video_path=str(clip), id="c1", seed=7, prompt="slow pan")],
    )]
    client.store.save_job(job)

    saved = client.get(f"/api/jobs/{job.id}/scenes/0/versions/v2").json()
    assert saved["kind"] == "keyframe" and saved["prompt"] == "the prompt that made it"
    assert saved["negative"] == "waxy" and saved["prompt_from"] == "record" and saved["seed"] == 53
    assert saved["matches_scene_prompt"] is False          # the scene's prompt has moved on
    assert saved["has_manifest"] is False and saved["url"] == "/media/k2.png"

    legacy = client.get(f"/api/jobs/{job.id}/scenes/0/versions/v1").json()
    assert legacy["prompt"] == "the old prompt" and legacy["prompt_from"] == "manifest"
    assert legacy["params"]["steps"] == 24 and legacy["params"]["cfg"] == 6.0
    assert "positive" not in legacy["params"]              # the prompt is its own field
    assert legacy["workflow"] == "keyframe" and legacy["backend"] == "local" and legacy["has_manifest"] is True

    c = client.get(f"/api/jobs/{job.id}/scenes/0/versions/c1").json()
    assert c["kind"] == "clip" and c["prompt"] == "slow pan" and c["matches_scene_prompt"] is True

    assert client.get(f"/api/jobs/{job.id}/scenes/0/versions/nope").status_code == 404
    assert client.get(f"/api/jobs/{job.id}/scenes/9/versions/v1").status_code == 404


def test_animate_assets_job_refuses_a_face_swap_and_an_unknown_asset(client, tmp_path, monkeypatch):
    """Their own images are the keyframes: a swap would paste a face onto them."""
    from vorch import config, projects as projects_mod

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    p = projects_mod.save(client.store, projects_mod.new_project(name="Perfume Co"))
    asset = projects_mod.add_asset(p, b"x", kind="product", name="bottle.png")
    projects_mod.save(client.store, p)

    body = {"idea": "intimate, slow", "project_id": p.id, "keyframe_asset_ids": [asset.id],
            "face_swap": True, "master_face": client.master}
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 400 and "face swap off" in r.json()["error"]["message"]

    body["face_swap"] = False
    body["keyframe_asset_ids"] = ["ghost"]
    assert client.post("/api/jobs", json=body).status_code == 400

    body["keyframe_asset_ids"] = [asset.id]
    ok = client.post("/api/jobs", json=body)
    assert ok.status_code == 200 and ok.json()["project_id"] == p.id


def test_a_clip_thumbnail_is_a_poster_frame_never_the_clip(client, tmp_path):
    """`downscale` returns the source untouched when it is already small, and the route then
    serves it — which for a clip means streaming the whole file under an immutable cache."""
    import shutil
    import subprocess
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        import pytest as _pytest
        _pytest.skip("ffmpeg not installed")
    from vorch import projects

    p = projects.new_project("Stock")
    big = tmp_path / "big.mp4"
    small = tmp_path / "small.mp4"
    for path, size in ((big, "640x480"), (small, "160x120")):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc=size={size}:rate=8:duration=1", "-pix_fmt", "yuv420p", str(path)], check=True)
    a = projects.add_asset(p, big, kind="video", name="big.mp4")
    b = projects.add_asset(p, small, kind="video", name="small.mp4")
    projects.save(client.store, p)

    r = client.get(f"/api/projects/{p.id}/assets/{a.id}/thumb")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:3] == b"\xff\xd8\xff" and len(r.content) < 200_000

    # Already smaller than a thumbnail: there is nothing to serve, and the clip is not it.
    r2 = client.get(f"/api/projects/{p.id}/assets/{b.id}/thumb")
    assert r2.status_code in (200, 404)
    if r2.status_code == 200:
        assert r2.headers["content-type"] == "image/jpeg", "never the video itself"


def test_narration_url_changes_when_the_take_is_respoken(tmp_path, monkeypatch):
    """Re-speaking wrote the same filename, so the URL never changed and the browser kept playing
    the previous take from cache — a cached byte range against a file of a different length even
    answers 416. Every other kind of media is numbered by ComfyUI and changes name by itself."""
    import os
    import time

    from vorch import config
    from vorch.api.public import media_url

    out = tmp_path / "out"
    (out / "vorch").mkdir(parents=True)
    monkeypatch.setattr(config, "COMFY_OUTPUT", out)
    wav = out / "vorch" / "s06_narration.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 100)

    first = media_url(str(wav), bust=True)
    assert first is not None and first.startswith("/media/vorch/s06_narration.wav?v=")
    # The plain URL is unchanged for everything else.
    assert media_url(str(wav)) == "/media/vorch/s06_narration.wav"

    time.sleep(0.01)
    wav.write_bytes(b"RIFF" + b"\0" * 250)          # a different take, same path
    os.utime(wav, (time.time() + 2, time.time() + 2))
    assert media_url(str(wav), bust=True) != first

    # A missing file degrades to the plain URL rather than failing.
    wav.unlink()
    assert media_url(str(wav), bust=True) == "/media/vorch/s06_narration.wav"
    assert media_url(None, bust=True) is None
