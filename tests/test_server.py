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

    job = new_job(idea="asm", kind="video", profile={"video": {"width": 832, "height": 480, "fps": 16, "length": 49}}, **extra)
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
