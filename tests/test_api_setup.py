"""Setup-phase routes: env, config, workflows, profiles, projects + assets, export."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vorch import config, envcheck
from vorch.api import create_app
from vorch.envcheck import CheckResult, EnvReport
from vorch.queue import JobManager
from vorch.state import JobStatus, Scene, new_job


class FakeOrch:
    def __init__(self, store, job):
        self.store, self.emit = store, (lambda e: None)

    def run(self, job):
        job.status = JobStatus.DONE
        self.store.save_job(job)

    approve = run


@pytest.fixture
def client(tmp_path, make_store, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(config, "EXPORT_ROOT", tmp_path / "export")
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(config, "COMFY_OUTPUT", media)
    store = make_store()
    manager = JobManager(store, orchestrator_factory=lambda s, j: FakeOrch(s, j), auto_export=False)
    with TestClient(create_app(store=store, manager=manager, media_dir=media)) as c:
        c.store, c.media = store, media
        yield c


def test_env_and_config(client, monkeypatch):
    fake = EnvReport([CheckResult("comfy", "ok", "0.34")], "now", 1, profile_id="social_9x16")
    seen = {}

    def run_checks(**kw):
        seen.update(kw)
        return fake

    monkeypatch.setattr(envcheck, "run_checks", run_checks)
    r = client.get("/api/setup/env?deep=1&force=1&profile_id=social_9x16")
    assert r.status_code == 200 and r.json()["checks"][0]["name"] == "comfy" and r.json()["ok"] is True
    assert seen["deep"] is True and seen["force"] is True and seen["profile"].id == "social_9x16"
    assert client.get("/api/setup/env?profile_id=nope").status_code == 404
    cfg = client.get("/api/setup/config").json()
    assert cfg["schema_version"] == cfg["latest_schema_version"] >= 2 and "password" not in cfg["database_url"]
    assert cfg["export_root"].endswith("export")
    assert [b["name"] for b in cfg["backends"]] == ["local"] and cfg["backends_error"] is None
    assert "headers" not in cfg["backends"][0] and cfg["backends_file"].endswith("backends.yaml")


def test_workflows_listing_and_detail(client):
    body = client.get("/api/workflows").json()
    names = {w["name"] for w in body["workflows"]}
    assert {"keyframe", "animate", "faceswap", "identity_edit", "kv_edit"} <= names
    assert "postprocess" in body["kinds"]
    edit_names = {w["name"] for w in client.get("/api/workflows?kind=edit").json()["workflows"]}
    assert {"identity_edit", "kv_edit"} <= edit_names   # plus the curated hosted edit entries
    d = client.get("/api/workflows/animate").json()
    assert d["params"]["seed"]["targets"][0]["node"] == "9" and d["problems"] == []
    assert client.get("/api/workflows/nope").status_code == 404


def test_profiles_crud_clone_and_builtin_protection(client):
    ids = {p["id"] for p in client.get("/api/profiles").json()}
    assert "cinematic_16x9" in ids
    r = client.post("/api/profiles", json={"name": "Long 9:16", "base": "social_9x16", "overrides": {"video": {"length": 81}}})
    assert r.status_code == 201
    p = r.json()
    assert p["id"].startswith("u_") and p["video"]["length"] == 81 and p["builtin"] is False
    r = client.put(f"/api/profiles/{p['id']}", json={"video": {"length": 121}, "id": "hack", "builtin": True})
    assert r.status_code == 200 and r.json()["video"]["length"] == 121 and r.json()["id"] == p["id"]
    assert client.put(f"/api/profiles/{p['id']}", json={"video": {"length": 50}}).status_code == 422
    assert client.put("/api/profiles/cinematic_16x9", json={"name": "x"}).status_code == 403
    assert client.delete("/api/profiles/cinematic_16x9").status_code == 403
    c = client.post("/api/profiles/cinematic_16x9/clone", json={"name": "Mine"})
    assert c.status_code == 201 and c.json()["name"] == "Mine"
    assert client.get(f"/api/profiles/{p['id']}").status_code == 200
    assert client.delete(f"/api/profiles/{p['id']}").status_code == 204
    assert client.get(f"/api/profiles/{p['id']}").status_code == 404
    assert client.post("/api/profiles", json={"name": "x", "base": "nope"}).status_code == 404


def test_projects_crud_assets_and_jobs(client):
    r = client.post("/api/projects", json={"name": "Autumn Launch", "default_profile_id": "social_9x16"})
    assert r.status_code == 201
    p = r.json()
    assert p["slug"] == "autumn-launch" and p["export_root_effective"].endswith("export") and p["n_jobs"] == 0
    assert client.post("/api/projects", json={"name": "Autumn Launch"}).status_code == 409
    assert client.post("/api/projects", json={"name": "X", "default_profile_id": "nope"}).status_code == 400

    up = client.post(f"/api/projects/{p['id']}/assets",
                     files={"file": ("Ivan.jpg", b"\xff\xd8\xffjpeg", "image/jpeg")},
                     data={"kind": "master_face", "label": "Ivan"})
    assert up.status_code == 201
    a = up.json()
    assert a["kind"] == "master_face" and a["url"].endswith(f"/assets/{a['id']}/file")
    assert client.get(a["url"]).content == b"\xff\xd8\xffjpeg"
    assert client.post(f"/api/projects/{p['id']}/assets", files={"file": ("e.png", b"", "image/png")},
                       data={"kind": "reference"}).status_code == 400
    assert client.post(f"/api/projects/{p['id']}/assets", files={"file": ("e.png", b"x", "image/png")},
                       data={"kind": "weird"}).status_code == 422

    got = client.get(f"/api/projects/{p['id']}").json()
    assert len(got["assets"]) == 1
    r = client.put(f"/api/projects/{p['id']}", json={"brand_notes": "warm", "default_profile_id": "social_1x1"})
    assert r.json()["brand_notes"] == "warm" and r.json()["default_profile_id"] == "social_1x1"

    # Submitting with the project uses its master face + default profile.
    j = client.post("/api/jobs", json={"idea": "x", "project_id": p["id"]})
    assert j.status_code == 200 and j.json()["project_id"] == p["id"] and j.json()["profile_id"] == "social_1x1"
    assert j.json()["master_face"].endswith("_ivan.jpg")
    jobs = client.get(f"/api/projects/{p['id']}/jobs").json()
    assert [x["id"] for x in jobs] == [j.json()["id"]]
    assert client.get("/api/projects").json()[0]["n_jobs"] == 1

    assert client.delete(f"/api/projects/{p['id']}/assets/{a['id']}").status_code == 204
    assert client.delete(f"/api/projects/{p['id']}/assets/{a['id']}").status_code == 404
    assert client.delete(f"/api/projects/{p['id']}").status_code == 204
    assert client.get(f"/api/projects/{p['id']}").status_code == 404
    assert client.get(f"/api/jobs/{j.json()['id']}").json()["project_id"] is None


def test_export_routes(client, tmp_path):
    job = new_job(idea="Bread")
    kf = client.media / "k.png"
    kf.write_bytes(b"k")
    job.scenes = [Scene(index=0, keyframe_path=str(kf))]
    client.store.save_job(job)
    assert client.get(f"/api/jobs/{job.id}/export").json()["exported"] is False
    assert client.post(f"/api/jobs/{job.id}/export").status_code == 409     # not done
    r = client.post(f"/api/jobs/{job.id}/export?force=1")
    assert r.status_code == 200 and r.json()["files"] == ["s01_keyframe.png"] and r.json()["ok"]
    st = client.get(f"/api/jobs/{job.id}/export").json()
    assert st["exported"] and st["exists"] and st["path"].endswith("_bread")
    assert client.post("/api/jobs/nope/export").status_code == 404
    job.status = JobStatus.DONE
    client.store.save_job(job)
    assert client.post(f"/api/jobs/{job.id}/export").status_code == 200


def test_config_reports_when_the_server_runs_older_code(client, monkeypatch):
    """A route added after the server started answers a bare 404; the UI needs to say
    "restart vorch serve" instead of "not found", so the server reports the mismatch."""
    from vorch.api import setup as setup_api

    cfg = client.get("/api/setup/config").json()
    assert cfg["started_at"].endswith("+00:00") and cfg["code_changed_since_start"] is False

    monkeypatch.setattr(setup_api, "_STARTED_AT", 0.0)   # every module is newer than the epoch
    assert client.get("/api/setup/config").json()["code_changed_since_start"] is True


def test_the_config_poll_does_not_re_read_the_backends_file(client, monkeypatch):
    """The setup page polls /config every 30 s. Forcing a re-read on each poll re-parsed the
    YAML and logged a full backends line every time, which reads like a runaway loop."""
    from vorch import backends

    reads = []
    real = backends.load
    monkeypatch.setattr(backends, "load", lambda *a, **kw: reads.append(kw.get("force", False)) or real(*a, **kw))
    for _ in range(3):
        assert client.get("/api/setup/config").status_code == 200
    assert reads and not any(reads), "the config route must not force a re-read"
