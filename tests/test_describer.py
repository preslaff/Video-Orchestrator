"""Background asset description: queued on upload, stored as a suggested label, never over a
person's label, deferred while a job runs, and safe against concurrent project writes."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from vorch import config, projects
from vorch.api import create_app
from vorch.asset_vision import AssetNote
from vorch.describer import AssetDescriber
from vorch.queue import JobManager


@pytest.fixture(autouse=True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")


def _project(store, n=2):
    p = projects.new_project("Bela", describe_hint="lighthouses; tag as lamp_room")
    assets = [projects.add_asset(p, f"img{i}".encode(), kind="reference", name=f"p{i}.jpg") for i in range(n)]
    projects.save(store, p)
    return p, assets


class FakeGemma:
    def __init__(self, fail_on: set[str] | None = None, gate: threading.Event | None = None):
        self.calls: list[tuple[str, str, str]] = []
        self.fail_on = fail_on or set()
        self.gate = gate

    def __call__(self, path, kind, hint):
        self.calls.append((path, kind, hint))
        if self.gate is not None:
            self.gate.wait(2)
        if any(path.endswith(f) for f in self.fail_on):
            raise RuntimeError("Gemma unreachable")
        return AssetNote(caption=f"a lighthouse in {path.rsplit('/', 1)[-1]}", setting="headland", tags=["lighthouse"])


def test_describes_queued_assets_into_suggested_labels(make_store):
    store = make_store()
    p, (a0, a1) = _project(store)
    gemma = FakeGemma()

    async def go():
        d = AssetDescriber(store, describe=gemma, auto=True)
        assert d.enqueue(p.id, [a0.id, a1.id]) == 2
        assert d.enqueue(p.id, [a0.id]) == 0  # already queued
        assert d.state(a0.id)[0] == "queued"
        await d.join()
        return d

    d = asyncio.run(go())
    saved = projects.load(store, p.id)
    assert [(a.label_source, a.label.startswith("headland — a lighthouse")) for a in saved.assets] == [("auto", True)] * 2
    assert saved.assets[0].note["tags"] == ["lighthouse"]
    assert gemma.calls[0][1:] == ("reference", "lighthouses; tag as lamp_room")  # kind + project hint
    assert d.state(a0.id) == ("", "")


def test_waits_while_a_job_is_running(make_store):
    store = make_store()
    p, (a0,) = _project(store, n=1)
    busy = {"on": True}
    gemma = FakeGemma()

    async def go():
        d = AssetDescriber(store, describe=gemma, busy=lambda: busy["on"], poll_seconds=0.01)
        d.enqueue(p.id, [a0.id])
        await asyncio.sleep(0.1)
        assert gemma.calls == [] and d.state(a0.id)[0] == "queued"
        busy["on"] = False
        await d.join()

    asyncio.run(go())
    assert len(gemma.calls) == 1


def test_a_label_written_while_describing_wins(make_store):
    store = make_store()
    p, (a0,) = _project(store, n=1)
    gate = threading.Event()
    gemma = FakeGemma(gate=gate)

    async def go():
        d = AssetDescriber(store, describe=gemma)
        d.enqueue(p.id, [a0.id])
        while not gemma.calls:
            await asyncio.sleep(0.01)
        assert d.state(a0.id)[0] == "describing"
        with projects.editing(store, p.id) as fresh:        # the user names it meanwhile
            projects.set_asset_label(fresh, a0.id, "Hook Head at dusk")
            projects.save(store, fresh)
        gate.set()
        await d.join()

    asyncio.run(go())
    a = projects.load(store, p.id).assets[0]
    assert (a.label, a.label_source) == ("Hook Head at dusk", "human")


def test_skips_deleted_and_human_labelled_assets_and_reports_failures(make_store):
    store = make_store()
    p, (a0, a1, a2) = _project(store, n=3)
    projects.set_asset_label(p, a1.id, "named by hand")
    projects.remove_asset(p, a2.id)
    projects.save(store, p)
    gemma = FakeGemma(fail_on={"p0.jpg"})

    async def go():
        d = AssetDescriber(store, describe=gemma)
        d.enqueue(p.id, [a0.id, a1.id, a2.id])
        await d.join()
        return d

    d = asyncio.run(go())
    assert len(gemma.calls) == 1  # only a0 was looked at
    assert d.state(a0.id) == ("failed", "Gemma unreachable")
    assert d.state(a1.id) == ("", "") and d.state(a2.id) == ("", "")
    assert projects.load(store, p.id).asset(a1.id).label == "named by hand"


def test_the_edit_lock_keeps_concurrent_writes(make_store):
    """Two writers that each load, change and save must not drop each other's change."""
    store = make_store()
    p, (a0,) = _project(store, n=1)
    errors: list[BaseException] = []

    def add_many(tag):
        try:
            for i in range(10):
                with projects.editing(store, p.id) as fresh:
                    projects.add_asset(fresh, f"{tag}{i}".encode(), kind="reference", name=f"{tag}{i}.jpg")
                    time.sleep(0.002)  # widen the load-change-save window
                    projects.save(store, fresh)
        except BaseException as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=add_many, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(projects.load(store, p.id).assets) == 21


# --- through the API -------------------------------------------------------------------


@pytest.fixture
def client(make_store, tmp_path):
    store = make_store()
    manager = JobManager(store, auto_export=False)
    gemma = FakeGemma()
    describer = AssetDescriber(store, describe=gemma, busy=manager.busy, auto=True)
    app = create_app(store=store, manager=manager, media_dir=tmp_path, describer=describer)
    with TestClient(app) as c:
        c.store, c.describer, c.gemma = store, describer, gemma
        yield c


def _wait_described(c, pid, n, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        body = c.get(f"/api/projects/{pid}").json()
        if sum(1 for a in body["assets"] if a["label_source"] == "auto") >= n:
            return body
        time.sleep(0.02)
    raise AssertionError("assets were not described in time")


def test_upload_queues_description_unless_a_label_was_given(client):
    pid = client.post("/api/projects", json={"name": "Up"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/assets", files={"file": ("tower.jpg", b"x", "image/jpeg")},
                    data={"kind": "reference"})
    assert r.status_code == 201 and r.json()["describe_state"] in ("queued", "describing", "")
    named = client.post(f"/api/projects/{pid}/assets", files={"file": ("bela.jpg", b"y", "image/jpeg")},
                        data={"kind": "reference", "label": "Bela portrait"})
    assert named.json()["label_source"] == "human" and named.json()["describe_state"] == ""
    body = _wait_described(client, pid, 1)
    by_name = {a["name"]: a for a in body["assets"]}
    assert by_name["tower.jpg"]["label_source"] == "auto" and by_name["tower.jpg"]["note_tags"] == ["lighthouse"]
    assert by_name["bela.jpg"]["label"] == "Bela portrait"
    assert len(client.gemma.calls) == 1


def test_describe_endpoint_picks_unlabelled_by_default_and_suggestions_with_force(client):
    pid = client.post("/api/projects", json={"name": "D"}).json()["id"]
    with projects.editing(client.store, pid) as p:
        blank = projects.add_asset(p, b"a", kind="reference", name="blank.jpg")
        human = projects.add_asset(p, b"b", kind="reference", name="human.jpg", label="mine")
        auto = projects.add_asset(p, b"c", kind="reference", name="auto.jpg")
        projects.set_asset_note(p, auto.id, {"caption": "old"}, label="old suggestion")
        projects.save(client.store, p)

    r = client.post(f"/api/projects/{pid}/assets/describe", json={})
    assert r.status_code == 202 and r.json() == {"queued": 1}
    _wait_described(client, pid, 2)
    r = client.post(f"/api/projects/{pid}/assets/describe", json={"asset_ids": [auto.id, human.id], "force": True})
    assert r.json() == {"queued": 1}  # the human label is never redone
    assert client.post(f"/api/projects/{pid}/assets/describe", json={"asset_ids": ["nope"]}).status_code == 404


def test_a_video_upload_waits_to_be_asked(make_store, tmp_path, monkeypatch):
    """Three sampled frames cost about three times an image on the one serial CPU worker, so a
    clip is not captioned the moment it lands — the describe button still works."""
    from fastapi.testclient import TestClient

    from vorch.api import create_app
    from vorch.describer import AssetDescriber
    from vorch.queue import JobManager

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    store = make_store()
    p = projects.new_project("Stock")
    projects.save(store, p)
    describer = AssetDescriber(store, describe=lambda *a, **k: None, auto=True)
    app = create_app(store=store, manager=JobManager(store, auto_export=False), describer=describer)
    with TestClient(app) as c:
        img = c.post(f"/api/projects/{p.id}/assets",
                     files={"file": ("a.jpg", b"\xff\xd8\xff", "image/jpeg")}, data={"kind": "reference"})
        vid = c.post(f"/api/projects/{p.id}/assets",
                     files={"file": ("b.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")}, data={"kind": "video"})
    assert img.status_code == 201 and vid.status_code == 201
    assert img.json()["describe_state"] == "queued"
    assert vid.json()["describe_state"] == "", "a clip is not queued on upload"
