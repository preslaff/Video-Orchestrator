"""Projects: slugs, assets on disk, Store CRUD, job linkage."""

from __future__ import annotations

import pytest

from vorch import config, projects
from vorch.state import new_job


@pytest.fixture(autouse=True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(config, "EXPORT_ROOT", tmp_path / "export")


def test_slugify():
    assert projects.slugify("Кампания Есен 2026!") == "kampaniya-esen-2026"
    assert projects.slugify("  Café  Crème / Launch ") == "cafe-creme-launch"
    assert projects.slugify("x" * 80, max_len=10) == "x" * 10
    assert projects.slugify("!!!") == "untitled"


def test_new_project_and_export_root():
    p = projects.new_project("Autumn Campaign")
    assert p.slug == "autumn-campaign" and p.id and p.created_at == p.updated_at
    assert projects.export_root(p) == config.EXPORT_ROOT
    p.export_root = "/mnt/deliveries"
    assert str(projects.export_root(p)) == "/mnt/deliveries"
    assert projects.export_root(None) == config.EXPORT_ROOT


def test_add_asset_copies_file_hashes_and_master_face(tmp_path):
    p = projects.new_project("Faces")
    src = tmp_path / "Ivan Face.JPG"
    src.write_bytes(b"\xff\xd8\xffjpeg")
    a = projects.add_asset(p, src, kind="master_face", name=src.name, label="Ivan")
    assert a.path.startswith(str(config.PROJECTS_ROOT / "faces" / "assets")) and a.path.endswith("_ivan-face.jpg")
    assert len(a.sha256) == 64 and a.name == "Ivan Face.JPG"
    assert projects.master_face(p) is a
    b = projects.add_asset(p, b"pngbytes", kind="product", name="bottle.png")
    assert b.kind == "product" and (tmp_path / "projects" / "faces" / "assets").exists()
    assert [x.id for x in p.assets_of("product")] == [b.id]
    with pytest.raises(ValueError):
        projects.add_asset(p, src, kind="weird", name="x.png")
    with pytest.raises(FileNotFoundError):
        projects.add_asset(p, tmp_path / "missing.png", kind="reference", name="m.png")
    assert projects.remove_asset(p, a.id) is True and projects.master_face(p) is None
    assert projects.remove_asset(p, "nope") is False


def test_project_roundtrip_by_id_and_slug(store, tmp_path):
    p = projects.new_project("Round Trip", brand_notes="warm palette")
    src = tmp_path / "f.jpg"
    src.write_bytes(b"x")
    projects.add_asset(p, src, kind="reference", name="f.jpg")
    projects.save(store, p)
    got = projects.load(store, p.id)
    assert got.name == "Round Trip" and got.brand_notes == "warm palette" and len(got.assets) == 1
    assert got.assets[0].kind == "reference"
    assert projects.load_by_slug(store, "round-trip").id == p.id
    assert projects.load(store, "nope") is None and projects.load_by_slug(store, "nope") is None
    assert [x.id for x in projects.list_all(store)] == [p.id]


def test_duplicate_slug_rejected(store):
    projects.save(store, projects.new_project("Same Name"))
    with pytest.raises(projects.SlugTaken):
        projects.save(store, projects.new_project("Same Name"))
    # ...unless an explicit distinct slug is given.
    projects.save(store, projects.new_project("Same Name", slug="same-name-2"))


def test_delete_project_nulls_job_project_id(store):
    p = projects.save(store, projects.new_project("Del"))
    job = new_job(idea="j", project_id=p.id)
    store.save_job(job)
    assert store.count_jobs(project_id=p.id) == 1
    assert store.delete_project(p.id) is True
    assert store.count_jobs(project_id=p.id) == 0
    assert store.load_job(job.id).project_id is None  # job survives; link nulled
    assert store.list_jobs()[0].project_id is None
    assert store.delete_project(p.id) is False
