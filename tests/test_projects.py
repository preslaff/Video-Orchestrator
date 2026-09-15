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


def test_identity_assets_resolve_paths_and_labels(tmp_path, monkeypatch):
    from vorch import config, projects

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    p = projects.new_project("Ages")
    young = projects.add_asset(p, b"a", kind="master_face", name="young.png", label="Maria, age 8")
    old = projects.add_asset(p, b"b", kind="master_face", name="old.png")
    prod = projects.add_asset(p, b"c", kind="product", name="bottle.png")
    assert [a.id for a in projects.identity_assets(p)] == [young.id, old.id]
    assert projects.identity_assets(None) == []
    assert projects.resolve_asset_paths(p, [prod.id, young.id]) == [prod.path, young.path]
    assert projects.resolve_asset_paths(p, []) == [] and projects.resolve_asset_paths(None, []) == []
    with pytest.raises(KeyError, match="nope"):
        projects.resolve_asset_paths(p, ["nope"])
    with pytest.raises(KeyError, match="no project"):
        projects.resolve_asset_paths(None, [young.id])
    assert projects.set_asset_label(p, old.id, "  Maria, age 70 ").label == "Maria, age 70"
    with pytest.raises(KeyError):
        projects.set_asset_label(p, "nope", "x")


# --- asset notes and label provenance -------------------------------------------------


def test_label_provenance_human_vs_auto(tmp_path, monkeypatch):
    """Who wrote a label decides whether a captioning run may replace it."""
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    p = projects.new_project("Notes")
    a = projects.add_asset(p, b"a", kind="reference", name="fb_780482127.jpg")
    assert a.label_source == "" and a.note is None

    note = {"caption": "a lighthouse on a headland", "tags": ["exterior"]}
    projects.set_asset_note(p, a.id, note, label="coastal headland — a lighthouse")
    assert a.label_source == "auto" and a.note == note

    projects.set_asset_label(p, a.id, " Fanad Head, evening ")
    assert a.label == "Fanad Head, evening" and a.label_source == "human"

    # A later run may refresh the description without touching the person's wording.
    projects.set_asset_note(p, a.id, {"caption": "redescribed"})
    assert a.note == {"caption": "redescribed"}
    assert a.label == "Fanad Head, evening" and a.label_source == "human"

    with pytest.raises(KeyError):
        projects.set_asset_note(p, "nope", {})


def test_asset_note_round_trips_and_old_assets_load(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    p = projects.new_project("Round")
    a = projects.add_asset(p, b"a", kind="reference", name="x.png")
    projects.set_asset_note(p, a.id, {"caption": "нещо на брега"}, label="брегът")
    back = projects.project_from_dict(projects.project_to_dict(p))
    assert back.assets[0].note == {"caption": "нещо на брега"}
    assert back.assets[0].label_source == "auto" and back.assets[0].label == "брегът"

    # Stored before this feature existed: no note, no provenance, still loads.
    legacy = {"id": "abc", "kind": "reference", "name": "old.png", "path": "/tmp/old.png",
              "label": "hand typed", "sha256": "", "created_at": ""}
    old = projects.asset_from_dict(legacy)
    assert old.note is None and old.label_source == "" and old.label == "hand typed"
