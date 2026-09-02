"""resolve_submission: profile order, legacy knobs, project assets, references."""

from __future__ import annotations

import pytest

from vorch import config, profiles, projects
from vorch.submission import resolve_submission


@pytest.fixture(autouse=True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")


def test_profile_resolution_order(store):
    user = profiles.save(store, profiles.clone(profiles.builtin_profiles()["social_1x1"], "Square long",
                                               overrides={"video": {"length": 81}}))
    proj = projects.save(store, projects.new_project("P", default_profile_id="social_9x16"))
    assert resolve_submission(store).profile.id == "cinematic_16x9"                       # built-in default
    assert resolve_submission(store, project_id=proj.id).profile.id == "social_9x16"       # project default
    assert resolve_submission(store, project_id=proj.id, profile_id=user.id).profile.video.length == 81
    with pytest.raises(KeyError):
        resolve_submission(store, project_id="nope")
    with pytest.raises(KeyError):
        resolve_submission(store, profile_id="nope")


def test_overrides_and_legacy_fields_only_when_set(store):
    s = resolve_submission(store, profile_overrides={"video": {"length": 121}, "style": {"keyframe_prefix": "impasto"}})
    assert s.profile.video.length == 121 and s.profile.style.keyframe_prefix == "impasto"
    assert s.profile.evaluate.enabled is True and s.profile.face_swap.enabled is True   # untouched
    t = resolve_submission(store, evaluate=False, face_swap=False, max_attempts=4)
    assert t.profile.evaluate.enabled is False and t.profile.face_swap.enabled is False
    assert t.profile.evaluate.max_attempts == 4 and t.kind == "video"
    with pytest.raises(ValueError):
        resolve_submission(store, profile_overrides={"video": {"length": 50}})


def test_master_face_and_references_from_project_assets(store, tmp_path):
    proj = projects.new_project("Faces")
    face = tmp_path / "face.jpg"
    face.write_bytes(b"f")
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"r")
    a_face = projects.add_asset(proj, face, kind="master_face", name="face.jpg")
    a_ref = projects.add_asset(proj, ref, kind="reference", name="ref.png")
    projects.save(store, proj)

    s = resolve_submission(store, project_id=proj.id, reference_asset_ids=[a_ref.id],
                           reference_images=["/extra/r2.png"])
    assert s.master_face == a_face.path and s.project_id == proj.id
    assert s.reference_images == [a_ref.path, "/extra/r2.png"]
    # An explicit path wins over the project asset.
    assert resolve_submission(store, project_id=proj.id, master_face="/my/face.jpg").master_face == "/my/face.jpg"
    with pytest.raises(KeyError):
        resolve_submission(store, project_id=proj.id, reference_asset_ids=["nope"])
    with pytest.raises(KeyError):
        resolve_submission(store, reference_asset_ids=[a_ref.id])  # no project
    assert resolve_submission(store).master_face == ""
