"""Output profiles: built-ins, validation, overrides, legacy mapping, store CRUD."""

from __future__ import annotations

import pytest

from vorch import profiles
from vorch.genparams import GenParams
from vorch.profiles import OutputProfile

BUILTINS = {"cinematic_16x9", "social_9x16", "social_1x1", "stills_16x9"}


def test_builtins_load_validate_and_map_to_genparams():
    b = profiles.builtin_profiles()
    assert BUILTINS <= set(b)
    for p in b.values():
        assert p.builtin
        g = GenParams.from_profile(p)
        assert g.aspect == p.aspect and g.profile_name == p.id
    assert profiles.default_profile().id == "cinematic_16x9"
    s = b["stills_16x9"]
    assert s.kind == "stills" and s.video is None and s.stills.variants == 4
    assert GenParams.from_profile(s).variants_per_scene == 4
    v = b["social_9x16"]
    assert (v.keyframe.width, v.keyframe.height, v.video.width, v.video.height) == (576, 1024, 480, 832)
    assert v.workflows_used() == ["keyframe", "animate", "faceswap"]


def test_gates_for_kind():
    assert profiles.gates_for(profiles.builtin_profiles()["cinematic_16x9"]) == ["storyboard", "keyframes"]
    assert profiles.gates_for({"kind": "stills"}) == ["storyboard"]
    assert profiles.gates_for(None) == ["storyboard", "keyframes"]


def test_validation_rules():
    base = profiles.default_profile().model_dump(mode="json")
    with pytest.raises(ValueError):
        OutputProfile.model_validate({**base, "video": {**base["video"], "length": 50}})   # 4n+1
    with pytest.raises(ValueError):
        OutputProfile.model_validate({**base, "keyframe": {**base["keyframe"], "width": 1000}})  # %16
    with pytest.raises(ValueError):
        OutputProfile.model_validate({**base, "video": None})                              # video needs block
    with pytest.raises(ValueError):
        OutputProfile.model_validate({**base, "aspect": "9:16"})                            # size/aspect mismatch
    with pytest.raises(ValueError):
        OutputProfile.model_validate({**base, "aspect": "wide"})
    with pytest.raises(ValueError):
        OutputProfile.model_validate({**base, "stills": {"variants": 9}})
    stills = OutputProfile.model_validate({**base, "kind": "stills", "purpose": "stills"})
    assert stills.video is None  # dropped for stills


def test_apply_overrides_deep_merges_and_clone():
    p = profiles.default_profile()
    q = profiles.apply_overrides(p, {"video": {"length": 81}, "style": {"keyframe_prefix": "impasto"}})
    assert q.video.length == 81 and q.video.width == 832 and q.style.keyframe_prefix == "impasto"
    assert q.style.keyframe_negative == p.style.keyframe_negative  # untouched sibling
    assert profiles.apply_overrides(p, None) is p
    c = profiles.clone(p, "Long cinematic", overrides={"video": {"length": 121}})
    assert c.id.startswith("u_") and not c.builtin and c.name == "Long cinematic" and c.video.length == 121
    assert c.created_at and c.updated_at


def test_from_legacy_spec_maps_fields():
    p = profiles.from_legacy_spec({"idea": "x", "evaluate": False, "face_swap": False,
                                   "retry": {"max_attempts": 3}, "video": {"length": 81}})
    assert p.id == "cinematic_16x9" and p.evaluate.enabled is False and p.evaluate.max_attempts == 3
    assert p.face_swap.enabled is False and p.video.length == 81
    q = profiles.from_legacy_spec({"idea": "x", "aspect": "9:16"})
    assert (q.keyframe.width, q.video.height) == (576, 832) and q.aspect == "9:16"
    r = profiles.from_legacy_spec({"idea": "x", "profile": "stills_16x9", "profile_overrides": {"stills": {"variants": 2}}})
    assert r.kind == "stills" and r.stills.variants == 2
    with pytest.raises(KeyError):
        profiles.from_legacy_spec({"profile": "nope"})


def test_legacy_profile_for_job():
    from vorch.state import new_job

    job = new_job(idea="old", evaluate=False, max_attempts=4, face_swap=False)
    p = profiles.legacy_profile_for_job(job)
    assert p.evaluate.enabled is False and p.evaluate.max_attempts == 4 and p.face_swap.enabled is False


def test_store_roundtrip_and_builtin_fallback(store):
    user = profiles.clone(profiles.builtin_profiles()["social_9x16"], "My vertical",
                          overrides={"video": {"length": 81}})
    profiles.save(store, user)
    got = profiles.resolve(store, user.id)
    assert got.name == "My vertical" and got.video.length == 81 and not got.builtin
    assert profiles.resolve(store, "cinematic_16x9").builtin
    assert profiles.resolve(store, None).id == "cinematic_16x9"
    with pytest.raises(KeyError):
        profiles.resolve(store, "nope")
    ids = {p.id for p in profiles.list_all(store)}
    assert BUILTINS <= ids and user.id in ids
    with pytest.raises(PermissionError):
        profiles.save(store, profiles.builtin_profiles()["cinematic_16x9"])
    assert store.delete_profile(user.id) is True and store.delete_profile(user.id) is False
    assert store.load_profile(user.id) is None


def test_snapshot_roundtrips_through_genparams():
    p = profiles.builtin_profiles()["social_1x1"]
    snap = p.snapshot()
    assert GenParams.from_profile(snap) == GenParams.from_profile(p)
    assert OutputProfile.model_validate(snap) == p


def test_from_legacy_spec_resolves_user_profiles_with_a_store(store):
    cloud = profiles.save(store, profiles.clone(profiles.default_profile(), "Cloud",
                                                overrides={"video": {"length": 81}}))
    p = profiles.from_legacy_spec({"idea": "x", "profile": cloud.id}, store)
    assert p.id == cloud.id and p.video.length == 81
    with pytest.raises(KeyError, match="user profiles"):
        profiles.from_legacy_spec({"idea": "x", "profile": "u_nope"}, store)
    with pytest.raises(KeyError):
        profiles.from_legacy_spec({"idea": "x", "profile": cloud.id})  # no store: built-ins only
