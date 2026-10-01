"""Per-scene choice of the keyframe model.

Worth having per scene because the reasons differ shot by shot: one frame needs type drawn
inside the picture (Ideogram), the rest do not and should not pay for it. Mirrors
``Scene.edit_workflow`` exactly, including that an unknown name is a 400 when the scene is
patched rather than a KeyError halfway through the run.
"""

from __future__ import annotations

import pytest

from vorch import profiles, scene_assets
from vorch.genparams import GenParams
from vorch.orchestrator import Orchestrator
from vorch.state import Scene, _scene_from_dict, _scene_to_dict


def test_the_field_round_trips_and_defaults_to_the_profiles():
    assert Scene(index=0).keyframe_workflow == ""        # "" = the profile's
    s = Scene(index=0, keyframe_workflow="api_ideogram_v4_t2i")
    assert _scene_from_dict(_scene_to_dict(s)).keyframe_workflow == "api_ideogram_v4_t2i"


def test_scene_params_overrides_only_what_the_scene_names(make_store):
    store = make_store()
    try:
        prof = profiles.apply_overrides(profiles.default_profile(), {"face_swap": {"enabled": False}})
        o = Orchestrator(store, profile=prof)
        base = o.params.keyframe_workflow

        assert o.scene_params(Scene(index=0)).keyframe_workflow == base
        one = o.scene_params(Scene(index=0, keyframe_workflow="api_wan_t2i"))
        assert one.keyframe_workflow == "api_wan_t2i"
        assert one.edit_workflow == o.params.edit_workflow      # untouched
        # both at once, which the old single-field implementation could not express
        both = o.scene_params(Scene(index=0, keyframe_workflow="api_wan_t2i", edit_workflow="kv_edit"))
        assert (both.keyframe_workflow, both.edit_workflow) == ("api_wan_t2i", "kv_edit")
    finally:
        store.close()


@pytest.mark.parametrize("name,msg", [
    ("no_such_model", "unknown keyframe workflow"),
    ("identity_edit", "is kind 'edit', not 'keyframe'"),   # right name, wrong kind
])
def test_a_bad_name_is_rejected_when_the_scene_is_patched(name, msg):
    scene = Scene(index=2, keyframe_workflow=name)
    with pytest.raises(ValueError, match=msg):
        scene_assets.validate_scene(scene)


def test_a_good_name_passes_validation():
    scene_assets.validate_scene(Scene(index=0, keyframe_workflow="api_wan_t2i"))
    scene_assets.validate_scene(Scene(index=0))   # empty is always fine


def test_the_edit_workflow_check_still_works_the_same_way():
    """The two checks share one loop now; the original behaviour must be unchanged."""
    with pytest.raises(ValueError, match="unknown edit workflow"):
        scene_assets.validate_scene(Scene(index=0, mode="t2i", edit_workflow="nope"))
    with pytest.raises(ValueError, match="is kind 'keyframe', not 'edit'"):
        scene_assets.validate_scene(Scene(index=0, mode="t2i", edit_workflow="keyframe"))


def test_genparams_carries_it_into_the_keyframe_call():
    p = GenParams.for_aspect("16:9").with_(keyframe_workflow="api_wan_t2i")
    assert p.keyframe_kwargs()["workflow"] == "api_wan_t2i"
