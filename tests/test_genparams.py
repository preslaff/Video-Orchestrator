"""GenParams: defaults, aspect constructors, profile mapping, seed derivation."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from vorch import config
from vorch.genparams import DEFAULT, GenParams, PostStep


def test_default_reproduces_constants():
    kw = DEFAULT.keyframe_kwargs()
    assert (kw["width"], kw["height"], kw["steps"], kw["cfg"]) == (1024, 576, 24, 5)
    assert "negative" not in kw and "unet" not in kw and kw["workflow"] == "keyframe"
    akw = DEFAULT.animate_kwargs()
    assert (akw["width"], akw["height"], akw["length"], akw["fps"]) == (832, 480, 49, 16)
    assert DEFAULT.clip_seconds == 3.06 and DEFAULT.orientation == "landscape"
    assert DEFAULT.validate() == []


def test_for_aspect_portrait_and_square():
    p = GenParams.for_aspect("9:16")
    assert p.keyframe_size == (576, 1024) and p.wan_size == (480, 832)
    assert p.orientation == "portrait"
    q = GenParams.for_aspect("1:1", length=81)
    assert q.keyframe_size == (768, 768) and q.wan_size == (640, 640) and q.length == 81


def test_validate_rejects_bad_sizes_and_mismatch():
    with pytest.raises(ValueError):
        GenParams(keyframe_size=(1000, 576)).validate()
    with pytest.raises(ValueError):
        GenParams(length=50).validate()
    with pytest.raises(ValueError):
        GenParams(keyframe_size=(1024, 576), wan_size=(480, 832)).validate()  # aspects differ
    with pytest.raises(ValueError):
        GenParams(kind="movie").validate()
    assert GenParams(kind="stills", wan_size=(1, 1)).validate() == []  # video settings ignored
    assert GenParams(length=121).validate()  # loud VRAM warning, not an error


def test_from_profile_nested_object_and_dict():
    profile = NS(
        id="social_9x16", name="Social 9:16", kind="video", aspect="9:16", purpose="social_vertical",
        keyframe=NS(workflow="keyframe", width=576, height=1024, steps=20, cfg=4.5, unet="flux-2-klein-base-4b-fp8.safetensors"),
        video=NS(workflow="animate", width=480, height=832, length=81, fps=16, negative="warped"),
        stills=NS(variants=1),
        style=NS(keyframe_prefix="phone photo", keyframe_negative="smooth", animation_suffix="handheld", animation_negative="jitter"),
        face_swap=NS(enabled=False, workflow="faceswap", restore_model="codeformer.pth"),
        evaluate=NS(enabled=True, max_attempts=3, review_threshold=5),
        edit_workflow="kv_edit",
        post_steps=[NS(workflow="gimm_interp", params={"multiplier": 2})],
    )
    p = GenParams.from_profile(profile)
    assert p.profile_name == "social_9x16" and p.aspect == "9:16" and p.purpose == "social_vertical"
    assert p.keyframe_size == (576, 1024) and p.wan_size == (480, 832) and p.length == 81
    assert p.steps == 20 and p.cfg == 4.5 and p.unet.startswith("flux-2-klein-base")
    assert p.keyframe_negative == "smooth" and p.animate_negative == "warped"
    assert p.style_prefix == "phone photo" and p.animation_suffix == "handheld"
    assert p.face_swap is False and p.restore_model == "codeformer.pth"
    assert p.max_attempts == 3 and p.review_threshold == 5
    assert p.edit_workflow == "kv_edit"
    assert p.post_steps == (PostStep("gimm_interp", {"multiplier": 2}),)
    assert p.keyframe_kwargs()["negative"] == "smooth" and p.keyframe_kwargs()["unet"] == p.unet

    # The same profile as a JSON snapshot (dict) maps identically.
    snap = {
        "id": "social_9x16", "kind": "video", "aspect": "9:16", "purpose": "social_vertical",
        "keyframe": {"workflow": "keyframe", "width": 576, "height": 1024, "steps": 20, "cfg": 4.5, "unet": p.unet},
        "video": {"workflow": "animate", "width": 480, "height": 832, "length": 81, "fps": 16, "negative": "warped"},
        "stills": {"variants": 1},
        "style": {"keyframe_prefix": "phone photo", "keyframe_negative": "smooth", "animation_suffix": "handheld", "animation_negative": "jitter"},
        "face_swap": {"enabled": False, "workflow": "faceswap", "restore_model": "codeformer.pth"},
        "evaluate": {"enabled": True, "max_attempts": 3, "review_threshold": 5},
        "edit_workflow": "kv_edit",
        "post_steps": [{"workflow": "gimm_interp", "params": {"multiplier": 2}}],
    }
    assert GenParams.from_profile(snap) == p


def test_from_profile_stills_without_video_block():
    p = GenParams.from_profile({"id": "stills_16x9", "kind": "stills", "aspect": "16:9",
                                "keyframe": {"width": 1024, "height": 576}, "stills": {"variants": 4}})
    assert p.kind == "stills" and p.variants_per_scene == 4 and p.wan_size == (832, 480)


def test_derive_seed_rule():
    assert config.derive_seed(100) == 100
    assert config.derive_seed(100, scene=2) == 20_100
    assert config.derive_seed(100, scene=1, variant=3) == 10_400
    assert config.derive_seed(100, scene=1, variant=3, attempt=2) == 10_401
    seeds = {config.derive_seed(100, scene=s, variant=v, attempt=a)
             for s in range(3) for v in range(8) for a in range(1, 4)}
    assert len(seeds) == 3 * 8 * 3  # all distinct
    assert config.derive_seed(config.SEED_RANDOM, scene=5) == config.SEED_RANDOM  # stays random
