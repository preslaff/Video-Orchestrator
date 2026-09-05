"""Aspect presets, computed sizes, validation and VRAM warnings (vorch/formats.py)."""

from __future__ import annotations

import pytest

from vorch import formats


@pytest.mark.parametrize("aspect", list(formats.ASPECTS))
def test_presets_are_16_aligned_and_match_aspect(aspect):
    for w, h in (formats.keyframe_size(aspect), formats.wan_size(aspect)):
        assert w % 16 == 0 and h % 16 == 0
        assert abs((w / h) - formats.ratio(aspect)) / formats.ratio(aspect) < 0.06


def test_computed_sizes_for_unlisted_aspect():
    w, h = formats.keyframe_size("21:9")
    assert w % 16 == 0 and h % 16 == 0 and 0.5e6 <= w * h <= 0.7e6 and w > h
    w, h = formats.wan_size("2:3")
    assert (w, h) == (480, 720)


def test_orientation_and_hint():
    assert formats.orientation("16:9") == "landscape"
    assert formats.orientation("9:16") == "portrait"
    assert formats.orientation("1:1") == "square"
    assert "vertical" in formats.framing_hint("9:16")


def test_parse_aspect_rejects_garbage():
    for bad in ("wide", "16x9", "0:1", ""):
        with pytest.raises(ValueError):
            formats.parse_aspect(bad)


def test_validate_size_and_length():
    formats.validate_size(1024, 576)
    with pytest.raises(ValueError):
        formats.validate_size(1000, 576)
    with pytest.raises(ValueError):
        formats.validate_size(128, 576)
    formats.validate_length(81)
    with pytest.raises(ValueError):
        formats.validate_length(80)


def test_vram_warnings_thresholds():
    assert formats.vram_warnings(832, 480, 49) == []
    assert formats.vram_warnings(832, 480, 81) == []            # 1.65x landscape: fine
    assert len(formats.vram_warnings(480, 832, 81)) == 1        # 1.65x portrait: watch
    assert "OOM" in formats.vram_warnings(832, 480, 121)[0]     # 2.47x: loud
    assert formats.clip_seconds(49, 16) == 3.06


def test_aspects_match_tolerance():
    assert formats.aspects_match((1024, 576), (832, 480))
    assert formats.aspects_match((576, 1024), (480, 832))   # 2.5% apart after 16-alignment
    assert not formats.aspects_match((1024, 576), (480, 832))
    assert not formats.aspects_match((768, 768), (832, 480))


# --- workflow-driven rules (Phase C: LTX 8n+1 / 32, remote backends) ------------------


def _ltx_like_workflow(tmp_path, monkeypatch, *, backend="modal-h100"):
    import json

    import yaml

    from vorch import config, registry

    tpl = tmp_path / "ltx.json"
    tpl.write_text(json.dumps({
        "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
        "2": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 1280, "height": 720, "length": 97, "batch_size": 1}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "p", "clip": ["1", 0]}},
        "4": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "v", "video": ["2", 0]}},
    }))
    config.USER_WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    (config.USER_WORKFLOW_DIR / "animate_ltx_test.yaml").write_text(yaml.safe_dump({
        "kind": "animate", "template": str(tpl), "backend": backend,
        "inputs": {"start_image": "image"}, "outputs": {"type": "video", "nodes": ["4"]},
        "params": {
            "start_image": {"node": "1", "field": "image", "type": "str", "required": True},
            "positive": {"node": "3", "field": "text", "type": "str", "required": True},
            "length": {"node": "2", "field": "length", "type": "int", "constraint": "8n+1", "required": True},
            "width": {"node": "2", "field": "width", "type": "int", "constraint": "multiple_of_32", "max": 1920},
            "height": {"node": "2", "field": "height", "type": "int", "constraint": "multiple_of_32", "max": 1920},
        },
    }))
    registry.reset_cache()
    return "animate_ltx_test"


def test_constraints_8n1_and_multiple_of_32():
    from vorch import registry

    registry.CONSTRAINTS["8n+1"](97)
    registry.CONSTRAINTS["multiple_of_32"](1280)
    registry.CONSTRAINTS["8n+1"](81)  # 81 = 8*10+1 satisfies both Wan and LTX
    with pytest.raises(ValueError, match="8n\\+1"):
        registry.CONSTRAINTS["8n+1"](85)  # 4n+1 but not 8n+1
    with pytest.raises(ValueError, match="multiple of 32"):
        registry.CONSTRAINTS["multiple_of_32"](1264)


def test_workflow_rules_defaults_and_ltx(tmp_path, monkeypatch):
    default = formats.workflow_rules("animate")
    assert (default.length_constraint, default.size_multiple, default.size_max, default.remote) == ("4n+1", 16, 1280, False)
    unknown = formats.workflow_rules("not_installed_yet")
    assert unknown.length_constraint == "4n+1" and unknown.known is False
    assert formats.workflow_rules(None) == formats.WorkflowRules()
    name = _ltx_like_workflow(tmp_path, monkeypatch)
    rules = formats.workflow_rules(name)
    assert (rules.length_constraint, rules.size_multiple, rules.size_max, rules.remote) == ("8n+1", 32, 1920, True)
    assert rules.short_side_default() == 720


def test_720p_presets_and_rounding():
    assert formats.wan_size("16:9") == (832, 480)
    assert formats.wan_size("16:9", short_side=720) == (1280, 720)
    assert formats.wan_size("9:16", short_side=720) == (720, 1280)
    assert all(w % 16 == 0 and h % 16 == 0 for w, h in formats.WAN_PRESETS_720.values())
    assert formats.wan_size("16:9", short_side=720, multiple=32) == (1280, 704)   # LTX: 720 is not /32
    assert formats.wan_size("9:16", short_side=720, multiple=32) == (704, 1280)
    assert formats.wan_size("1:1", short_side=720, multiple=32) == (960, 960)     # preset already /32
    assert formats.wan_size("21:9", short_side=720, multiple=32) == (1664, 704)  # 1680/32 = 52.5 -> banker's 52
    formats.validate_size(1280, 704, multiple=32)
    with pytest.raises(ValueError, match="multiple of 32"):
        formats.validate_size(1264, 720, multiple=32)


def test_profile_and_params_follow_the_workflow_rules(tmp_path, monkeypatch):
    from vorch import profiles
    from vorch.genparams import GenParams

    name = _ltx_like_workflow(tmp_path, monkeypatch)
    ltx = profiles.apply_overrides(profiles.default_profile(), {
        "keyframe": {"width": 1280, "height": 704},
        "video": {"workflow": name, "width": 1280, "height": 704, "length": 97, "fps": 24},
    })
    assert ltx.video.length == 97
    with pytest.raises(ValueError, match="8n\\+1"):
        profiles.apply_overrides(ltx, {"video": {"length": 85}})       # fine for Wan (4n+1), not for LTX
    with pytest.raises(ValueError, match="multiple of 32"):
        profiles.apply_overrides(ltx, {"video": {"width": 1264}})
    # a remote animate workflow: no local VRAM warning even far above the 16 GB budget
    assert GenParams.from_profile(ltx).validate() == []
    wan = profiles.apply_overrides(profiles.default_profile(), {"video": {"length": 121}})
    assert GenParams.from_profile(wan).validate()                        # local Wan still warns
