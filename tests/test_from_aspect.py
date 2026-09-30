"""Aspect-derived size params (`from_aspect`).

A hosted API node usually expresses size as an enum in its own dialect ("1792x1024", "16:9",
"1080:1920") instead of width/height ints. `from_aspect` maps the job's aspect onto that label,
and an aspect the table has no entry for is refused loudly — at validate/submit, not at render,
because the silent failure is a 9:16 job coming back square.
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from vorch import config, formats, genparams, profiles, registry


def _reg(tmp_path, table, *, wire_size=False, field="size"):
    user = tmp_path / "user_workflows"
    user.mkdir(exist_ok=True)
    shutil.copy(config.WORKFLOW_DIR / "image_flux2_klein_text_to_image.json", user / "api.json")
    params = {
        "positive": {"node": "76", "field": "value", "type": "str"},
        "seed": {"targets": [], "type": "int"},
        "height": {"targets": [], "type": "int"},
        "aspect": {"node": "75:67", "field": "text", "type": "enum", "from_aspect": table},
    }
    params["width"] = ({"node": "75:68", "field": "value", "type": "int",
                        "constraint": "multiple_of_16", "min": 256, "max": 4096}
                       if wire_size else {"targets": [], "type": "int"})
    (user / "api_t2i.yaml").write_text(yaml.safe_dump({
        "name": "api_t2i", "kind": "keyframe", "template": "api.json",
        "outputs": {"type": "image", "nodes": ["9"]}, "params": params,
    }))
    return registry.load_registry([config.WORKFLOW_DIR, user], force=True)


TABLE = {"16:9": "1792x1024", "9:16": "1024x1792", "1:1": "1024x1024"}


def test_aspect_maps_to_the_nodes_own_label(tmp_path):
    reg = _reg(tmp_path, TABLE)
    for aspect, label in TABLE.items():
        wf = reg.apply("api_t2i", strict=False, positive="x", seed=1, aspect=aspect)
        assert wf["75:67"]["inputs"]["text"] == label


@pytest.mark.parametrize("table,label", [
    ({"9:16": "9:16"}, "9:16"),                    # plain ratio
    ({"9:16": "1080:1920"}, "1080:1920"),          # Runway's colon-separated pixels
    ({"9:16": "1024x1792"}, "1024x1792"),          # literal WxH
    ({"9:16": "1440x2560 (9:16)"}, "1440x2560 (9:16)"),  # Ideogram's decorated label
])
def test_every_label_dialect_round_trips(tmp_path, table, label):
    reg = _reg(tmp_path, table)
    wf = reg.apply("api_t2i", strict=False, positive="x", seed=1, aspect="9:16")
    assert wf["75:67"]["inputs"]["text"] == label


def test_unrenderable_aspect_is_refused_not_silently_defaulted(tmp_path):
    reg = _reg(tmp_path, {"16:9": "1792x1024"})
    with pytest.raises(ValueError, match="cannot render aspect '9:16'"):
        reg.apply("api_t2i", strict=False, positive="x", seed=1, aspect="9:16")


def test_rules_report_the_enum_and_a_wired_size_overrides_it(tmp_path):
    reg = _reg(tmp_path, TABLE)
    registry.install(reg)
    try:
        r = formats.keyframe_rules("api_t2i")
        assert r.from_aspect and r.aspects == ("16:9", "1:1", "9:16")  # plain string sort
        # a workflow that really does take ints is not aspect-driven, and its own max wins
        reg2 = _reg(tmp_path, TABLE, wire_size=True)
        registry.install(reg2)
        r2 = formats.keyframe_rules("api_t2i")
        assert not r2.from_aspect and r2.size_max == 4096
        formats.validate_keyframe_size(3840, 2160, "16:9", "api_t2i")  # over the old 2048 ceiling
    finally:
        registry.reset_cache()


def test_profile_and_genparams_refuse_an_unrenderable_aspect(tmp_path):
    reg = _reg(tmp_path, {"16:9": "1792x1024"})
    registry.install(reg)
    try:
        with pytest.raises(ValueError, match="cannot render aspect"):
            profiles.apply_overrides(profiles.default_profile(), {
                "aspect": "9:16",
                "keyframe": {"workflow": "api_t2i", "width": 576, "height": 1024},
                "video": {"width": 480, "height": 832},
            })
        # for_aspect validates in the constructor, so the refusal lands there
        with pytest.raises(ValueError, match="cannot render aspect"):
            genparams.GenParams.for_aspect("9:16", keyframe_workflow="api_t2i")
        ok = genparams.GenParams.for_aspect("16:9", keyframe_workflow="api_t2i")
        assert ok.validate() == []
    finally:
        registry.reset_cache()


def test_bad_from_aspect_tables_are_rejected_at_parse(tmp_path):
    tpl = config.WORKFLOW_DIR / "image_flux2_klein_text_to_image.json"
    for table, msg in (({}, "non-empty mapping"),
                       ({"wide": "1792x1024"}, "not an aspect"),
                       ({"16:9": ""}, "non-empty string"),
                       ({"16:9": 1024}, "non-empty string")):
        with pytest.raises(registry.RegistryError, match=msg):
            registry.parse_spec({
                "name": "bad", "kind": "keyframe", "template": tpl.name,
                "params": {"positive": {"node": "76", "field": "value"},
                           "seed": {"targets": []}, "width": {"targets": []},
                           "height": {"targets": []},
                           "aspect": {"node": "75:67", "field": "text", "from_aspect": table}},
            }, source=tpl.with_suffix(".yaml"))


def test_missing_aspect_is_refused_rather_than_rendering_the_default(tmp_path, monkeypatch):
    """The hole a live run found: callers that compute width/height from the aspect and never
    pass the aspect itself would silently render the template's own size."""
    from vorch.steps import keyframe as kfmod

    reg = _reg(tmp_path, TABLE)
    monkeypatch.setattr(registry, "load_registry", lambda *a, **k: reg)
    monkeypatch.setattr(kfmod, "run_step", lambda *a, **k: pytest.fail("should not have submitted"))
    with pytest.raises(ValueError, match="takes its size from the aspect"):
        kfmod.keyframe("x", workflow="api_t2i", width=1024, height=576, seed=1)


def test_combo_options_read_both_object_info_encodings():
    """Local (V1) nodes inline the option list; api_node (V3) nodes put it in the metadata.
    Reading only the first shape made every hosted enum look empty."""
    v1 = {"UNETLoader": {"input": {"required": {"unet_name": [["a.safetensors", "b.safetensors"]]}}}}
    v3 = {"OpenAIDalle3": {"input": {"optional": {"size": ["COMBO", {"options": ["1024x1024", "1792x1024"]}]}}}}
    assert registry._combo_options(v1, "UNETLoader", "unet_name") == ["a.safetensors", "b.safetensors"]
    assert registry._combo_options(v3, "OpenAIDalle3", "size") == ["1024x1024", "1792x1024"]
    assert registry._combo_options(v3, "OpenAIDalle3", "nope") is None
    assert registry._combo_options({}, "Missing", "x") is None
