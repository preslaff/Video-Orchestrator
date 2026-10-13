"""GPU-free checks for the workflow registry — node surgery and validation.

Loads the real templates from workflows/ and confirms apply_params writes the
expected node fields and that Wan length validation rejects non-4n+1 values.
"""

import json

import pytest

from vorch import registry, workflows


def test_registry_node_ids_exist_in_templates():
    """Every (node_id, field) target in every built-in YAML must be a literal in its template."""
    reg = registry.load_registry()
    assert {"keyframe", "animate", "faceswap", "identity_edit", "kv_edit"} <= set(reg.names())
    for spec in reg.list():
        assert reg.validate(spec) == [], spec.name


def test_apply_params_sets_fields_without_mutating_template():
    wf = workflows.apply_params(
        "keyframe", positive="a cat", seed=42, width=768, height=512
    )
    assert wf["76"]["inputs"]["value"] == "a cat"
    assert wf["75:73"]["inputs"]["noise_seed"] == 42
    assert wf["75:68"]["inputs"]["value"] == 768
    assert wf["75:69"]["inputs"]["value"] == 512

    # The on-disk template must be untouched (apply_params deep-copies).
    fresh = workflows.load_template("keyframe")
    assert fresh["76"]["inputs"]["value"] != "a cat"


def test_apply_params_rejects_unknown_param():
    with pytest.raises(KeyError):
        workflows.apply_params("keyframe", nonsense=1)


@pytest.mark.parametrize("length", [49, 81, 121])
def test_wan_length_accepts_4n_plus_1(length):
    wf = workflows.apply_params(
        "animate", start_image="x.png", positive="move", length=length
    )
    assert wf["8"]["inputs"]["length"] == length


@pytest.mark.parametrize("length", [48, 50, 80, 0, -3])
def test_wan_length_rejects_non_4n_plus_1(length):
    with pytest.raises(ValueError):
        workflows.apply_params("animate", length=length)


def test_templates_are_api_format():
    """API-format templates are flat node dicts, not UI exports (no 'nodes' key)."""
    for spec in registry.load_registry().list():
        data = json.load(open(spec.template))
        assert "nodes" not in data, f"{spec.name}: looks like a UI export, not API format"
        assert all("class_type" in node for node in data.values())


def test_legacy_workflows_shim():
    """Old callers (and manifests) keep working through the REGISTRY view."""
    legacy = workflows.REGISTRY
    assert legacy["keyframe"]["template"] == "image_flux2_klein_text_to_image.json"
    assert legacy["keyframe"]["params"]["seed"] == ("75:73", "noise_seed")
    assert legacy["animate"]["output_node"] == "13"
    assert workflows.output_node("faceswap") == "4"


def test_the_matte_workflow_reports_a_missing_custom_node_and_checkpoint():
    """`comfy_nodes/README.md` tells the user `registry validate --online` is the install check.

    This is that claim, tested. `_finalize` puts every template `class_type` into
    `requires_nodes`, so a missing symlink surfaces as a named missing node class rather than as a
    render-time failure — and the `ckpt` enum does the same for an absent checkpoint, which is the
    other half of the install.
    """
    reg = registry.load_registry()
    spec = reg.get("matte_sam2")
    assert spec.kind == "custom", "postprocess would put it in the profile post-step picker"
    assert "VorchSAM2BoxMatte" in spec.requires_nodes
    # `LoadVideo` also emits an output descriptor (the file it read), so the matte must pin node 6
    # or vorch could resolve the step to the clip it was handed.
    assert spec.output_nodes == ["6"]

    installed = {
        "LoadVideo": {"input": {"required": {"file": [["clip.mp4"]]}}},
        "GetVideoComponents": {"input": {"required": {"video": ["VIDEO"]}}},
        "MaskToImage": {"input": {"required": {"mask": ["MASK"]}}},
        "CreateVideo": {"input": {"required": {"images": ["IMAGE"], "fps": ["FLOAT"]}}},
        "SaveVideo": {"input": {"required": {
            "video": ["VIDEO"], "filename_prefix": ["STRING"],
            "format": [["mp4"]], "codec": [["h264"]]}}},
        "VorchSAM2BoxMatte": {"input": {"required": {
            "images": ["IMAGE"], "ckpt_name": [["sam2.1_hiera_small.pt"]],
            "box_x": ["FLOAT"], "box_y": ["FLOAT"], "box_w": ["FLOAT"], "box_h": ["FLOAT"],
            "threshold": ["FLOAT"], "dilate": ["INT"], "feather": ["INT"],
            "offload": ["BOOLEAN"]}}},
    }
    assert reg.validate(spec, object_info=installed) == []

    no_symlink = {k: v for k, v in installed.items() if k != "VorchSAM2BoxMatte"}
    problems = reg.validate(spec, object_info=no_symlink)
    assert any("VorchSAM2BoxMatte" in p for p in problems), problems

    # The checkpoint half of the install. This works only because the node calls its input
    # `ckpt_name`, which is in `registry.MODEL_FIELDS`, so the template literal becomes a derived
    # model requirement. `enum_from` alone would NOT catch it: that only checks the class exists.
    assert ("3", "VorchSAM2BoxMatte", "ckpt_name", "sam2.1_hiera_small.pt") in spec.requires_models
    no_ckpt = dict(installed)
    no_ckpt["VorchSAM2BoxMatte"] = {"input": {"required": dict(
        installed["VorchSAM2BoxMatte"]["input"]["required"], ckpt_name=[[]])}}
    problems = reg.validate(spec, object_info=no_ckpt)
    assert any("sam2.1_hiera_small.pt" in p and "not installed" in p for p in problems), problems
