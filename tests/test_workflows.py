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
