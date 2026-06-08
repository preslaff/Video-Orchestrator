"""GPU-free checks for the workflow registry — node surgery and validation.

Loads the real templates from workflows/ and confirms apply_params writes the
expected node fields and that Wan length validation rejects non-4n+1 values.
"""

import json

import pytest

from vorch import config, workflows


def test_registry_node_ids_exist_in_templates():
    """Every (node_id, field) in the registry must exist in its template."""
    for step, spec in workflows.REGISTRY.items():
        wf = workflows.load_template(step)
        assert spec["output_node"] in wf, f"{step}: missing output node"
        for name, (node_id, field) in spec["params"].items():
            assert node_id in wf, f"{step}.{name}: node {node_id} absent"
            assert field in wf[node_id]["inputs"], (
                f"{step}.{name}: field {field!r} absent on node {node_id}"
            )


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
    for step in workflows.REGISTRY:
        path = config.WORKFLOW_DIR / workflows.REGISTRY[step]["template"]
        data = json.load(open(path))
        assert "nodes" not in data, f"{step}: looks like a UI export, not API format"
        assert all("class_type" in node for node in data.values())
