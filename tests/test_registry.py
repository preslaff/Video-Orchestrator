"""GPU-free checks for the declarative workflow registry (vorch/registry.py)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from vorch import config, registry

BUILTINS = {"keyframe", "animate", "faceswap", "identity_edit", "kv_edit"}


@pytest.fixture
def reg():
    return registry.load_registry()


def test_builtin_yaml_targets_exist_and_are_literal(reg):
    assert BUILTINS <= set(reg.names())
    for spec in reg.list():
        assert registry.check_template(spec) == [], spec.name
        assert spec.builtin and spec.template.exists()


def test_kind_contracts_satisfied(reg):
    for spec in reg.list():
        assert registry._contract_problems(spec) == [], spec.name
    assert reg.get("identity_edit").image_inputs() == ["reference", "reference_2"]
    assert reg.get("animate").image_inputs() == ["start_image"]
    assert reg.get("animate").output_type == "video" and reg.get("animate").timeout == 1800


def test_fanout_writes_all_targets(reg):
    wf = reg.apply("identity_edit", reference="a.png", reference_2="b.png",
                   positive="p", seed=1, width=576, height=1024)
    assert wf["17"]["inputs"]["width"] == 576 and wf["19"]["inputs"]["width"] == 576
    assert wf["17"]["inputs"]["height"] == 1024 and wf["19"]["inputs"]["height"] == 1024
    wf = reg.apply("animate", start_image="x.png", positive="m", length=49, seed=7, steps=30, split_step=15)
    assert wf["9"]["inputs"]["noise_seed"] == 7 and wf["10"]["inputs"]["noise_seed"] == 7
    assert wf["9"]["inputs"]["steps"] == 30 and wf["10"]["inputs"]["steps"] == 30
    assert wf["9"]["inputs"]["end_at_step"] == 15 and wf["10"]["inputs"]["start_at_step"] == 15


def test_type_coercion_and_constraints(reg):
    wf = reg.apply("keyframe", positive="p", seed="42", width="1024", height=576, cfg="5")
    assert wf["75:73"]["inputs"]["noise_seed"] == 42 and wf["75:68"]["inputs"]["value"] == 1024
    assert wf["75:63"]["inputs"]["cfg"] == 5.0
    with pytest.raises(ValueError):
        reg.apply("keyframe", positive="p", seed=1, width=1000, height=576)  # not multiple of 16
    with pytest.raises(ValueError):
        reg.apply("keyframe", positive="p", seed=1, width=128, height=576)  # < min
    with pytest.raises(ValueError):
        reg.apply("keyframe", positive="p", seed="abc")
    with pytest.raises(ValueError):
        reg.apply("animate", start_image="x", positive="m", length=50)
    with pytest.raises(ValueError):
        reg.apply("animate", start_image="x", positive="m", length=49, batch_size=2)  # max 1


def test_unknown_optional_param_dropped_non_strict(reg, caplog):
    # kv_edit declares no width/height; the edit contract lists them as optional.
    wf = reg.apply("kv_edit", strict=False, reference="a", reference_2="b", positive="p",
                   seed=1, width=1024, height=576)
    assert wf["135"]["inputs"]["text"] == "p"
    assert "ignores param 'width'" in caplog.text
    with pytest.raises(KeyError):  # strict by default
        reg.apply("kv_edit", reference="a", reference_2="b", positive="p", seed=1, width=1024)


def test_unknown_required_param_raises(reg):
    with pytest.raises(KeyError):
        reg.apply("keyframe", strict=False, positive="p", seed=1, nonsense=1)


def _user_dir(tmp_path: Path) -> Path:
    d = tmp_path / "user_workflows"
    d.mkdir()
    return d


def test_user_dir_overrides_builtin_and_adds_new(tmp_path):
    user = _user_dir(tmp_path)
    # Override "faceswap" with a copy that exposes fewer params, plus a new postprocess entry.
    shutil.copy(config.WORKFLOW_DIR / "ReActor_Face_Swap.json", user / "my_swap.json")
    (user / "faceswap.yaml").write_text(yaml.safe_dump({
        "name": "faceswap", "kind": "faceswap", "template": "my_swap.json",
        "inputs": {"input_image": "image", "source_image": "image"},
        "outputs": {"type": "image", "nodes": ["4"]},
        "params": {"input_image": {"node": "1", "field": "image"},
                   "source_image": {"node": "2", "field": "image"}},
    }))
    (user / "restore_only.yaml").write_text(yaml.safe_dump({
        "kind": "postprocess", "template": "my_swap.json",
        "inputs": {"input_image": "image"},
        "outputs": {"type": "image", "nodes": ["4"]},
        "params": {"input_image": {"node": "1", "field": "image"},
                   "source_image": {"node": "2", "field": "image"}},
    }))
    reg = registry.load_registry([config.WORKFLOW_DIR, user], force=True)
    fs = reg.get("faceswap")
    assert not fs.builtin and fs.source == user / "faceswap.yaml"
    assert set(fs.params) == {"input_image", "source_image"}
    # The lean override still runs the pipeline's optional params away (non-strict).
    wf = reg.apply("faceswap", strict=False, input_image="a", source_image="b", restore_model="x.pth")
    assert wf["1"]["inputs"]["image"] == "a"
    rp = reg.get("restore_only")  # name defaults to the file stem
    assert rp.kind == "postprocess" and rp.image_inputs() == ["input_image"]
    assert reg.errors == {}


def test_user_dir_bad_entry_is_skipped_and_reported(tmp_path):
    user = _user_dir(tmp_path)
    (user / "broken.yaml").write_text(yaml.safe_dump({
        "kind": "keyframe", "template": "missing.json",
        "params": {"positive": {"node": "1", "field": "text"}},
    }))
    reg = registry.load_registry([config.WORKFLOW_DIR, user], force=True)
    assert "broken" not in reg
    assert "template not found" in reg.errors["broken"]
    assert BUILTINS <= set(reg.names())  # built-ins unaffected


def test_validate_reports_missing_node_link_target_and_contract(tmp_path, reg):
    wf = reg.load_template("keyframe")
    spec = registry.parse_spec({
        "name": "x", "kind": "keyframe", "template": str(reg.get("keyframe").template),
        "params": {
            "positive": {"node": "999", "field": "value"},          # missing node
            "seed": {"node": "75:73", "field": "nope"},              # missing field
            "width": {"node": "75:66", "field": "width"},            # link, not literal
        },
    })
    problems = registry.check_template(spec, wf) + registry._contract_problems(spec)
    joined = "\n".join(problems)
    assert "node '999' not in template" in joined
    assert "no input 'nope'" in joined
    assert "is a link, not a literal" in joined
    assert "requires param 'height'" in joined


def test_validate_online_checks_classes_and_model_files(reg):
    spec = reg.get("keyframe")
    good = {cls: {"input": {"required": {}}} for cls in spec.requires_nodes}
    good["UNETLoader"] = {"input": {"required": {"unet_name": [["flux-2-klein-4b-fp8.safetensors"], {}]}}}
    good["KSamplerSelect"] = {"input": {"required": {"sampler_name": [["euler"], {}]}}}
    assert reg.validate(spec, object_info=good) == []
    bad = dict(good)
    bad["UNETLoader"] = {"input": {"required": {"unet_name": [["other.safetensors"], {}]}}}
    problems = reg.validate(spec, object_info=bad)
    assert any("flux-2-klein-4b-fp8.safetensors" in p and "not installed" in p for p in problems)
    missing_cls = {k: v for k, v in good.items() if k != "VAELoader"}
    assert any("no node class 'VAELoader'" in p for p in reg.validate(spec, object_info=missing_cls))


def test_requirements_derived_from_loaders(reg):
    req = reg.requirements("animate")
    assert {"UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader", "WanImageToVideo"} <= set(req["nodes"])
    files = {m["value"] for m in req["models"]}
    assert "Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf" in files and "wan_2.1_vae.safetensors" in files
    assert "ReActorFaceSwap" in reg.requirements("faceswap")["nodes"]
    assert "inswapper_128.onnx" in {m["value"] for m in reg.requirements("faceswap")["models"]}


def test_spec_to_dict_is_json_serialisable(reg):
    d = reg.get("animate").to_dict()
    json.dumps(d)
    assert d["params"]["seed"]["targets"] == [{"node": "9", "field": "noise_seed"}, {"node": "10", "field": "noise_seed"}]
    assert d["outputs"] == {"type": "video", "nodes": ["13"]}


def test_run_step_selects_declared_output_node(monkeypatch, tmp_path):
    from vorch import steps
    from vorch.clients import comfy

    (tmp_path / "video").mkdir()
    descriptors = [
        {"filename": "preview_00001_.png", "subfolder": "", "node_id": "11"},
        {"filename": "Wan22_00001_.mp4", "subfolder": "video", "node_id": "13"},
    ]
    monkeypatch.setattr(comfy, "run", lambda wf, timeout=None, **kw: descriptors)
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path)
    paths = steps.run_step("animate", start_image="x.png", positive="m", length=49, seed=1)
    assert paths == [tmp_path / "video" / "Wan22_00001_.mp4"]
