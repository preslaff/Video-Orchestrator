"""Scaffolder heuristics (vorch/registry_tools.py) against the five shipped templates."""

from __future__ import annotations

import pytest
import yaml

from vorch import __main__ as cli
from vorch import config, registry, registry_tools

TEMPLATES = {
    "keyframe": "image_flux2_klein_text_to_image.json",
    "animate": "wan22_i2v_gguf.json",
    "faceswap": "ReActor_Face_Swap.json",
    "edit": "klein_identity_edit.json",
    "edit2": "image_flux2_klein_9b_kv_image_edit.json",
}


def _wf(name):
    return registry.load_template_file(config.WORKFLOW_DIR / TEMPLATES[name])


@pytest.mark.parametrize("tpl,kind", [
    ("keyframe", "keyframe"), ("animate", "animate"), ("faceswap", "faceswap"),
    ("edit", "edit"), ("edit2", "edit"),
])
def test_guess_kind(tpl, kind):
    assert registry_tools.guess_kind(_wf(tpl)) == kind


def _targets(doc, pname):
    p = doc["params"][pname]
    if "targets" in p:
        return {(t["node"], t["field"]) for t in p["targets"]}
    return {(p["node"], p["field"])}


def test_guess_spec_keyframe():
    doc, notes = registry_tools.guess_spec(_wf("keyframe"), name="kf", template="x.json")
    assert doc["kind"] == "keyframe"
    assert _targets(doc, "positive") == {("76", "value")}
    assert _targets(doc, "negative") == {("75:67", "text")}
    assert _targets(doc, "seed") == {("75:73", "noise_seed")}
    assert _targets(doc, "width") == {("75:68", "value")} and _targets(doc, "height") == {("75:69", "value")}
    assert _targets(doc, "steps") == {("75:62", "steps")} and _targets(doc, "cfg") == {("75:63", "cfg")}
    assert doc["params"]["unet"]["enum_from"] == {"class_type": "UNETLoader", "input": "unet_name"}
    assert doc["outputs"] == {"type": "image", "nodes": ["9"]}
    assert doc["params"]["positive"]["required"] is True


def test_guess_spec_animate():
    doc, _ = registry_tools.guess_spec(_wf("animate"), name="wan", template="x.json")
    assert doc["inputs"] == {"start_image": "image"}
    assert _targets(doc, "start_image") == {("5", "image")}
    assert _targets(doc, "positive") == {("6", "text")} and _targets(doc, "negative") == {("7", "text")}
    assert _targets(doc, "seed") == {("9", "noise_seed"), ("10", "noise_seed")}
    assert _targets(doc, "steps") == {("9", "steps"), ("10", "steps")}
    assert _targets(doc, "split_step") == {("9", "end_at_step"), ("10", "start_at_step")}
    assert _targets(doc, "length") == {("8", "length")} and doc["params"]["length"]["constraint"] == "4n+1"
    assert _targets(doc, "width") == {("8", "width")}
    assert _targets(doc, "fps") == {("12", "fps")}
    assert doc["outputs"] == {"type": "video", "nodes": ["13"]}
    assert _targets(doc, "unet") == {("1", "unet_name")} and _targets(doc, "unet_2") == {("2", "unet_name")}


def test_guess_spec_faceswap():
    doc, _ = registry_tools.guess_spec(_wf("faceswap"), name="fs", template="x.json")
    assert doc["inputs"] == {"input_image": "image", "source_image": "image"}
    assert _targets(doc, "input_image") == {("1", "image")} and _targets(doc, "source_image") == {("2", "image")}
    assert _targets(doc, "restore_model") == {("3", "face_restore_model")}
    assert _targets(doc, "swap_model") == {("3", "swap_model")}
    assert doc["outputs"] == {"type": "image", "nodes": ["4"]}


def test_guess_spec_edits():
    doc, _ = registry_tools.guess_spec(_wf("edit"), name="ie", template="x.json")
    assert doc["inputs"] == {"reference": "image", "reference_2": "image"}
    assert _targets(doc, "positive") == {("10", "text")} and _targets(doc, "negative") == {("13", "text")}
    assert _targets(doc, "seed") == {("18", "noise_seed")}
    assert _targets(doc, "width") == {("17", "width"), ("19", "width")}
    assert _targets(doc, "steps") == {("17", "steps")} and _targets(doc, "cfg") == {("15", "cfg")}
    assert doc["outputs"]["nodes"] == ["22"]
    doc, notes = registry_tools.guess_spec(_wf("edit2"), name="kv", template="x.json")
    assert _targets(doc, "positive") == {("135", "text")} and _targets(doc, "seed") == {("125", "noise_seed")}
    assert _targets(doc, "steps") == {("137", "steps")} and _targets(doc, "cfg") == {("138", "cfg")}
    assert "width" not in doc["params"]  # size is derived from the reference in this graph
    assert any("megapixels" in n for n in notes)  # surfaced for the human


def test_inspect_lists_candidates():
    rows = {r.node: r for r in registry_tools.inspect_template(_wf("keyframe"))}
    assert rows["76"].candidates["value"] == "positive"
    assert rows["75:67"].candidates["text"] == "negative"
    assert rows["75:68"].candidates["value"] == "width"
    assert rows["75:70"].candidates["unet_name"] == "loader:unet_name"
    assert rows["9"].candidates["__output__"] == "output:image"
    text = registry_tools.format_inspect(list(rows.values()))
    assert "<- positive" in text and "-> output:image" in text


@pytest.mark.parametrize("tpl", ["keyframe", "animate", "faceswap", "edit"])
def test_add_writes_yaml_that_validates(tmp_path, tpl, capsys):
    out = tmp_path / "user"
    src = config.WORKFLOW_DIR / TEMPLATES[tpl]
    assert cli.main(["registry", "add", str(src), "--name", f"my_{tpl}", "--out", str(out)]) == 0
    doc = yaml.safe_load((out / f"my_{tpl}.yaml").read_text().split("\n# Unmapped")[0])
    assert doc["name"] == f"my_{tpl}"
    reg = registry.load_registry([config.WORKFLOW_DIR, out], force=True)
    spec = reg.get(f"my_{tpl}")
    assert not spec.builtin
    assert reg.validate(spec) == []
    # Refuses to overwrite without --force.
    assert cli.main(["registry", "add", str(src), "--name", f"my_{tpl}", "--out", str(out)]) == 1


def test_cli_registry_list_and_validate(capsys):
    assert cli.main(["registry", "list"]) == 0
    out = capsys.readouterr().out
    assert "keyframe" in out and "kv_edit" in out and "[builtin]" in out
    assert cli.main(["registry", "validate"]) == 0
    assert cli.main(["registry", "validate", "animate"]) == 0
    assert "animate          ok" in capsys.readouterr().out
