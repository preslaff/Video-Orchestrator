"""steps.keyframe only sends params the workflow declares.

Local samplers take literal width/height/steps/cfg; a hosted API node often has none
of them (size is an enum, there is no sampler) and some have no usable seed at all.
A param declared with an empty ``targets`` list is the seam for that last case: it
satisfies the kind contract and still type-checks, but writes nowhere.
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from vorch import config, provenance, registry
from vorch.clients import comfy
from vorch.steps.keyframe import keyframe


@pytest.fixture
def capture(monkeypatch, tmp_path):
    posted: list[dict] = []
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(config, "COMFY_OUTPUT", out_dir)
    n = {"i": 0}

    def fake_run(wf, timeout=None, **kw):
        posted.append(wf)
        n["i"] += 1
        name = f"k{n['i']}.png"
        (out_dir / name).write_bytes(b"x")
        return [{"filename": name, "subfolder": "", "node_id": "9"}]

    monkeypatch.setattr(comfy, "run", fake_run)
    return posted


def test_builtin_keyframe_still_sends_every_param(capture):
    p = keyframe("a cat on a wall", width=1024, height=576, steps=24, cfg=5, seed=42,
                 negative="blurry", batch_size=1, filename_prefix="vorch/aaa/s00_keyframe")
    wf = capture[-1]
    assert wf["76"]["inputs"]["value"] == "a cat on a wall"
    assert wf["75:73"]["inputs"]["noise_seed"] == 42
    assert wf["75:68"]["inputs"]["value"] == 1024 and wf["75:69"]["inputs"]["value"] == 576
    assert wf["75:62"]["inputs"]["steps"] == 24 and wf["75:63"]["inputs"]["cfg"] == 5
    assert wf["75:67"]["inputs"]["text"] == "blurry"
    assert provenance.read_manifest(p)["workflow"] == "keyframe"


def _lean_registry(tmp_path):
    """A keyframe workflow with no size/sampler params and a seed that goes nowhere."""
    user = tmp_path / "user_workflows"
    user.mkdir()
    shutil.copy(config.WORKFLOW_DIR / "image_flux2_klein_text_to_image.json", user / "lean.json")
    (user / "lean_api.yaml").write_text(yaml.safe_dump({
        "name": "lean_api", "kind": "keyframe", "template": "lean.json",
        "outputs": {"type": "image", "nodes": ["9"]},
        "params": {
            "positive": {"node": "76", "field": "value", "type": "str"},
            # the node has no seed input of its own — declared so the contract is met, wired nowhere
            "seed": {"targets": [], "type": "int"},
            "width": {"targets": [], "type": "int"},
            "height": {"targets": [], "type": "int"},
        },
    }))
    return registry.load_registry([config.WORKFLOW_DIR, user], force=True)


def test_undeclared_size_and_sampler_params_are_not_sent(capture, tmp_path, monkeypatch):
    reg = _lean_registry(tmp_path)
    monkeypatch.setattr(registry, "load_registry", lambda *a, **k: reg)
    before = reg.load_template("lean_api")

    keyframe("a cat", workflow="lean_api", width=1024, height=576, steps=24, cfg=5, seed=42,
             negative="blurry", filename_prefix="vorch/aaa/s00_keyframe")

    wf = capture[-1]
    assert wf["76"]["inputs"]["value"] == "a cat"
    # steps/cfg/negative/filename_prefix were never declared: not sent, so not written and no KeyError
    assert wf["75:62"]["inputs"]["steps"] == before["75:62"]["inputs"]["steps"]
    assert wf["75:63"]["inputs"]["cfg"] == before["75:63"]["inputs"]["cfg"]
    assert wf["75:67"]["inputs"]["text"] == before["75:67"]["inputs"]["text"]
    assert wf["9"]["inputs"]["filename_prefix"] == before["9"]["inputs"]["filename_prefix"]
    # seed/width/height were declared but target nothing: the graph keeps its own values
    assert wf["75:73"]["inputs"]["noise_seed"] == before["75:73"]["inputs"]["noise_seed"]
    assert wf["75:68"]["inputs"]["value"] == before["75:68"]["inputs"]["value"]


def test_targetless_param_satisfies_the_contract_and_still_coerces(tmp_path):
    reg = _lean_registry(tmp_path)
    spec = reg.get("lean_api")
    assert registry._contract_problems(spec) == []
    assert registry.check_template(spec) == []
    assert spec.accepts("seed") and not spec.accepts("steps")
    # declared-but-unwired params are still coerced, so a bad value is still caught
    assert reg.apply("lean_api", strict=False, positive="x", seed="42", width=1024, height=576)
    with pytest.raises(ValueError):
        reg.apply("lean_api", strict=False, positive="x", seed="not-a-number", width=1, height=1)
