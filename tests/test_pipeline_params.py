"""GenParams threading: what actually reaches the ComfyUI workflow dicts."""

from __future__ import annotations

import json
import shutil

import pytest
import yaml

from vorch import config, pipeline, provenance, registry, vram
from vorch.clients import comfy
from vorch.genparams import DEFAULT, GenParams


@pytest.fixture
def capture(monkeypatch, tmp_path):
    """Fake comfy.run that records the posted workflow and 'produces' a file."""
    posted: list[dict] = []
    out_dir = tmp_path / "out"
    (out_dir / "video").mkdir(parents=True)
    monkeypatch.setattr(config, "COMFY_OUTPUT", out_dir)
    counter = {"n": 0}

    def fake_run(wf, timeout=None, **kw):
        posted.append({"wf": wf, "timeout": timeout})
        counter["n"] += 1
        is_video = any(n.get("class_type") == "SaveVideo" for n in wf.values())
        name = f"o{counter['n']}.mp4" if is_video else f"o{counter['n']}.png"
        sub = "video" if is_video else ""
        (out_dir / sub / name).write_bytes(b"x")
        return [{"filename": name, "subfolder": sub, "node_id": "13" if is_video else "9"}]

    monkeypatch.setattr(comfy, "run", fake_run)
    monkeypatch.setattr(comfy, "upload_image", lambda p, **kw: str(p))
    monkeypatch.setattr(vram, "reclaim", lambda *a, **k: None)
    monkeypatch.setattr(vram, "log_state", lambda *a, **k: None)
    return posted


def test_keyframe_threads_size_and_negative(capture, tmp_path):
    p = GenParams.for_aspect("9:16", keyframe_negative="smooth, airbrushed", steps=20, cfg=4.0)
    path, verdict = pipeline._fresh_keyframe("a cat", seed=7, evaluate=False, max_attempts=1, params=p)
    wf = capture[-1]["wf"]
    assert wf["75:68"]["inputs"]["value"] == 576 and wf["75:69"]["inputs"]["value"] == 1024
    assert wf["75:67"]["inputs"]["text"] == "smooth, airbrushed"
    assert wf["75:62"]["inputs"]["steps"] == 20 and wf["75:63"]["inputs"]["cfg"] == 4.0
    assert wf["75:73"]["inputs"]["noise_seed"] == 7
    m = provenance.read_manifest(path)
    assert m["workflow"] == "keyframe" and m["params"]["width"] == 576


def test_animate_single_seed_fanout_length_and_timeout(capture):
    p = GenParams.for_aspect("1:1", length=81, fps=24, animate_negative="warped face")
    pipeline.animate_frame("frame.png", "slow pan", seed=11, params=p)
    posted = capture[-1]
    wf = posted["wf"]
    assert wf["8"]["inputs"]["width"] == 640 and wf["8"]["inputs"]["height"] == 640
    assert wf["8"]["inputs"]["length"] == 81 and wf["12"]["inputs"]["fps"] == 24
    assert wf["9"]["inputs"]["noise_seed"] == 11 and wf["10"]["inputs"]["noise_seed"] == 11
    assert wf["7"]["inputs"]["text"] == "warped face"
    assert posted["timeout"] == int(1800 * 81 / 49)  # spec timeout scaled by length


def test_scene_negative_overrides_params(capture):
    p = GenParams(animate_negative="from-profile")
    pipeline.animate_frame("f.png", "m", seed=1, params=p, negative="from-scene")
    assert capture[-1]["wf"]["7"]["inputs"]["text"] == "from-scene"


def test_workflow_override_uses_named_entry(capture, tmp_path):
    user = tmp_path / "user_wf"
    user.mkdir()
    shutil.copy(config.WORKFLOW_DIR / "image_flux2_klein_text_to_image.json", user / "klein_alt.json")
    (user / "klein_alt.yaml").write_text(yaml.safe_dump({
        "kind": "keyframe", "template": "klein_alt.json",
        "outputs": {"type": "image", "nodes": ["9"]},
        "params": {
            "positive": {"node": "76", "field": "value", "type": "str", "required": True},
            "seed": {"node": "75:73", "field": "noise_seed", "type": "int"},
            "width": {"node": "75:68", "field": "value", "type": "int"},
            "height": {"node": "75:69", "field": "value", "type": "int"},
        },  # no steps/cfg/negative: those optional params are dropped, not errors
    }))
    registry.load_registry([config.WORKFLOW_DIR, user], force=True)
    try:
        # Point the default lookup at the same dirs for this test.
        registry.install(registry.load_registry([config.WORKFLOW_DIR, user]))
        p = DEFAULT.with_(keyframe_workflow="klein_alt", keyframe_negative="x")
        path, _ = pipeline._fresh_keyframe("p", seed=3, evaluate=False, max_attempts=1, params=p)
        wf = capture[-1]["wf"]
        assert wf["76"]["inputs"]["value"] == "p" and wf["75:73"]["inputs"]["noise_seed"] == 3
        assert wf["75:62"]["inputs"]["steps"] == 20  # template default kept (steps not declared)
        assert provenance.read_manifest(path)["workflow"] == "klein_alt"
    finally:
        registry.reset_cache()


def test_run_scene_and_sequence_accept_params(capture, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "last_frame", lambda v, dest=None: f"{v}_last.png")
    p = GenParams.for_aspect("9:16")
    r = pipeline.run_scene("kf", "an", face_swap=False, seed=5, params=p)
    kf_wf, an_wf = capture[-2]["wf"], capture[-1]["wf"]
    assert kf_wf["75:68"]["inputs"]["value"] == 576 and an_wf["8"]["inputs"]["width"] == 480
    assert r.video.suffix == ".mp4"
    scenes = [pipeline.Scene("k1", "a1", keyframe_negative="neg1"),
              pipeline.Scene("k2", "a2", continuous=True, animation_negative="neg2")]
    res = pipeline.run_sequence(scenes, face_swap=False, seed=5, params=p)
    assert len(res) == 2 and res[1].continued_from_previous
    wfs = [c["wf"] for c in capture]
    assert any(w.get("75:67", {}).get("inputs", {}).get("text") == "neg1" for w in wfs)
    assert wfs[-1]["7"]["inputs"]["text"] == "neg2" and wfs[-1]["8"]["inputs"]["height"] == 832
