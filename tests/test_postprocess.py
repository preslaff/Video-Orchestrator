"""Postprocess chains: ordering, type matching, auto-fill, VRAM reclaim, orchestrator wiring."""

from __future__ import annotations

import shutil

import pytest
import yaml

from vorch import config, registry, vram
from vorch.clients import comfy
from vorch.genparams import PostStep
from vorch.steps import postprocess


@pytest.fixture
def user_registry(tmp_path):
    """Two fake postprocess workflows: a video one (fps+seed aware) and an image one."""
    user = tmp_path / "wf"
    user.mkdir()
    shutil.copy(config.WORKFLOW_DIR / "wan22_i2v_gguf.json", user / "vid.json")
    shutil.copy(config.WORKFLOW_DIR / "ReActor_Face_Swap.json", user / "img.json")
    (user / "interp.yaml").write_text(yaml.safe_dump({
        "kind": "postprocess", "template": "vid.json",
        "inputs": {"input_video": "video"},
        "outputs": {"type": "video", "nodes": ["13"]},
        "params": {"input_video": {"node": "5", "field": "image", "type": "str", "required": True},
                   "fps": {"node": "12", "field": "fps", "type": "int"},
                   "seed": {"node": "9", "field": "noise_seed", "type": "int"},
                   "multiplier": {"node": "8", "field": "length", "type": "int"},
                   "filename_prefix": {"node": "13", "field": "filename_prefix", "type": "str"}},
    }))
    (user / "upscale.yaml").write_text(yaml.safe_dump({
        "kind": "postprocess", "template": "img.json",
        "inputs": {"input_image": "image"},
        "outputs": {"type": "image", "nodes": ["4"]},
        "params": {"input_image": {"node": "1", "field": "image", "type": "str", "required": True}},
    }))
    reg = registry.load_registry([config.WORKFLOW_DIR, user], force=True)
    registry.install(reg)
    yield reg
    registry.reset_cache()


@pytest.fixture
def capture(monkeypatch, tmp_path):
    posted, reclaims = [], []
    out = tmp_path / "out"
    (out / "video").mkdir(parents=True)
    monkeypatch.setattr(config, "COMFY_OUTPUT", out)
    n = {"i": 0}

    def fake_run(wf, timeout=None, **kw):
        posted.append(wf)
        n["i"] += 1
        video = "13" in wf and wf["13"].get("class_type") == "SaveVideo"
        name = f"p{n['i']}.mp4" if video else f"p{n['i']}.png"
        sub = "video" if video else ""
        (out / sub / name).write_bytes(b"x")
        return [{"filename": name, "subfolder": sub, "node_id": "13" if video else "4"}]

    monkeypatch.setattr(comfy, "run", fake_run)
    monkeypatch.setattr(comfy, "upload_image", lambda p, **kw: f"staged:{p}")
    monkeypatch.setattr(vram, "reclaim", lambda *a, **k: reclaims.append(1))
    monkeypatch.setattr(vram, "log_state", lambda *a, **k: None)
    return posted, reclaims


def test_chain_order_autofill_and_reclaim(user_registry, capture, tmp_path):
    posted, reclaims = capture
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"v")
    chain = [PostStep("interp", {"multiplier": 97}), PostStep("interp", {"fps": 30}), PostStep("upscale")]
    outs = postprocess.run_chain(clip, chain, fps=24, seed=5, filename_prefix="vorch/abc/s00_post")
    assert len(outs) == 2 and outs[-1].suffix == ".mp4"          # upscale skipped: video, not image
    first, second = posted
    assert first["5"]["inputs"]["image"] == f"staged:{clip}"
    assert first["12"]["inputs"]["fps"] == 24 and first["9"]["inputs"]["noise_seed"] == 5
    assert first["8"]["inputs"]["length"] == 97
    assert first["13"]["inputs"]["filename_prefix"] == "vorch/abc/s00_post_interp"
    assert second["12"]["inputs"]["fps"] == 30                    # explicit step param wins
    assert second["5"]["inputs"]["image"] == f"staged:{outs[0]}"  # fed forward
    assert len(reclaims) == 1                                      # same workflow twice: one reclaim


def test_chain_image_and_type_mismatch(user_registry, capture, tmp_path):
    posted, _ = capture
    still = tmp_path / "s.png"
    still.write_bytes(b"i")
    outs = postprocess.run_chain(still, [PostStep("interp"), PostStep("upscale")])
    assert len(outs) == 1 and outs[0].suffix == ".png" and len(posted) == 1
    assert postprocess.run_chain(still, []) == []
    with pytest.raises(ValueError):
        postprocess.run_chain(still, [PostStep("keyframe")])


def test_orchestrator_sets_video_path_to_final(make_store, tmp_path, monkeypatch):
    from vorch import orchestrator as om
    from vorch import pipeline, profiles, storyboard
    from vorch.orchestrator import Orchestrator
    from vorch.state import JobStatus
    from vorch.storyboard import PromptPair, SceneBrief

    store = make_store()
    prof = profiles.apply_overrides(profiles.default_profile(), {
        "face_swap": {"enabled": False}, "evaluate": {"enabled": False},
        "post_steps": [{"workflow": "interp", "params": {"multiplier": 2}}],
    })
    monkeypatch.setattr(storyboard, "generate_storyboard",
                        lambda idea, *, n_scenes=None, **kw: [SceneBrief(scene=1, description="d")])
    monkeypatch.setattr(storyboard, "keyframe_prompt", lambda b, *, style="", **kw: PromptPair(positive="kf"))
    monkeypatch.setattr(storyboard, "animation_prompt", lambda b, **kw: PromptPair(positive="an"))
    prefixes = {}
    monkeypatch.setattr(pipeline, "_fresh_keyframe",
                        lambda *a, filename_prefix=None, **k: prefixes.update(kf=filename_prefix) or ("/kf.png", None))
    monkeypatch.setattr(pipeline, "animate_frame",
                        lambda *a, filename_prefix=None, **k: prefixes.update(clip=filename_prefix) or "/raw.mp4")
    seen = {}

    def fake_chain(path, chain, *, fps=None, seed=None, filename_prefix=None, **kw):
        seen.update(path=str(path), chain=[s.workflow for s in chain], fps=fps, prefix=filename_prefix)
        return ["/post1.mp4", "/final.mp4"]

    monkeypatch.setattr(om, "run_chain", fake_chain)
    o = Orchestrator(store, profile=prof, seed=3)
    job = o.create_job("idea", autonomy="full")
    o.run(job)
    s = job.scenes[0]
    assert job.status is JobStatus.DONE
    assert s.raw_video_path == "/raw.mp4" and s.video_path == "/final.mp4" and s.post_paths == ["/post1.mp4", "/final.mp4"]
    assert seen == {"path": "/raw.mp4", "chain": ["interp"], "fps": 16, "prefix": f"vorch/{job.id[:8]}/s00_post"}
    assert prefixes == {"kf": f"vorch/{job.id[:8]}/s00_keyframe", "clip": f"vorch/{job.id[:8]}/s00_clip"}
    assert any(d.phase == "post" for d in store.decisions(job.id))


def test_video_step_gets_the_master_face_on_its_backend(user_registry, capture, tmp_path, monkeypatch):
    """A `postprocess` step declaring `source_image` (video face swap) receives the
    job's master face, uploaded once per backend."""
    import json
    from pathlib import Path

    from vorch.clients import comfy

    (tmp_path / "wf" / "vswap.json").write_text(json.dumps({
        "1": {"class_type": "LoadVideo", "inputs": {"file": "clip.mp4"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "face.png"}},
        "13": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/sw", "video": ["1", 0]}},
    }))
    (tmp_path / "wf" / "vswap.yaml").write_text(yaml.safe_dump({
        "kind": "postprocess", "template": "vswap.json",
        "inputs": {"input_video": "video", "source_image": "image"},
        "outputs": {"type": "video", "nodes": ["13"]},
        "params": {"input_video": {"node": "1", "field": "file", "type": "str", "required": True},
                   "source_image": {"node": "2", "field": "image", "type": "str", "required": True}},
    }))
    reg = registry.load_registry([config.WORKFLOW_DIR, tmp_path / "wf"], force=True)
    registry.install(reg)
    uploads: list = []
    monkeypatch.setattr(comfy, "upload_image", lambda p, **kw: uploads.append((Path(p).name, (kw.get("backend") or type("L", (), {"name": "local"})).name)) or f"staged:{Path(p).name}")
    posted, _ = capture
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"v")
    master = tmp_path / "master.jpg"
    master.write_bytes(b"i")
    outs = postprocess.run_chain(clip, [PostStep("vswap"), PostStep("vswap")], master_face=master)
    assert len(outs) == 2
    assert uploads[0] == ("clip.mp4", "local") and ("master.jpg", "local") in uploads
    assert sum(1 for n, _ in uploads if n == "master.jpg") == 1          # staged once per backend
    assert posted[-1]["2"]["inputs"]["image"] == "staged:master.jpg"
    posted.clear()
    postprocess.run_chain(clip, [PostStep("vswap", {"source_image": "other.png"})], master_face=master)
    assert posted[-1]["2"]["inputs"]["image"] == "other.png"             # explicit param wins
    postprocess.run_chain(clip, [PostStep("vswap")])                      # no master: not filled
    assert posted[-1]["2"]["inputs"]["image"] == "face.png"


def test_generative_step_gets_the_scene_prompts(user_registry, capture, tmp_path):
    """A step declaring `positive`/`motion` receives the scene prompts unless set explicitly."""
    import json

    (tmp_path / "wf" / "gen.json").write_text(json.dumps({
        "1": {"class_type": "LoadVideo", "inputs": {"file": "clip.mp4"}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "default", "clip": ["1", 0]}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "default motion", "clip": ["1", 0]}},
        "13": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/g", "video": ["1", 0]}},
    }))
    (tmp_path / "wf" / "gen.yaml").write_text(yaml.safe_dump({
        "kind": "postprocess", "template": "gen.json",
        "inputs": {"input_video": "video"}, "outputs": {"type": "video", "nodes": ["13"]},
        "params": {"input_video": {"node": "1", "field": "file", "type": "str", "required": True},
                   "positive": {"node": "9", "field": "text", "type": "str"},
                   "motion": {"node": "11", "field": "text", "type": "str"}},
    }))
    reg = registry.load_registry([config.WORKFLOW_DIR, tmp_path / "wf"], force=True)
    registry.install(reg)
    posted, _ = capture
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"v")
    postprocess.run_chain(clip, [PostStep("gen")], prompts={"positive": "a woman on a rooftop", "motion": "she smiles"})
    assert posted[-1]["9"]["inputs"]["text"] == "a woman on a rooftop" and posted[-1]["11"]["inputs"]["text"] == "she smiles"
    postprocess.run_chain(clip, [PostStep("gen", {"positive": "fixed"})], prompts={"positive": "scene", "motion": None})
    assert posted[-1]["9"]["inputs"]["text"] == "fixed" and posted[-1]["11"]["inputs"]["text"] == "default motion"
