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
    registry._cache[tuple(str(p) for p, _ in registry._default_dirs())] = reg
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

    def fake_chain(path, chain, *, fps=None, seed=None, filename_prefix=None):
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
