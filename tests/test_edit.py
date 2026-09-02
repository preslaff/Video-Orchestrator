"""Reference-locked keyframes: steps.edit, stage_references, edit mode end to end."""

from __future__ import annotations

import pytest

from vorch import config, pipeline, provenance, vram
from vorch.clients import comfy
from vorch.genparams import DEFAULT, GenParams
from vorch.steps.edit import edit


@pytest.fixture
def capture(monkeypatch, tmp_path):
    posted: list[dict] = []
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(config, "COMFY_OUTPUT", out_dir)
    n = {"i": 0}

    def fake_run(wf, timeout=None):
        posted.append(wf)
        n["i"] += 1
        name = f"e{n['i']}.png"
        (out_dir / name).write_bytes(b"x")
        return [{"filename": name, "subfolder": "", "node_id": "22"}]

    monkeypatch.setattr(comfy, "run", fake_run)
    monkeypatch.setattr(comfy, "upload_image", lambda p: f"staged:{p}")
    monkeypatch.setattr(pipeline, "fit_image", lambda src, w, h, dest=None: f"{src}@{w}x{h}")
    monkeypatch.setattr(vram, "reclaim", lambda *a, **k: None)
    return posted


def test_edit_maps_references_to_inputs_and_repeats_last(capture):
    p = edit("same man, now smiling", ["a.png"], seed=5, width=576, height=1024, negative="blurry")
    wf = capture[-1]
    assert wf["4"]["inputs"]["image"] == "a.png" and wf["5"]["inputs"]["image"] == "a.png"  # repeated
    assert wf["10"]["inputs"]["text"] == "same man, now smiling" and wf["13"]["inputs"]["text"] == "blurry"
    assert wf["18"]["inputs"]["noise_seed"] == 5
    assert wf["17"]["inputs"]["width"] == 576 and wf["19"]["inputs"]["width"] == 576
    assert provenance.read_manifest(p)["workflow"] == "identity_edit"
    edit("x", ["a.png", "b.png"], seed=1)
    assert capture[-1]["5"]["inputs"]["image"] == "b.png"


def test_kv_edit_drops_size_params_and_rejects_non_edit(capture):
    edit("x", ["a.png", "b.png"], workflow="kv_edit", seed=2, width=1024, height=576, steps=6)
    wf = capture[-1]
    assert wf["76"]["inputs"]["image"] == "a.png" and wf["81"]["inputs"]["image"] == "b.png"
    assert wf["137"]["inputs"]["steps"] == 6
    with pytest.raises(ValueError):
        edit("x", ["a.png"], workflow="keyframe")
    with pytest.raises(ValueError):
        edit("x", [])


def test_stage_references_fits_uploads_and_caches(capture, tmp_path):
    a = tmp_path / "ref_a.jpg"
    a.write_bytes(b"j")
    cache: dict = {}
    p = GenParams.for_aspect("9:16")
    names = pipeline.stage_references([a, str(a)], p, cache=cache)
    assert names == [f"staged:{a}@576x1024"] * 2 and len(cache) == 1
    assert pipeline.stage_references([a], p, fit=False) == [f"staged:{a}"]
    with pytest.raises(FileNotFoundError):
        pipeline.stage_references([tmp_path / "nope.png"], p)


def test_fresh_keyframe_edit_mode_uses_edit_workflow(capture):
    p = DEFAULT.with_(keyframe_negative="smooth")
    path, _ = pipeline._fresh_keyframe("p", seed=9, evaluate=False, max_attempts=1, params=p,
                                       mode="edit", references=["r1.png"])
    wf = capture[-1]
    assert "4" in wf and wf["4"]["inputs"]["image"] == "r1.png"  # identity_edit graph
    assert wf["13"]["inputs"]["text"] == "smooth"
    with pytest.raises(ValueError):
        pipeline._fresh_keyframe("p", seed=1, evaluate=False, max_attempts=1, mode="edit")
    with pytest.raises(ValueError):
        pipeline._fresh_keyframe("p", seed=1, evaluate=False, max_attempts=1, mode="weird")


def test_run_sequence_edit_scene(capture, tmp_path, monkeypatch):
    ref = tmp_path / "r.png"
    ref.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "last_frame", lambda v, dest=None: f"{v}_last.png")
    monkeypatch.setattr(vram, "log_state", lambda *a, **k: None)
    # animate needs a SaveVideo-less fake: reuse capture (returns png), fine for wiring checks
    scenes = [pipeline.Scene("k", "a", mode="edit", references=(str(ref),))]
    pipeline.run_sequence(scenes, face_swap=False, seed=3, params=DEFAULT)
    kf_wf = capture[0]
    assert kf_wf["4"]["inputs"]["image"] == f"staged:{ref}@1024x576"


def test_orchestrator_edit_job_copies_refs_to_scenes(make_store, tmp_path, monkeypatch):
    from vorch import storyboard
    from vorch.orchestrator import Orchestrator
    from vorch.storyboard import PromptPair, SceneBrief

    ref = tmp_path / "r.png"
    ref.write_bytes(b"x")
    store = make_store()
    o = Orchestrator(store, face_swap=False, evaluate=False)
    with pytest.raises(ValueError):
        o.create_job("idea", mode="edit")  # no refs
    with pytest.raises(FileNotFoundError):
        o.create_job("idea", mode="edit", reference_images=[str(tmp_path / "missing.png")])
    job = o.create_job("idea", mode="edit", reference_images=[str(ref)], autonomy="full")
    assert store.load_job(job.id).reference_images == [str(ref)]

    monkeypatch.setattr(storyboard, "generate_storyboard",
                        lambda idea, *, n_scenes=None, **kw: [SceneBrief(scene=1, description="d")])
    monkeypatch.setattr(storyboard, "keyframe_prompt", lambda b, *, style="", **kw: PromptPair(positive="kf"))
    monkeypatch.setattr(storyboard, "animation_prompt", lambda b, **kw: PromptPair(positive="an"))
    seen = {}

    def fake_fresh(prompt, *, seed, evaluate, max_attempts, params, negative, mode, references, **kw):
        seen.update(mode=mode, references=tuple(references))
        return "/kf.png", None

    monkeypatch.setattr(pipeline, "_fresh_keyframe", fake_fresh)
    monkeypatch.setattr(pipeline, "stage_references", lambda paths, params, cache=None: ["staged"] * len(paths))
    monkeypatch.setattr(pipeline, "animate_frame", lambda *a, **k: "/v.mp4")
    o.run(job)
    assert seen == {"mode": "edit", "references": ("staged",)}
    assert store.load_job(job.id).scenes[0].mode == "edit"
