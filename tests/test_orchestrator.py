"""GPU-free checks for orchestrator.py — the phased job machine + gates.

A real Store (temp SQLite) verifies persistence, the decision log, and gate
resume; the Gemma/ComfyUI-bound calls (storyboard, keyframe eval, swap, animate)
are monkeypatched. Asserts which phase runs when, that semi-auto gates pause the
job, that approvals resume it, and that a reloaded job continues correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vorch import orchestrator as orch_mod
from vorch import pipeline, storyboard
from vorch.clients import comfy
from vorch.evaluator import Verdict
from vorch.orchestrator import GATE_KEYFRAMES, GATE_STORYBOARD, Orchestrator
from vorch.state import JobStatus, SceneStatus, Store
from vorch.storyboard import PromptPair, SceneBrief


def _verdict(score=8):
    return Verdict(
        action_correct=True, identity_match=True, style_match=True, artifacts=False,
        composition_ok=True, score=score, verdict="accept", fix_suggestion="",
    )


def _verdict_retry():
    return Verdict(
        action_correct=False, identity_match=True, style_match=True, artifacts=True,
        composition_ok=True, score=2, verdict="retry", fix_suggestion="use a peel",
    )


@pytest.fixture
def wired(monkeypatch, tmp_path, make_store):
    """Store + monkeypatched generation seams; returns (store, master_path, calls)."""
    calls = {"keyframe": 0, "swap": 0, "animate": 0}

    briefs = [
        SceneBrief(scene=1, description="d1", character="c", setting="s", mood="m"),
        SceneBrief(scene=2, description="d2", character="c", setting="s", mood="m"),
    ]
    monkeypatch.setattr(storyboard, "generate_storyboard", lambda idea, *, n_scenes=None, **kw: briefs)
    monkeypatch.setattr(
        storyboard, "keyframe_prompt", lambda b, *, style="", **kw: PromptPair(positive=f"kf::{b.description}")
    )
    monkeypatch.setattr(
        storyboard, "animation_prompt", lambda b, **kw: PromptPair(positive=f"an::{b.description}")
    )

    def fake_kf_eval(prompt, task, *, seed, max_attempts, on_attempt=None, **kw):
        calls["keyframe"] += 1
        path, v = f"/kf/{calls['keyframe']}.png", _verdict()
        if on_attempt:
            on_attempt(1, path, v)
        return path, v

    def fake_kf(prompt, *, seed, **kw):
        calls["keyframe"] += 1
        return f"/kf/{calls['keyframe']}.png"

    def fake_swap(frame, master_name, **kw):
        calls["swap"] += 1
        return f"/swap/{calls['swap']}.png"

    def fake_animate(frame, prompt, *, seed, **kw):
        calls["animate"] += 1
        return f"/video/{calls['animate']}.mp4"

    monkeypatch.setattr(pipeline, "generate_keyframe_evaluated", fake_kf_eval)
    monkeypatch.setattr(pipeline, "keyframe", fake_kf)
    monkeypatch.setattr(pipeline, "swap_master_onto", fake_swap)
    monkeypatch.setattr(pipeline, "animate_frame", fake_animate)
    monkeypatch.setattr(comfy, "upload_image", lambda p, **kw: str(p))

    master = tmp_path / "master.jpg"
    master.write_bytes(b"\xff\xd8\xff")

    store = make_store()
    yield store, master, calls


def test_create_job_validates_master(wired):
    store, _master, _ = wired
    o = Orchestrator(store)
    with pytest.raises(FileNotFoundError):
        o.create_job("idea", master_face="/no/such/face.jpg")


def test_semi_pauses_at_storyboard_gate(wired):
    store, master, calls = wired
    o = Orchestrator(store, n_scenes=2)
    job = o.create_job("an idea", style="oil", master_face=str(master), autonomy="semi")

    o.run(job)
    assert job.status is JobStatus.NEEDS_APPROVAL
    assert len(job.scenes) == 2
    assert job.scenes[0].keyframe_prompt == "kf::d1"
    assert job.scenes[0].animation_prompt == "an::d1"
    assert calls["keyframe"] == 0  # gate stopped before keyframes
    # The storyboard generation is logged.
    actions = [(d.phase, d.action) for d in store.decisions(job.id)]
    assert ("storyboard", "generated") in actions


def test_approve_storyboard_then_pauses_before_wan(wired):
    store, master, calls = wired
    o = Orchestrator(store, n_scenes=2)
    job = o.create_job("idea", master_face=str(master), autonomy="semi")
    o.run(job)

    o.approve(job, GATE_STORYBOARD)
    assert job.status is JobStatus.NEEDS_APPROVAL  # now paused before the Wan phase
    assert calls["keyframe"] == 2  # both keyframes generated + evaluated
    assert calls["animate"] == 0  # not yet animated
    assert all(s.keyframe_path is not None for s in job.scenes)
    assert all(s.status is SceneStatus.KEYFRAME for s in job.scenes)
    # Each keyframe verdict is recorded in the decision log.
    kf_decisions = [d for d in store.decisions(job.id) if d.phase == "keyframe"]
    assert len(kf_decisions) == 2 and all(d.action == "accept" for d in kf_decisions)


def test_full_run_to_completion(wired):
    store, master, calls = wired
    o = Orchestrator(store, n_scenes=2)
    job = o.create_job("idea", master_face=str(master), autonomy="semi")
    o.run(job)
    o.approve(job, GATE_STORYBOARD)
    o.approve(job, GATE_KEYFRAMES)

    assert job.status is JobStatus.DONE
    assert calls == {"keyframe": 2, "swap": 2, "animate": 2}
    assert all(s.status is SceneStatus.COMPLETE for s in job.scenes)
    assert all(s.video_path and s.swapped_path for s in job.scenes)
    video_decisions = [d for d in store.decisions(job.id) if d.phase == "video"]
    assert len(video_decisions) == 2


def test_face_swap_off_skips_swap_and_animates_keyframe(wired):
    store, master, calls = wired
    o = Orchestrator(store, n_scenes=2, evaluate=False, face_swap=False)
    job = o.create_job("idea", master_face="", autonomy="full")
    o.run(job)

    assert job.status is JobStatus.DONE
    assert calls["swap"] == 0           # no ReActor swap ran
    assert calls["animate"] == 2        # both keyframes animated directly
    assert all(s.swapped_path is None for s in job.scenes)
    assert all(s.video_path for s in job.scenes)


def test_create_job_allows_empty_master_when_face_swap_off(wired):
    store, _master, _ = wired
    o = Orchestrator(store, face_swap=False)
    job = o.create_job("idea")  # no master_face — must not raise
    assert job.face_swap is False and job.master_face == ""


def test_full_autonomy_runs_straight_through(wired):
    store, master, calls = wired
    o = Orchestrator(store, n_scenes=2)
    job = o.create_job("idea", master_face=str(master), autonomy="full")

    o.run(job)  # no gates
    assert job.status is JobStatus.DONE
    assert calls == {"keyframe": 2, "swap": 2, "animate": 2}


def test_evaluate_false_skips_verdicts(wired):
    store, master, calls = wired
    o = Orchestrator(store, n_scenes=2, evaluate=False)
    job = o.create_job("idea", master_face=str(master), autonomy="full")

    o.run(job)
    assert job.status is JobStatus.DONE
    assert all(s.keyframe_verdict is None for s in job.scenes)
    kf_decisions = [d for d in store.decisions(job.id) if d.phase == "keyframe"]
    assert all(d.action == "generated" for d in kf_decisions)


def test_low_score_accept_logs_review_flag(wired, monkeypatch):
    """An objectively-accepted keyframe with a low score gets a 'review' flag."""
    store, master, _ = wired

    def low_score(prompt, task, *, seed, max_attempts, on_attempt=None, **kw):
        path, v = "/kf/x.png", _verdict(score=1)  # objective accept, advisory score low
        if on_attempt:
            on_attempt(1, path, v)
        return path, v

    monkeypatch.setattr(pipeline, "generate_keyframe_evaluated", low_score)
    o = Orchestrator(store, n_scenes=2)
    job = o.create_job("idea", master_face=str(master), autonomy="full")
    o.run(job)

    actions = [d.action for d in store.decisions(job.id) if d.phase == "keyframe"]
    assert actions.count("accept") == 2  # both keyframes accepted (objective gate)
    assert actions.count("review") == 2  # both flagged for review (score=1)


def test_run_config_persisted_and_restored_on_resume(wired):
    """create_job stores seed/max_attempts/evaluate; from_job restores them."""
    store, master, _ = wired
    o = Orchestrator(store, n_scenes=2, seed=11, max_attempts=4, evaluate=False)
    job = o.create_job("idea", master_face=str(master), autonomy="semi")

    reloaded = store.load_job(job.id)
    assert reloaded.seed == 11 and reloaded.max_attempts == 4 and reloaded.evaluate is False

    o2 = Orchestrator.from_job(store, reloaded)
    assert o2.seed == 11 and o2.max_attempts == 4 and o2.evaluate is False and o2.n_scenes == 2


def test_every_keyframe_attempt_is_logged(wired, monkeypatch):
    """A retry-then-accept keyframe logs both attempts to the decision log."""
    store, master, _ = wired

    def retry_then_accept(prompt, task, *, seed, max_attempts, on_attempt=None, **kw):
        on_attempt(1, "/kf/a.png", _verdict_retry())
        on_attempt(2, "/kf/b.png", _verdict(score=8))
        return "/kf/b.png", _verdict(score=8)

    monkeypatch.setattr(pipeline, "generate_keyframe_evaluated", retry_then_accept)
    o = Orchestrator(store, n_scenes=1)
    job = o.create_job("idea", master_face=str(master), autonomy="full")
    o.run(job)

    kf = [(d.attempt, d.action) for d in store.decisions(job.id) if d.phase == "keyframe"]
    assert (1, "retry") in kf and (2, "accept") in kf


def test_resume_from_reloaded_job(wired):
    """Persist at the storyboard gate, reload, and finish with a fresh Orchestrator."""
    store, master, calls = wired
    o1 = Orchestrator(store, n_scenes=2)
    job = o1.create_job("idea", master_face=str(master), autonomy="semi")
    o1.run(job)
    job_id = job.id

    # Fresh orchestrator + reloaded job — nothing carried in memory.
    o2 = Orchestrator(store, n_scenes=2)
    reloaded = store.load_job(job_id)
    assert reloaded.status is JobStatus.NEEDS_APPROVAL
    o2.approve(reloaded, GATE_STORYBOARD)
    assert reloaded.status is JobStatus.NEEDS_APPROVAL  # keyframes gate

    reloaded2 = store.load_job(job_id)
    o2.approve(reloaded2, GATE_KEYFRAMES)
    assert reloaded2.status is JobStatus.DONE
    assert all(s.video_path for s in reloaded2.scenes)


# --- profile-driven plan: snapshot, legacy resume, stills + variants ---------------


def test_profile_snapshot_persisted_and_restored(wired):
    from vorch import profiles

    store, master, _ = wired
    prof = profiles.builtin_profiles()["social_9x16"]
    o = Orchestrator(store, profile=prof, seed=7)
    job = o.create_job("vertical idea", master_face=str(master))
    saved = store.load_job(job.id)
    assert saved.profile_id == "social_9x16" and saved.kind == "video"
    assert saved.profile["keyframe"]["width"] == 576
    resumed = Orchestrator.from_job(store, saved)
    assert resumed.params.aspect == "9:16" and resumed.params.wan_size == (480, 832)
    assert resumed.params.profile_name == "social_9x16"
    assert resumed.gates(saved) == ["storyboard", "keyframes"]


def test_legacy_job_without_snapshot_resumes(wired):
    store, master, _ = wired
    job = Orchestrator(store, evaluate=False, face_swap=False, max_attempts=3).create_job("old")
    assert store.load_job(job.id).profile is None
    o = Orchestrator.from_job(store, store.load_job(job.id))
    assert o.evaluate is False and o.face_swap is False and o.max_attempts == 3
    assert o.params.aspect == "16:9" and o.profile is not None  # default profile + job knobs


def test_stills_job_single_gate_no_videos(wired):
    from vorch import profiles

    store, master, calls = wired
    prof = profiles.apply_overrides(profiles.builtin_profiles()["stills_16x9"], {"stills": {"variants": 2}})
    o = Orchestrator(store, profile=prof)
    job = o.create_job("posters", master_face=str(master))
    assert job.kind == "stills" and o.gates(job) == ["storyboard"]

    o.run(job)
    assert job.status is JobStatus.NEEDS_APPROVAL and Orchestrator.pending_gate(job, store) == "storyboard"
    assert calls["keyframe"] == 0

    o.approve(job, GATE_STORYBOARD)
    assert job.status is JobStatus.DONE
    assert calls["keyframe"] == 4 and calls["animate"] == 0          # 2 scenes x 2 variants, no Wan
    assert calls["swap"] == 4                                        # master imposed on every variant
    for s in job.scenes:
        assert len(s.variants) == 2 and s.keyframe_path in {v.path for v in s.variants}
        assert all(v.swapped_path for v in s.variants) and s.swapped_path
        assert s.status is SceneStatus.COMPLETE
    assert Orchestrator.pending_gate(job, store) is None
    phases = {d.phase for d in store.decisions(job.id)}
    assert "stills" in phases and "video" not in phases


def test_stills_variants_best_chosen_and_resume_mid_scene(wired, monkeypatch):
    from vorch import profiles

    store, master, calls = wired
    scores = iter([3, 9, 6, 5])
    seen_seeds = []

    def scored_eval(prompt, task, *, seed, max_attempts, on_attempt=None, **kw):
        calls["keyframe"] += 1
        seen_seeds.append(seed)
        v = _verdict(score=next(scores))
        path = f"/kf/{calls['keyframe']}.png"
        if on_attempt:
            on_attempt(1, path, v)
        return path, v

    monkeypatch.setattr(pipeline, "generate_keyframe_evaluated", scored_eval)
    prof = profiles.apply_overrides(
        profiles.builtin_profiles()["stills_16x9"],
        {"stills": {"variants": 2}, "face_swap": {"enabled": False}},
    )
    o = Orchestrator(store, profile=prof, seed=100)
    job = o.create_job("choose", autonomy="full")
    o.run(job)
    assert job.status is JobStatus.DONE
    assert job.scenes[0].keyframe_path == "/kf/2.png"   # score 9 beats 3
    assert job.scenes[1].keyframe_path == "/kf/3.png"   # 6 beats 5
    assert [v.seed for v in job.scenes[0].variants] == [100, 200]         # derive_seed per variant
    assert seen_seeds == [100, 200, 10_100, 10_200]

    # Resume mid-scene: drop the second variant of scene 1 and re-run -> only that one regenerates.
    job.scenes[1].variants = job.scenes[1].variants[:1]
    job.scenes[1].keyframe_path = None
    store.save_job(job)
    scores = iter([8])
    before = calls["keyframe"]
    Orchestrator.from_job(store, store.load_job(job.id)).run(store.load_job(job.id))
    assert calls["keyframe"] == before + 1
    reloaded = store.load_job(job.id)
    assert len(reloaded.scenes[1].variants) == 2 and reloaded.status is JobStatus.DONE


def test_choose_variant_rules():
    from vorch.orchestrator import choose_variant
    from vorch.state import Variant

    a, b, c = Variant("a", verdict=_verdict_retry()), Variant("b", verdict=_verdict(5)), Variant("c")
    assert choose_variant([a, b, c]) is b                # accepted beats retry / unscored
    assert choose_variant([a, c]) is a                   # scored beats unscored
    assert choose_variant([c, Variant("d")]).path == "d" # no verdicts: last
    import pytest as _pt
    with _pt.raises(ValueError):
        choose_variant([])


def test_update_scenes_chosen_variant(wired):
    from vorch.queue import JobManager
    from vorch.state import Scene, Variant, new_job

    store, _, _ = wired
    job = new_job(idea="v", kind="stills")
    job.status = JobStatus.DONE
    job.scenes = [Scene(index=0, keyframe_path="/a.png", variants=[
        Variant("/a.png", swapped_path="/as.png"), Variant("/b.png", verdict=_verdict(7), swapped_path="/bs.png")])]
    store.save_job(job)
    mgr = JobManager(store)
    updated = mgr.update_scenes(job.id, [{"index": 0, "chosen_variant": 1}])
    assert updated.scenes[0].keyframe_path == "/b.png" and updated.scenes[0].swapped_path == "/bs.png"
    assert updated.scenes[0].keyframe_verdict.score == 7
    with pytest.raises(ValueError):
        mgr.update_scenes(job.id, [{"index": 0, "chosen_variant": 5}])


def test_progress_events_phase_and_step(wired):
    """Every phase announces itself and every generation step is announced before it runs."""
    store, master, _ = wired
    o = Orchestrator(store)
    seen: list[dict] = []
    o.emit = seen.append
    job = o.create_job("progress", master_face=str(master), autonomy="full")
    o.run(job)
    types = [(e["type"], e.get("phase") or e.get("step")) for e in seen]
    assert types[:3] == [("status", None), ("phase", "storyboard"), ("step", "storyboard")]
    assert ("phase", "keyframes") in types and ("phase", "videos") in types
    kf_steps = [e for e in seen if e["type"] == "step" and e["step"] == "keyframe"]
    assert [e["index"] for e in kf_steps] == [0, 1] and kf_steps[0]["attempt"] == 1 and kf_steps[0]["max_attempts"] == 2
    order = [(e["type"], e.get("step"), e.get("index")) for e in seen if e["type"] in ("step", "keyframe", "video")]
    assert order.index(("step", "keyframe", 0)) < order.index(("keyframe", None, 0))
    assert ("step", "swap", 0) in order and order.index(("step", "animate", 1)) < order.index(("video", None, 1))
    assert seen[-1] == {"job_id": job.id, "type": "status", "status": "done"}


# --- regenerate one scene ---------------------------------------------------------------


def test_regenerate_keyframe_reopens_gate_and_reruns_only_that_scene(wired):
    store, master, calls = wired
    o = Orchestrator(store, evaluate=False, seed=100)
    job = o.create_job("redo", master_face=str(master))
    o.run(job)
    o.approve(job, GATE_STORYBOARD)
    o.approve(job, GATE_KEYFRAMES)
    assert job.status is JobStatus.DONE and calls == {"keyframe": 2, "swap": 2, "animate": 2}
    old_kf0, old_vid1 = job.scenes[0].keyframe_path, job.scenes[1].video_path
    job.export = {"path": "/x", "ok": True}
    store.save_job(job)

    o.regenerate(job, 0, edits={"keyframe_prompt": "kf::better"})
    # Only scene 0 was redone, and the job paused at the keyframes gate again.
    assert calls == {"keyframe": 3, "swap": 2, "animate": 2}
    assert job.status is JobStatus.NEEDS_APPROVAL and Orchestrator.pending_gate(job, store) == GATE_KEYFRAMES
    s0 = job.scenes[0]
    assert s0.keyframe_path != old_kf0 and s0.video_path is None and s0.keyframe_prompt == "kf::better"
    assert s0.regenerations == 1 and job.scenes[1].video_path == old_vid1
    assert store.load_job(job.id).export is None
    log = [(d.scene_index, d.phase, d.action) for d in store.decisions(job.id)]
    assert (0, "keyframe", "regenerate") in log and (None, "keyframes", "reopened") in log

    o.approve(job, GATE_KEYFRAMES)
    assert job.status is JobStatus.DONE and calls == {"keyframe": 3, "swap": 3, "animate": 3}


def test_regenerate_video_only_and_errors(wired):
    store, master, calls = wired
    o = Orchestrator(store, evaluate=False, face_swap=False)
    job = o.create_job("clip", autonomy="full")
    o.run(job)
    assert job.status is JobStatus.DONE and calls["animate"] == 2
    o.regenerate(job, 1, what="video", edits={"animation_prompt": "an::slower"})
    assert job.status is JobStatus.DONE and calls == {"keyframe": 2, "swap": 0, "animate": 3}
    assert job.scenes[1].animation_prompt == "an::slower" and job.scenes[1].regenerations == 1
    with pytest.raises(KeyError):
        o.reopen_scene(job, 9)
    with pytest.raises(ValueError):
        o.reopen_scene(job, 0, what="audio")


def test_regenerate_seed_offsets(wired, monkeypatch):
    store, master, _ = wired
    seeds = []
    monkeypatch.setattr(pipeline, "_fresh_keyframe",
                        lambda prompt, *, seed, **kw: seeds.append(seed) or (f"/kf/{seed}.png", None))
    o = Orchestrator(store, evaluate=False, face_swap=False, seed=100)
    job = o.create_job("seeds", autonomy="full")
    o.run(job)
    o.regenerate(job, 0)
    assert seeds == [100, 10_100, 110]  # scene 0, scene 1, scene 0 regenerated (+10)


def test_regenerate_applies_evaluator_suggestion(wired, monkeypatch):
    store, master, calls = wired
    seen_prompts = []

    def retry_eval(prompt, task, *, seed, max_attempts, on_attempt=None, **kw):
        seen_prompts.append(prompt)
        calls["keyframe"] += 1
        v = _verdict_retry()  # fix_suggestion="use a peel"
        if on_attempt:
            on_attempt(1, f"/kf/{calls['keyframe']}.png", v)
        return f"/kf/{calls['keyframe']}.png", v

    monkeypatch.setattr(pipeline, "generate_keyframe_evaluated", retry_eval)
    o = Orchestrator(store, max_attempts=1, face_swap=False)
    job = o.create_job("fix", autonomy="full")
    o.run(job)
    assert o.last_fix_suggestion(job, job.scenes[0]) == "use a peel"

    o.regenerate(job, 0, edits={"apply_suggestion": True})
    assert seen_prompts[-1] == "kf::d1. Correction: use a peel"
    assert job.scenes[0].keyframe_prompt.endswith("Correction: use a peel")
    reasons = [d.reason for d in store.decisions(job.id) if d.action == "regenerate"]
    assert reasons[-1] == "edited: applied evaluator suggestion"
    # Applying again does not duplicate the correction.
    o.regenerate(job, 0, edits={"apply_suggestion": True})
    assert job.scenes[0].keyframe_prompt.count("Correction:") == 1


def test_n_scenes_caps_the_storyboard(wired):
    """The fake storyboard returns 2 briefs; n_scenes=1 keeps only the first."""
    store, master, _ = wired
    o = Orchestrator(store, n_scenes=1)
    job = o.create_job("idea", master_face=str(master), autonomy="semi")
    o.run(job)
    assert len(job.scenes) == 1 and job.scenes[0].description == "d1"


def _faces_project(store, tmp_path, monkeypatch):
    from vorch import config, projects

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    p = projects.new_project("Ages")
    young = projects.add_asset(p, b"y", kind="master_face", name="young.png", label="Maria, age 8")
    old = projects.add_asset(p, b"o", kind="master_face", name="old.png", label="Maria, age 70")
    projects.save(store, p)
    return p, young, old


def test_storyboard_assigns_identities_per_scene_and_swaps_accordingly(wired, tmp_path, monkeypatch):
    store, master, calls = wired
    p, young, old = _faces_project(store, tmp_path, monkeypatch)
    briefs = [
        SceneBrief(scene=1, description="child", identity=young.id),
        SceneBrief(scene=2, description="street", identity=None),          # explicit: no face
        SceneBrief(scene=3, description="grandmother", identity="ghost"),  # unknown id -> default kept
    ]
    monkeypatch.setattr(storyboard, "generate_storyboard", lambda idea, *, n_scenes=None, **kw: briefs)
    swaps = []
    monkeypatch.setattr(pipeline, "swap_master_onto", lambda frame, master_name, **kw: swaps.append(master_name) or f"/swap/{len(swaps)}.png")
    o = Orchestrator(store, n_scenes=3, evaluate=False)
    job = o.create_job("a life", master_face=str(master), autonomy="full", project_id=p.id)
    o.run(job)
    assert job.status is JobStatus.DONE
    s0, s1, s2 = job.scenes
    assert (s0.identity, s0.identity_asset_id) == (young.path, young.id)
    assert s1.identity == "" and s1.identity_asset_id is None and s1.swapped_path is None
    assert s2.identity == str(master) and s2.identity_asset_id is None      # default face kept
    assert swaps == [young.path, str(master)]                                # scene 1 skipped
    actions = [(d.scene_index, d.action, d.reason) for d in store.decisions(job.id) if d.phase == "storyboard"]
    assert (0, "assigned", "identity: Maria, age 8") in actions
    assert (1, "assigned", "identity: none") in actions
    assert any(i == 2 and a == "warning" and "ghost" in r for i, a, r in actions)


def test_regenerate_video_reswaps_with_the_scene_identity(wired, tmp_path, monkeypatch):
    store, master, calls = wired
    p, young, old = _faces_project(store, tmp_path, monkeypatch)
    monkeypatch.setattr(storyboard, "generate_storyboard",
                        lambda idea, *, n_scenes=None, **kw: [SceneBrief(scene=1, description="d", identity=young.id)])
    swaps = []
    monkeypatch.setattr(pipeline, "swap_master_onto", lambda frame, master_name, **kw: swaps.append(master_name) or f"/swap/{len(swaps)}.png")
    o = Orchestrator(store, n_scenes=1, evaluate=False)
    job = o.create_job("a life", master_face=str(master), autonomy="full", project_id=p.id)
    o.run(job)
    assert swaps == [young.path] and job.scenes[0].swapped_path == "/swap/1.png"
    o.regenerate(job, 0, what="video", edits={"identity_asset_id": old.id})
    assert job.scenes[0].identity == old.path and swaps == [young.path, old.path]
    assert job.scenes[0].swapped_path == "/swap/2.png" and job.status is JobStatus.DONE


def test_face_swap_without_default_master_needs_project_faces(wired, tmp_path, monkeypatch):
    store, master, _ = wired
    o = Orchestrator(store, n_scenes=1)
    with pytest.raises(FileNotFoundError, match="no master_face assets"):
        o.create_job("x", master_face="", autonomy="full")
    p, _young, _old = _faces_project(store, tmp_path, monkeypatch)
    job = o.create_job("x", master_face="", autonomy="full", project_id=p.id)
    assert job.master_face == "" and job.face_swap is True


def _refs_project(store, tmp_path, monkeypatch):
    from vorch import config, projects

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    p = projects.new_project("Coast")
    stairs = projects.add_asset(p, b"s", kind="reference", name="stairs.png", label="spiral staircase")
    projects.save(store, p)
    return p, stairs


def test_t2i_storyboard_attaches_reference_hints_without_applying_them(wired, tmp_path, monkeypatch):
    from vorch.state import job_from_dict, job_to_dict

    store, master, _ = wired
    p, stairs = _refs_project(store, tmp_path, monkeypatch)
    seen = {}

    def fake(briefs, assets, **kw):
        seen["assets"] = assets
        return {1: storyboard.ReferencePick(scene=2, asset_id=stairs.id, why="same stairs", fit="same_shot")}, ["ghost"]

    monkeypatch.setattr(storyboard, "suggest_references", fake)
    o = Orchestrator(store, n_scenes=2, face_swap=False)
    job = o.create_job("idea", master_face=str(master), autonomy="semi", project_id=p.id)
    o.run(job)
    s0, s1 = job.scenes
    assert s0.reference_suggestion is None
    assert (s1.reference_suggestion.asset_id, s1.reference_suggestion.fit) == (stairs.id, "same_shot")
    assert s1.mode == "t2i" and s1.reference_asset_ids == []        # a hint, not an assignment
    assert any(a[0] == stairs.id for a in seen["assets"])
    again = job_from_dict(job_to_dict(job)).scenes[1].reference_suggestion
    assert again == s1.reference_suggestion
    actions = [(d.scene_index, d.action) for d in store.decisions(job.id) if d.phase == "storyboard"]
    assert (1, "suggested") in actions and (None, "warning") in actions


def test_failed_reference_hints_do_not_fail_the_storyboard(wired, tmp_path, monkeypatch):
    store, master, _ = wired
    p, _ = _refs_project(store, tmp_path, monkeypatch)

    def boom(briefs, assets, **kw):
        raise storyboard.StoryboardError("could not parse")

    monkeypatch.setattr(storyboard, "suggest_references", boom)
    o = Orchestrator(store, n_scenes=2, face_swap=False)
    job = o.create_job("idea", master_face=str(master), autonomy="semi", project_id=p.id)
    o.run(job)
    assert job.status is JobStatus.NEEDS_APPROVAL and all(s.reference_suggestion is None for s in job.scenes)
    assert any("reference suggestions failed" in d.reason for d in store.decisions(job.id))


# --- animating images made elsewhere ----------------------------------------------------


def _assets_project(store, tmp_path, monkeypatch, n=2):
    from vorch import config, media, projects

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(media, "fit_image", lambda src, w, h, dest=None: Path(dest or f"{src}.fit"))
    monkeypatch.setattr(media, "probe", lambda p: media.MediaInfo(576, 1024, 0.0, 0.0, 0, "png", False))
    p = projects.new_project("Perfume")
    made = []
    for i in range(n):
        a = projects.add_asset(p, b"x", kind="product", name=f"shot{i}.png")
        projects.set_asset_note(p, a.id, {"caption": f"the flacon, take {i}", "setting": "cream stone",
                                          "subjects": ["glass flacon", "amber liquid"],
                                          "time_of_day": "day", "weather": "indoor"})
        made.append(a)
    projects.save(store, p)
    return p, made


def test_assets_become_scenes_with_their_own_keyframes_and_motion(wired, tmp_path, monkeypatch):
    """The images are the keyframes: no keyframe is generated, and the motion is written
    by looking at each frame."""
    from vorch.storyboard import PromptPair

    store, master, calls = wired
    project, assets = _assets_project(store, tmp_path, monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(storyboard, "motion_from_image",
                        lambda path, *, direction="", context=None, **kw:
                        seen.append(f"{Path(path).name}|{direction}") or
                        PromptPair(positive=f"motion for {Path(path).name}", negative="jitter"))

    o = Orchestrator(store, n_scenes=None, face_swap=False, evaluate=False)
    job = o.create_job("intimate, slow, warm", master_face="", autonomy="semi",
                       project_id=project.id, keyframe_asset_ids=[a.id for a in assets])
    o.run(job)

    assert [s.index for s in job.scenes] == [0, 1] and job.n_scenes == 2
    assert [s.description for s in job.scenes] == ["the flacon, take 0", "the flacon, take 1"]
    assert [s.setting for s in job.scenes] == ["cream stone"] * 2
    assert all(s.keyframe_path and s.variants[0].source == "user" for s in job.scenes)
    assert [v.asset_id for s in job.scenes for v in s.variants] == [a.id for a in assets]
    assert calls.get("keyframe", 0) == 0                     # nothing was generated
    assert all(d.endswith("|intimate, slow, warm") for d in seen)   # the direction reached Gemma
    assert [s.animation_prompt for s in job.scenes] == [f"motion for {Path(s.keyframe_path).name}"
                                                        for s in job.scenes]
    assert job.status is JobStatus.NEEDS_APPROVAL              # the gate reviews the clips to be
    assert Orchestrator.pending_gate(job, store) == "storyboard"


def test_a_resumed_asset_job_does_not_rebuild_its_scenes(wired, tmp_path, monkeypatch):
    from vorch.storyboard import PromptPair

    store, master, _ = wired
    project, assets = _assets_project(store, tmp_path, monkeypatch, n=1)
    monkeypatch.setattr(storyboard, "motion_from_image",
                        lambda path, **kw: PromptPair(positive="drift", negative=""))
    o = Orchestrator(store, face_swap=False, evaluate=False)
    job = o.create_job("d", master_face="", autonomy="semi", project_id=project.id,
                       keyframe_asset_ids=[assets[0].id])
    o.run(job)
    o.run(job)                                                # resume: same scenes, no duplicates
    assert len(job.scenes) == 1


def test_an_asset_of_another_aspect_warns_before_it_is_cropped(wired, tmp_path, monkeypatch):
    from vorch import media
    from vorch.storyboard import PromptPair

    store, master, _ = wired
    project, assets = _assets_project(store, tmp_path, monkeypatch, n=1)
    # A portrait asset in a 16:9 profile: the sides of a product shot would vanish silently.
    monkeypatch.setattr(media, "probe", lambda p: media.MediaInfo(1080, 1920, 0.0, 0.0, 0, "png", False))
    monkeypatch.setattr(storyboard, "motion_from_image",
                        lambda path, **kw: PromptPair(positive="drift", negative=""))
    o = Orchestrator(store, face_swap=False, evaluate=False)
    job = o.create_job("d", master_face="", autonomy="semi", project_id=project.id,
                       keyframe_asset_ids=[assets[0].id])
    o.run(job)
    warning = [d for d in store.decisions(job.id) if d.action == "warning"]
    assert warning and "1080x1920" in warning[0].reason and "centre-cropped" in warning[0].reason


def test_a_users_own_image_is_never_shrunk_below_the_clip_it_feeds(wired, tmp_path, monkeypatch):
    """A keyframe size smaller than the clip size is a budget for *generating* a frame.
    Applied to an image made elsewhere it would delete detail the animator then has to
    invent — which is how fine print on a product label is lost before rendering starts."""
    from vorch import media
    from vorch.genparams import GenParams
    from vorch.storyboard import PromptPair

    store, master, _ = wired
    project, assets = _assets_project(store, tmp_path, monkeypatch, n=1)
    sizes: list[tuple[int, int]] = []
    monkeypatch.setattr(media, "fit_image",
                        lambda src, w, h, dest=None: sizes.append((w, h)) or Path(dest or f"{src}.fit"))
    monkeypatch.setattr(storyboard, "motion_from_image",
                        lambda path, **kw: PromptPair(positive="drift", negative=""))
    params = GenParams(aspect="1:1", keyframe_size=(768, 768), wan_size=(960, 960),
                       face_swap=False, evaluate=False)
    o = Orchestrator(store, params=params, face_swap=False, evaluate=False)
    job = o.create_job("d", master_face="", autonomy="semi", project_id=project.id,
                       keyframe_asset_ids=[assets[0].id])
    o.run(job)
    assert sizes == [(960, 960)]                       # the clip size, not the keyframe size

    # The other way round the keyframe size already wins: never upscale for nothing.
    bigger = GenParams(aspect="1:1", keyframe_size=(1024, 1024), wan_size=(640, 640),
                       face_swap=False, evaluate=False)
    assert Orchestrator(store, params=bigger).install_size() == (1024, 1024)
    # A stills job has no clip to feed: the keyframe size is the whole story.
    stills = GenParams(kind="stills", aspect="1:1", keyframe_size=(768, 768), wan_size=(960, 960))
    assert Orchestrator(store, params=stills).install_size() == (768, 768)


def test_an_unknown_asset_stops_the_job_before_it_renders(wired, tmp_path, monkeypatch):
    store, master, _ = wired
    project, _assets = _assets_project(store, tmp_path, monkeypatch, n=1)
    o = Orchestrator(store, face_swap=False, evaluate=False)
    job = o.create_job("d", master_face="", autonomy="full", project_id=project.id,
                       keyframe_asset_ids=["ghost"])
    with pytest.raises(ValueError, match="unknown project asset"):
        o.run(job)
