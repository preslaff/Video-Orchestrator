"""GPU-free checks for orchestrator.py — the phased job machine + gates.

A real Store (temp SQLite) verifies persistence, the decision log, and gate
resume; the Gemma/ComfyUI-bound calls (storyboard, keyframe eval, swap, animate)
are monkeypatched. Asserts which phase runs when, that semi-auto gates pause the
job, that approvals resume it, and that a reloaded job continues correctly.
"""

from __future__ import annotations

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
    monkeypatch.setattr(comfy, "upload_image", lambda p: str(p))

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
