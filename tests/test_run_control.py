"""Stop and resume through the orchestrator: a stop lands on a safe point, finished
work is kept, and resuming does only what is left — including the two gaps a stop
would otherwise expose (post-steps after the raw clip, a swap already done)."""

from __future__ import annotations

import threading

import pytest

from vorch import cancel, pipeline, profiles, storyboard
from vorch import orchestrator as om
from vorch.clients import comfy
from vorch.orchestrator import GATE_KEYFRAMES, GATE_STORYBOARD, Orchestrator
from vorch.state import JobStatus, SceneStatus
from vorch.storyboard import PromptPair, SceneBrief


@pytest.fixture
def seams(monkeypatch, tmp_path):
    calls = {"keyframe": [], "swap": [], "animate": [], "chain": []}
    stop = threading.Event()
    hooks = {}  # name -> callable run after that step, e.g. to press Stop

    briefs = [SceneBrief(scene=i + 1, description=f"d{i}") for i in range(3)]
    monkeypatch.setattr(storyboard, "generate_storyboard", lambda idea, *, n_scenes=None, **kw: briefs[:n_scenes or 3])
    monkeypatch.setattr(storyboard, "keyframe_prompt", lambda b, *, style="", **kw: PromptPair(positive=f"kf::{b.description}"))
    monkeypatch.setattr(storyboard, "animation_prompt", lambda b, **kw: PromptPair(positive=f"an::{b.description}"))

    def fresh(prompt, *, seed, **kw):
        calls["keyframe"].append(prompt)
        path = f"/kf/{len(calls['keyframe'])}.png"
        hooks.get("keyframe", lambda: None)()
        return path, None

    def swap(frame, master, **kw):
        calls["swap"].append(frame)
        hooks.get("swap", lambda: None)()
        return f"/swap/{len(calls['swap'])}.png"

    def animate(frame, prompt, *, seed, **kw):
        calls["animate"].append((str(frame), seed))
        hooks.get("animate", lambda: None)()
        return f"/raw/{len(calls['animate'])}.mp4"

    def chain(path, steps, *, seed=None, **kw):
        cancel.check()  # the real run_chain checks before each step
        calls["chain"].append((str(path), seed))
        return [f"/post/{len(calls['chain'])}.mp4"]

    monkeypatch.setattr(pipeline, "_fresh_keyframe", fresh)
    monkeypatch.setattr(pipeline, "swap_master_onto", swap)
    monkeypatch.setattr(pipeline, "animate_frame", animate)
    monkeypatch.setattr(pipeline, "stage_identity", lambda face, params, cache=None: "staged-face")
    monkeypatch.setattr(om, "run_chain", chain)
    monkeypatch.setattr(comfy, "upload_image", lambda p, **kw: str(p))
    face = tmp_path / "face.png"
    face.write_bytes(b"x")
    return calls, stop, hooks, face


def _profile(post_steps=False, face_swap=False):
    over = {"evaluate": {"enabled": False}, "face_swap": {"enabled": face_swap}}
    if post_steps:
        over["post_steps"] = [{"workflow": "interp"}]
    return profiles.apply_overrides(profiles.default_profile(), over)


def _run_bound(stop, o, job, fn="run", *args):
    return cancel.call_bound(stop, getattr(o, fn), job, *args)


def test_stop_during_keyframes_keeps_finished_scenes_and_resume_does_the_rest(seams, make_store):
    calls, stop, hooks, _ = seams
    store = make_store()
    o = Orchestrator(store, profile=_profile(), n_scenes=3)
    job = o.create_job("idea", autonomy="full")
    hooks["keyframe"] = lambda: stop.set() if len(calls["keyframe"]) == 2 else None

    with pytest.raises(cancel.JobStopped):
        _run_bound(stop, o, job)
    saved = store.load_job(job.id)
    # the second keyframe finished before the checkpoint and was saved; the third never started
    assert [s.keyframe_path for s in saved.scenes] == ["/kf/1.png", "/kf/2.png", None]

    o2 = Orchestrator.from_job(store, saved)
    o2.run(saved)
    assert saved.status is JobStatus.DONE
    assert len(calls["keyframe"]) == 3  # only the missing one was generated
    assert [s.keyframe_path for s in saved.scenes] == ["/kf/1.png", "/kf/2.png", "/kf/3.png"]


def test_stop_between_clip_and_post_steps_resumes_only_the_chain(seams, make_store):
    calls, stop, hooks, _ = seams
    store = make_store()
    o = Orchestrator(store, profile=_profile(post_steps=True), n_scenes=1, seed=5)
    job = o.create_job("idea", autonomy="full")
    hooks["animate"] = stop.set

    with pytest.raises(cancel.JobStopped):
        _run_bound(stop, o, job)
    s = store.load_job(job.id).scenes[0]
    assert s.raw_video_path == "/raw/1.mp4" and s.video_path is None and calls["chain"] == []

    saved = store.load_job(job.id)
    Orchestrator.from_job(store, saved).run(saved)
    s = saved.scenes[0]
    assert len(calls["animate"]) == 1  # the clip was not rendered again
    assert calls["chain"] == [("/raw/1.mp4", calls["animate"][0][1])]  # same clip, same seed
    assert s.video_path == "/post/1.mp4" and s.status is SceneStatus.COMPLETE
    assert saved.status is JobStatus.DONE


def test_without_post_steps_the_clip_is_the_deliverable_at_once(seams, make_store):
    calls, stop, hooks, _ = seams
    store = make_store()
    o = Orchestrator(store, profile=_profile(), n_scenes=1)
    job = o.create_job("idea", autonomy="full")
    o.run(job)
    s = job.scenes[0]
    assert s.raw_video_path == s.video_path == "/raw/1.mp4" and calls["chain"] == []


def test_a_saved_swap_is_not_redone_on_resume(seams, make_store):
    calls, stop, hooks, face = seams
    store = make_store()
    o = Orchestrator(store, profile=_profile(face_swap=True), n_scenes=1)
    job = o.create_job("idea", master_face=str(face), autonomy="full")
    hooks["swap"] = stop.set

    with pytest.raises(cancel.JobStopped):
        _run_bound(stop, o, job)
    saved = store.load_job(job.id)
    assert saved.scenes[0].swapped_path == "/swap/1.png" and calls["animate"] == []

    Orchestrator.from_job(store, saved).run(saved)
    assert len(calls["swap"]) == 1
    assert calls["animate"][0][0] == "/swap/1.png"
    assert saved.status is JobStatus.DONE


def test_a_stop_after_the_last_phase_does_not_discard_the_finished_job(seams, make_store):
    calls, stop, hooks, _ = seams
    store = make_store()
    o = Orchestrator(store, profile=_profile(), n_scenes=1)
    job = o.create_job("idea", autonomy="full")
    hooks["animate"] = stop.set  # lands after the last unit of work
    _run_bound(stop, o, job)
    assert job.status is JobStatus.DONE


def test_gated_job_resume_after_stop_keeps_gate_state(seams, make_store):
    calls, stop, hooks, _ = seams
    store = make_store()
    o = Orchestrator(store, profile=_profile(), n_scenes=2)
    job = o.create_job("idea", autonomy="semi")
    o.run(job)
    assert job.status is JobStatus.NEEDS_APPROVAL
    hooks["keyframe"] = stop.set
    with pytest.raises(cancel.JobStopped):
        _run_bound(stop, o, job, "approve", GATE_STORYBOARD)
    saved = store.load_job(job.id)
    Orchestrator.from_job(store, saved).run(saved)
    # the storyboard approval survives the stop; the job continues to the keyframes gate
    assert saved.status is JobStatus.NEEDS_APPROVAL
    assert Orchestrator.pending_gate(saved, store) == GATE_KEYFRAMES
