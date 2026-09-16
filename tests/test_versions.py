"""Every keyframe and clip a scene has had is kept and selectable.

The scene's scalar media fields are the projection of the active pair; history lives in
``Scene.variants`` / ``Scene.clips``. Regenerating appends; nothing on disk is touched.
"""

from __future__ import annotations

import asyncio

import pytest

from vorch import assembly, pipeline, profiles, storyboard, versions
from vorch import orchestrator as om
from vorch.orchestrator import GATE_KEYFRAMES, Orchestrator
from vorch.queue import JobManager
from vorch.state import Clip, JobStatus, Scene, SceneStatus, Variant, new_job
from vorch.storyboard import PromptPair, SceneBrief


@pytest.fixture
def wired(monkeypatch, make_store):
    calls = {"keyframe": 0, "animate": 0}
    monkeypatch.setattr(storyboard, "generate_storyboard",
                        lambda idea, *, n_scenes=None, **kw: [SceneBrief(scene=1, description="d")])
    monkeypatch.setattr(storyboard, "keyframe_prompt", lambda b, **kw: PromptPair(positive="kf"))
    monkeypatch.setattr(storyboard, "animation_prompt", lambda b, **kw: PromptPair(positive="an"))

    def fresh(prompt, *, seed, **kw):
        calls["keyframe"] += 1
        return f"/kf/{calls['keyframe']}.png", None

    def animate(frame, prompt, *, seed, **kw):
        calls["animate"] += 1
        return f"/clip/{calls['animate']}.mp4"

    monkeypatch.setattr(pipeline, "_fresh_keyframe", fresh)
    monkeypatch.setattr(pipeline, "animate_frame", animate)
    return make_store(), calls


def _profile(**over):
    base = {"evaluate": {"enabled": False}, "face_swap": {"enabled": False}}
    base.update(over)
    return profiles.apply_overrides(profiles.default_profile(), base)


# --- history is kept -------------------------------------------------------------------


def test_regenerating_keeps_every_keyframe_and_clip(wired):
    store, calls = wired
    o = Orchestrator(store, profile=_profile(), n_scenes=1)
    job = o.create_job("idea", autonomy="full")
    o.run(job)
    s = job.scenes[0]
    assert [v.path for v in s.variants] == ["/kf/1.png"] and [c.video_path for c in s.clips] == ["/clip/1.mp4"]
    assert s.variants[0].batch == 0 and s.clips[0].keyframe_path == "/kf/1.png"

    o.regenerate(job, 0, what="keyframe")
    s = store.load_job(job.id).scenes[0]
    assert [v.path for v in s.variants] == ["/kf/1.png", "/kf/2.png"]      # the old frame survives
    assert [c.video_path for c in s.clips] == ["/clip/1.mp4", "/clip/2.mp4"]
    assert [v.batch for v in s.variants] == [0, 1] and s.keyframe_path == "/kf/2.png"
    assert s.clips[-1].keyframe_path == "/kf/2.png"   # each clip records the frame it came from


def test_re_animating_keeps_the_earlier_take(wired):
    store, calls = wired
    o = Orchestrator(store, profile=_profile(), n_scenes=1)
    job = o.create_job("idea", autonomy="full")
    o.run(job)
    o.regenerate(job, 0, what="video")
    s = store.load_job(job.id).scenes[0]
    assert [c.video_path for c in s.clips] == ["/clip/1.mp4", "/clip/2.mp4"]
    assert s.video_path == "/clip/2.mp4" and calls["keyframe"] == 1  # the keyframe was not redone


def test_a_stills_run_generates_a_fresh_batch_and_chooses_within_it(wired, monkeypatch):
    from vorch.evaluator import Verdict

    store, calls = wired
    scores = iter([5, 9, 2, 3, 4, 8])

    def evaluated(prompt, task, *, seed, max_attempts, on_attempt=None, **kw):
        calls["keyframe"] += 1
        score = next(scores)
        v = Verdict(action_correct=True, identity_match=True, style_match=True, artifacts=False,
                    composition_ok=True, score=score, verdict="accept", fix_suggestion="")
        return f"/kf/{calls['keyframe']}.png", v

    monkeypatch.setattr(pipeline, "generate_keyframe_evaluated", evaluated)
    prof = profiles.apply_overrides(profiles.builtin_profiles()["stills_16x9"],
                                    {"stills": {"variants": 3}, "face_swap": {"enabled": False}})
    o = Orchestrator(store, profile=prof, n_scenes=1)
    job = o.create_job("idea", autonomy="full")
    o.run(job)
    s = job.scenes[0]
    assert len(s.variants) == 3 and s.keyframe_path == "/kf/2.png"  # best of the batch (9)

    o.regenerate(job, 0, what="keyframe")
    s = store.load_job(job.id).scenes[0]
    assert len(s.variants) == 6                      # a whole new batch, not "already done"
    assert [v.batch for v in s.variants] == [0, 0, 0, 1, 1, 1]
    assert s.keyframe_path == "/kf/6.png"            # best of the NEW batch (8), not the old 9


def test_a_user_keyframe_in_history_does_not_freeze_later_runs():
    scene = Scene(index=0, keyframe_path="/u.png", status=SceneStatus.KEYFRAME,
                  variants=[Variant("/u.png", source="user"), Variant("/g.png", batch=1)])
    assert Orchestrator._scene_needs_keyframes(scene, 1) is False   # the user's image is active
    scene.keyframe_path = "/g.png"
    assert Orchestrator._scene_needs_keyframes(scene, 1) is False   # a generated one is active now
    scene.keyframe_path = None
    assert Orchestrator._scene_needs_keyframes(scene, 1) is True    # cleared: generate again


# --- selecting ---------------------------------------------------------------------------


def _video_job(store, **kw):
    job = new_job(idea="v", kind="video", autonomy="semi", **kw)
    job.status = JobStatus.DONE
    job.export = {"path": "/e"}
    job.scenes = [Scene(index=0, keyframe_path="/kf2.png", video_path="/clip2.mp4", raw_video_path="/clip2.mp4",
                        status=SceneStatus.COMPLETE,
                        variants=[Variant("/kf1.png", id="k1"), Variant("/kf2.png", id="k2")],
                        clips=[Clip("/clip1.mp4", raw_video_path="/clip1.mp4", keyframe_path="/kf1.png", id="c1"),
                               Clip("/clip2.mp4", raw_video_path="/clip2.mp4", keyframe_path="/kf2.png", id="c2")])]
    store.save_job(job)
    return job


def test_choosing_an_older_keyframe_brings_back_its_clip(make_store):
    store = make_store()
    job = _video_job(store)
    mgr = JobManager(store, auto_export=False)
    updated = mgr.update_scenes(job.id, [{"index": 0, "chosen_variant_id": "k1"}])
    s = updated.scenes[0]
    assert s.keyframe_path == "/kf1.png" and s.video_path == "/clip1.mp4"   # nothing to re-render
    assert s.status is SceneStatus.COMPLETE and updated.status is JobStatus.DONE
    assert not any(d.action == "reopened" for d in store.decisions(job.id))
    store.close()


def test_choosing_a_keyframe_with_no_clip_still_reopens_the_gate(make_store):
    store = make_store()
    job = _video_job(store)
    job.scenes[0].variants.append(Variant("/kf3.png", id="k3"))
    store.save_job(job)
    mgr = JobManager(store, auto_export=False)
    updated = mgr.update_scenes(job.id, [{"index": 0, "chosen_variant_id": "k3"}])
    s = updated.scenes[0]
    assert s.keyframe_path == "/kf3.png" and s.video_path is None
    assert updated.status is JobStatus.NEEDS_APPROVAL
    assert any(d.phase == GATE_KEYFRAMES and d.action == "reopened" for d in store.decisions(job.id))
    store.close()


def test_choosing_a_clip_changes_what_assembly_uses(make_store):
    store = make_store()
    job = _video_job(store, profile={"video": {"width": 832, "height": 480, "fps": 16, "length": 49}})
    mgr = JobManager(store, auto_export=False)
    updated = mgr.update_scenes(job.id, [{"index": 0, "chosen_clip_id": "c1"}])
    s = updated.scenes[0]
    assert s.video_path == "/clip1.mp4" and s.keyframe_path == "/kf2.png"  # the keyframe is untouched
    assert updated.export is None and updated.status is JobStatus.DONE
    assert [assembly._clip_path(x, "final") for x in assembly.clips_of(updated)] == ["/clip1.mp4"]
    store.close()


def test_unknown_version_ids_are_rejected(make_store):
    store = make_store()
    job = _video_job(store)
    mgr = JobManager(store, auto_export=False)
    with pytest.raises(ValueError, match="no keyframe version"):
        mgr.update_scenes(job.id, [{"index": 0, "chosen_variant_id": "nope"}])
    with pytest.raises(ValueError, match="no clip version"):
        mgr.update_scenes(job.id, [{"index": 0, "chosen_clip_id": "nope"}])
    store.close()


# --- deleting ----------------------------------------------------------------------------


def test_deleting_a_version_keeps_the_file_and_re_points_when_it_was_active(make_store):
    store = make_store()
    job = _video_job(store)
    mgr = JobManager(store, auto_export=False)

    updated = mgr.delete_version(job.id, 0, "c2")            # the active clip
    s = updated.scenes[0]
    assert [c.id for c in s.clips] == ["c1"] and s.video_path == "/clip1.mp4"

    updated = mgr.delete_version(job.id, 0, "k1")            # an inactive keyframe
    s = updated.scenes[0]
    assert [v.id for v in s.variants] == ["k2"] and s.keyframe_path == "/kf2.png"

    with pytest.raises(ValueError, match="no version"):
        mgr.delete_version(job.id, 0, "gone")
    with pytest.raises(KeyError):
        mgr.delete_version(job.id, 9, "k2")
    assert [d.action for d in store.decisions(job.id) if d.action == "deleted"] == ["deleted", "deleted"]
    store.close()


def test_deleting_the_last_keyframe_resets_the_scene(make_store):
    store = make_store()
    job = _video_job(store)
    mgr = JobManager(store, auto_export=False)
    mgr.delete_version(job.id, 0, "k1")
    updated = mgr.delete_version(job.id, 0, "k2")
    s = updated.scenes[0]
    assert s.variants == [] and s.keyframe_path is None and s.video_path is None
    assert s.status is SceneStatus.PENDING
    store.close()


def test_deleting_is_refused_while_the_job_runs(make_store):
    store = make_store()
    job = _video_job(store)
    mgr = JobManager(store, auto_export=False)
    mgr._active.add(job.id)
    with pytest.raises(RuntimeError):
        mgr.delete_version(job.id, 0, "k1")
    store.close()


# --- jobs from before versions existed ---------------------------------------------------


def test_a_job_saved_before_versions_shows_its_media_with_stable_ids(make_store):
    store = make_store()
    job = new_job(idea="old", kind="video")
    job.status = JobStatus.DONE
    job.scenes = [Scene(index=0, keyframe_path="/old_kf.png", video_path="/old_clip.mp4",
                        status=SceneStatus.COMPLETE)]
    store.save_job(job)

    from vorch.api.public import scene_to_public

    first = scene_to_public(store.load_job(job.id).scenes[0])
    again = scene_to_public(store.load_job(job.id).scenes[0])
    assert [v.id for v in first.variants] == [v.id for v in again.variants]  # stable without a write
    assert first.variants[0].active and first.clips[0].active
    assert first.variants[0].id == versions.stable_id("/old_kf.png")

    mgr = JobManager(store, auto_export=False)
    updated = mgr.update_scenes(job.id, [{"index": 0, "chosen_variant_id": first.variants[0].id}])
    assert updated.scenes[0].keyframe_path == "/old_kf.png"   # selectable straight away
    store.close()
