"""Editing scenes at any stage: validation before anything starts, and invalidation of
outputs built on what an edit replaced (a face, a keyframe variant)."""

from __future__ import annotations

import asyncio

import pytest

from vorch import config, projects
from vorch.orchestrator import GATE_KEYFRAMES
from vorch.queue import JobManager
from vorch.state import JobStatus, Scene, SceneStatus, Variant, new_job


class NoRunOrch:
    """Fails loudly if a regenerate that should have been rejected ever starts."""

    def __init__(self, store, job):
        self.emit = lambda e: None

    def regenerate(self, job, index, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("regenerate started despite an invalid edit")


@pytest.fixture
def mgr(make_store, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    store = make_store()
    m = JobManager(store, orchestrator_factory=lambda s, j: NoRunOrch(s, j), auto_export=False)
    yield m
    store.close()


def _done_video_job(store, **scene_kw):
    job = new_job(idea="v", kind="video", autonomy="semi")
    job.status = JobStatus.DONE
    job.export = {"path": "/export"}
    job.scenes = [Scene(index=0, keyframe_path="/kf1.png", swapped_path="/sw.png", raw_video_path="/raw.mp4",
                        video_path="/final.mp4", post_paths=["/final.mp4"], status=SceneStatus.COMPLETE,
                        variants=[Variant("/kf1.png"), Variant("/kf2.png")], **scene_kw)]
    store.save_job(job)
    return job


# --- validation ------------------------------------------------------------------------


def test_edit_mode_without_references_is_rejected_and_nothing_is_saved(mgr):
    job = _done_video_job(mgr.store)
    with pytest.raises(ValueError, match="at least one reference"):
        mgr.update_scenes(job.id, [{"index": 0, "description": "new", "mode": "edit", "references": []}])
    saved = mgr.store.load_job(job.id).scenes[0]
    assert saved.mode == "t2i" and saved.description == ""  # all-or-nothing


def test_missing_reference_file_is_rejected(mgr, tmp_path):
    job = _done_video_job(mgr.store)
    with pytest.raises(ValueError, match="not found"):
        mgr.update_scenes(job.id, [{"index": 0, "mode": "edit", "references": [str(tmp_path / "gone.png")]}])
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"x")
    mgr.update_scenes(job.id, [{"index": 0, "mode": "edit", "references": [str(ref)]}])
    assert mgr.store.load_job(job.id).scenes[0].mode == "edit"


def test_prompt_only_edits_are_not_blocked_by_validation(mgr):
    job = _done_video_job(mgr.store)
    mgr.update_scenes(job.id, [{"index": 0, "keyframe_prompt": "sharper"}])
    assert mgr.store.load_job(job.id).scenes[0].keyframe_prompt == "sharper"


def test_invalid_regenerate_is_rejected_before_the_job_starts(mgr):
    job = _done_video_job(mgr.store)

    async def go():
        with pytest.raises(ValueError, match="at least one reference"):
            await mgr.regenerate(job.id, 0, what="keyframe", edits={"mode": "edit", "references": []})
        with pytest.raises(ValueError, match="unknown project asset|no project"):
            await mgr.regenerate(job.id, 0, what="keyframe", edits={"reference_asset_ids": ["nope"]})
        with pytest.raises(ValueError, match="what must be"):
            await mgr.regenerate(job.id, 0, what="audio")

    asyncio.run(go())
    assert mgr.store.load_job(job.id).status is JobStatus.DONE  # never marked running
    assert not mgr.is_active(job.id)


def test_regenerating_a_stills_clip_is_rejected_up_front(mgr):
    job = new_job(idea="s", kind="stills")
    job.status = JobStatus.DONE
    job.scenes = [Scene(index=0, keyframe_path="/k.png")]
    mgr.store.save_job(job)
    with pytest.raises(ValueError, match="no clip"):
        asyncio.run(mgr.regenerate(job.id, 0, what="video"))


# --- invalidation ----------------------------------------------------------------------


def test_new_face_drops_the_swap_and_clip_built_on_the_old_one(mgr, tmp_path):
    p = projects.new_project("Faces")
    young = projects.add_asset(p, b"y", kind="master_face", name="young.png")
    old = projects.add_asset(p, b"o", kind="master_face", name="old.png")
    projects.save(mgr.store, p)
    job = _done_video_job(mgr.store, identity=young.path, identity_asset_id=young.id)
    job.project_id = p.id
    mgr.store.save_job(job)

    mgr.update_scenes(job.id, [{"index": 0, "identity_asset_id": young.id}])  # unchanged: keeps outputs
    assert mgr.store.load_job(job.id).scenes[0].video_path == "/final.mp4"

    mgr.update_scenes(job.id, [{"index": 0, "identity_asset_id": old.id}])
    saved = mgr.store.load_job(job.id)
    s = saved.scenes[0]
    assert s.identity == old.path and s.keyframe_path == "/kf1.png"  # the keyframe stays
    assert (s.swapped_path, s.raw_video_path, s.video_path, s.post_paths) == (None, None, None, [])
    assert s.status is SceneStatus.KEYFRAME and saved.export is None


def test_new_face_on_stills_drops_every_variant_swap(mgr, tmp_path):
    p = projects.new_project("S")
    a = projects.add_asset(p, b"a", kind="master_face", name="a.png")
    b = projects.add_asset(p, b"b", kind="master_face", name="b.png")
    projects.save(mgr.store, p)
    job = new_job(idea="s", kind="stills")
    job.project_id = p.id
    job.status = JobStatus.DONE
    job.scenes = [Scene(index=0, keyframe_path="/k1.png", identity=a.path, identity_asset_id=a.id,
                        swapped_path="/s1.png", status=SceneStatus.COMPLETE,
                        variants=[Variant("/k1.png", swapped_path="/s1.png", post_paths=["/p1.png"]),
                                  Variant("/k2.png", swapped_path="/s2.png")])]
    mgr.store.save_job(job)
    mgr.update_scenes(job.id, [{"index": 0, "identity_asset_id": b.id}])
    s = mgr.store.load_job(job.id).scenes[0]
    assert all(v.swapped_path is None and v.post_paths == [] for v in s.variants)
    assert s.swapped_path is None and s.status is SceneStatus.KEYFRAME


def test_choosing_another_variant_on_a_done_video_job_reopens_the_keyframes_gate(mgr):
    job = _done_video_job(mgr.store)
    mgr.update_scenes(job.id, [{"index": 0, "chosen_variant": 0}])  # the current one: nothing changes
    assert mgr.store.load_job(job.id).scenes[0].video_path == "/final.mp4"

    mgr.update_scenes(job.id, [{"index": 0, "chosen_variant": 1}])
    saved = mgr.store.load_job(job.id)
    s = saved.scenes[0]
    assert s.keyframe_path == "/kf2.png" and s.video_path is None and s.raw_video_path is None
    assert s.status is SceneStatus.KEYFRAME and saved.export is None
    assert saved.status is JobStatus.NEEDS_APPROVAL
    assert any(d.phase == GATE_KEYFRAMES and d.action == "reopened" for d in mgr.store.decisions(job.id))


def test_choosing_another_variant_on_stills_shows_that_variants_outputs(mgr):
    job = new_job(idea="s", kind="stills")
    job.status = JobStatus.DONE
    job.scenes = [Scene(index=0, keyframe_path="/a.png", swapped_path="/as.png", post_paths=["/ap.png"],
                        variants=[Variant("/a.png", swapped_path="/as.png", post_paths=["/ap.png"]),
                                  Variant("/b.png", swapped_path="/bs.png", post_paths=["/bp.png"])])]
    mgr.store.save_job(job)
    mgr.update_scenes(job.id, [{"index": 0, "chosen_variant": 1}])
    saved = mgr.store.load_job(job.id)
    s = saved.scenes[0]
    assert (s.keyframe_path, s.swapped_path, s.post_paths) == ("/b.png", "/bs.png", ["/bp.png"])
    assert saved.status is JobStatus.DONE  # stills have no clip to redo


def test_reference_slot_warning_when_a_scene_has_more_than_the_workflow_uses(make_store, tmp_path, monkeypatch):
    from vorch import orchestrator as om
    from vorch import pipeline, profiles, storyboard
    from vorch.orchestrator import Orchestrator
    from vorch.storyboard import PromptPair, SceneBrief

    store = make_store()
    refs = []
    for i in range(3):
        r = tmp_path / f"r{i}.png"
        r.write_bytes(b"x")
        refs.append(str(r))
    monkeypatch.setattr(storyboard, "generate_storyboard", lambda idea, **kw: [SceneBrief(scene=1, description="d")])
    monkeypatch.setattr(storyboard, "keyframe_prompt", lambda b, **kw: PromptPair(positive="kf"))
    monkeypatch.setattr(storyboard, "animation_prompt", lambda b, **kw: PromptPair(positive="an"))
    monkeypatch.setattr(pipeline, "stage_references", lambda paths, params, cache=None: list(paths))
    monkeypatch.setattr(pipeline, "_fresh_keyframe", lambda *a, **k: ("/kf.png", None))
    monkeypatch.setattr(pipeline, "animate_frame", lambda *a, **k: "/v.mp4")
    prof = profiles.apply_overrides(profiles.default_profile(), {"evaluate": {"enabled": False},
                                                                 "face_swap": {"enabled": False}})
    o = Orchestrator(store, profile=prof)
    job = o.create_job("idea", mode="edit", reference_images=refs, autonomy="full")
    o.run(job)
    warnings = [d.reason for d in store.decisions(job.id) if d.action == "warning"]
    assert warnings == [f"3 references but {o.params.edit_workflow} uses the first 2"]
    assert om.reference_slots(o.params.edit_workflow) == 2


# --- structural edits: add / delete / reorder scenes -----------------------------------


def _paused_job(store, n=3):
    job = new_job(idea="coast", kind="video", autonomy="semi", n_scenes=n)
    job.status = JobStatus.NEEDS_APPROVAL
    job.scenes = [Scene(index=i, description=f"scene {i}", keyframe_prompt=f"kf{i}") for i in range(n)]
    return store.save_job(job)


def test_add_scene_inserts_at_once_and_asks_gemma_for_its_prompts(mgr, monkeypatch):
    from vorch import storyboard
    from vorch.storyboard import PromptPair

    seen = {}

    def keyframe_prompt(brief, *, style="", context=None, **kw):
        seen["brief"] = brief
        return PromptPair(positive="impasto oil painting, the lamp turning", negative="smooth")

    monkeypatch.setattr(storyboard, "keyframe_prompt", keyframe_prompt)
    monkeypatch.setattr(storyboard, "animation_prompt",
                        lambda brief, **kw: PromptPair(positive="slow pan", negative="jitter"))
    job = _paused_job(mgr.store)

    async def go():
        j = mgr.add_scene(job.id, 1, {"description": "the lamp turning at night", "character": "",
                                      "setting": "Atlantic coast", "mood": "triumphant"})
        assert [s.description for s in j.scenes][1] == "the lamp turning at night"
        assert j.scenes[1].writing_prompts is True          # the row appears before Gemma answers
        await asyncio.sleep(0)                               # let the background task run
        for _ in range(50):
            fresh = mgr.store.load_job(job.id)
            if not fresh.scenes[1].writing_prompts:
                return fresh
            await asyncio.sleep(0.02)
        raise AssertionError("prompts were never written")

    fresh = asyncio.run(go())
    new = fresh.scenes[1]
    assert new.keyframe_prompt.startswith("impasto oil painting") and new.animation_prompt == "slow pan"
    assert [s.index for s in fresh.scenes] == [0, 1, 2, 3] and fresh.n_scenes == 4
    assert seen["brief"].description == "the lamp turning at night" and seen["brief"].setting == "Atlantic coast"
    actions = [(d.phase, d.action, d.scene_id) for d in mgr.store.decisions(job.id)]
    assert ("scenes", "added", new.id) in actions and ("scenes", "prompts", new.id) in actions


def test_a_failed_prompt_write_leaves_the_scene_editable(mgr, monkeypatch):
    from vorch import storyboard

    def boom(*a, **kw):
        raise storyboard.StoryboardError("Gemma is down")

    monkeypatch.setattr(storyboard, "keyframe_prompt", boom)
    job = _paused_job(mgr.store)

    async def go():
        mgr.add_scene(job.id, 3, {"description": "a new ending", "character": "", "setting": "", "mood": ""})
        for _ in range(50):
            fresh = mgr.store.load_job(job.id)
            if not fresh.scenes[3].writing_prompts:
                return fresh
            await asyncio.sleep(0.02)
        raise AssertionError("the writing flag was never cleared")

    fresh = asyncio.run(go())
    assert fresh.scenes[3].description == "a new ending" and fresh.scenes[3].keyframe_prompt == ""
    assert any(d.action == "warning" and "Gemma is down" in d.reason for d in mgr.store.decisions(job.id))


def test_delete_scene_renumbers_reopens_the_gate_and_keeps_the_files(mgr, tmp_path):
    kf = tmp_path / "k.png"
    kf.write_bytes(b"x")
    job = _done_video_job(mgr.store)
    job.scenes.append(Scene(index=1, description="filler", keyframe_path=str(kf)))
    mgr.store.save_job(job)

    fresh = mgr.delete_scene(job.id, 1)
    assert [s.index for s in fresh.scenes] == [0] and fresh.n_scenes == 1
    assert kf.exists()                                   # kept unless asked otherwise
    assert fresh.status is JobStatus.NEEDS_APPROVAL and fresh.export is None
    log = [(d.phase, d.action) for d in mgr.store.decisions(job.id)]
    assert ("scenes", "deleted") in log and (GATE_KEYFRAMES, "reopened") in log

    with pytest.raises(KeyError):
        mgr.delete_scene(job.id, 9)


def test_reorder_moves_scenes_and_follows_a_saved_assembly(mgr):
    job = _paused_job(mgr.store, n=3)
    job.assembly = {"spec": {"items": [{"scene_index": 0}, {"scene_index": 1}, {"scene_index": 2}]}}
    mgr.store.save_job(job)
    ids = [s.id for s in job.scenes]

    fresh = mgr.reorder_scenes(job.id, [ids[2], ids[0], ids[1]])
    assert [s.description for s in fresh.scenes] == ["scene 2", "scene 0", "scene 1"]
    assert [s.index for s in fresh.scenes] == [0, 1, 2]
    # The stored cut list still names the same three clips, not the numbers they used to have.
    assert [i["scene_index"] for i in fresh.assembly["spec"]["items"]] == [1, 2, 0]

    with pytest.raises(ValueError, match="every scene exactly once"):
        mgr.reorder_scenes(job.id, ids[:2])


def test_structural_edits_are_refused_while_a_segment_runs(mgr):
    job = _paused_job(mgr.store)
    mgr._active.add(job.id)
    try:
        for call in (lambda: mgr.add_scene(job.id, 0, {"description": "x", "character": "", "setting": "", "mood": ""}),
                     lambda: mgr.delete_scene(job.id, 0),
                     lambda: mgr.reorder_scenes(job.id, [s.id for s in job.scenes]),
                     lambda: mgr.write_prompts(job.id, 0)):
            with pytest.raises(RuntimeError, match="running"):
                call()
    finally:
        mgr._active.discard(job.id)


def test_write_prompts_needs_a_description(mgr):
    job = _paused_job(mgr.store)
    job.scenes[0].description = ""
    mgr.store.save_job(job)
    with pytest.raises(ValueError, match="no description"):
        mgr.write_prompts(job.id, 0)


def test_reanimate_renders_this_scene_while_the_keyframes_gate_waits(mgr, monkeypatch):
    """Animating one scene must not wait for the batch approval.

    It used to clear the clip, walk into the gate and pause — "regenerating video of scene 0"
    followed by nothing rendered.
    """
    from vorch import orchestrator as orch_mod
    from vorch.orchestrator import GATE_KEYFRAMES as KF, Orchestrator
    from vorch.state import Decision

    rendered: list[int] = []
    monkeypatch.setattr(Orchestrator, "_one_video",
                        lambda self, job, scene, **kw: rendered.append(scene.index))
    mgr._factory = lambda store, job: Orchestrator.from_job(store, job)

    job = _done_video_job(mgr.store)
    job.scenes.append(Scene(index=1, keyframe_path="/kf2.png", status=SceneStatus.KEYFRAME))
    job.status = JobStatus.NEEDS_APPROVAL
    mgr.store.save_job(job)
    mgr.store.log_decision(Decision(job_id=job.id, scene_index=None, phase="storyboard", attempt=1, action="approved"))
    mgr.store.log_decision(Decision(job_id=job.id, scene_index=None, phase=KF, attempt=1, action="reopened"))
    assert Orchestrator.pending_gate(mgr.store.load_job(job.id), mgr.store) == KF

    async def go():
        await mgr.regenerate(job.id, 0, what="video", edits={})
        for _ in range(100):
            if not mgr.is_active(job.id):
                return mgr.store.load_job(job.id)
            await asyncio.sleep(0.02)
        raise AssertionError("the segment never finished")

    fresh = asyncio.run(go())
    assert rendered == [0]                                   # that scene, and only that scene
    assert fresh.status is JobStatus.NEEDS_APPROVAL          # scene 2 still needs the gate
    assert Orchestrator.pending_gate(fresh, mgr.store) == KF
    assert orch_mod is not None


def test_animating_the_last_missing_clip_finishes_the_job(mgr, monkeypatch):
    """Otherwise the job would sit at a gate with nothing left for it to render."""
    from vorch.orchestrator import Orchestrator

    def fake_video(self, job, scene, **kw):
        scene.video_path = "/clip/new.mp4"
        scene.status = SceneStatus.COMPLETE

    monkeypatch.setattr(Orchestrator, "_one_video", fake_video)
    mgr._factory = lambda store, job: Orchestrator.from_job(store, job)

    job = _done_video_job(mgr.store)
    job.status = JobStatus.NEEDS_APPROVAL
    mgr.store.save_job(job)

    async def go():
        await mgr.regenerate(job.id, 0, what="video", edits={})
        for _ in range(100):
            if not mgr.is_active(job.id):
                return mgr.store.load_job(job.id)
            await asyncio.sleep(0.02)
        raise AssertionError("the segment never finished")

    fresh = asyncio.run(go())
    assert fresh.status is JobStatus.DONE and fresh.scenes[0].video_path == "/clip/new.mp4"


def test_animate_scene_refuses_what_it_cannot_render(mgr):
    from vorch.orchestrator import Orchestrator

    job = _done_video_job(mgr.store)
    job.scenes.append(Scene(index=1))                     # no keyframe yet
    o = Orchestrator.from_job(mgr.store, job)
    with pytest.raises(ValueError, match="no keyframe"):
        o.animate_scene(job, 1)
    with pytest.raises(KeyError):
        o.animate_scene(job, 9)
    job.kind = "stills"
    with pytest.raises(ValueError, match="no clip"):
        o.animate_scene(job, 0)
