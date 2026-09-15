"""Using an image of the user's choice as a scene's keyframe (regenerate with
``keyframe_asset_id``): fitted into COMFY_OUTPUT, kept as a user variant, never
generated over or evaluated, and gated exactly like a regenerated keyframe."""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from vorch import config, pipeline, profiles, projects, provenance, storyboard
from vorch.orchestrator import GATE_KEYFRAMES, Orchestrator
from vorch.queue import JobManager
from vorch.state import Decision, JobStatus, Scene, SceneStatus, Variant, new_job
from vorch.storyboard import PromptPair, SceneBrief

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _dims(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", str(path)]).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


@pytest.fixture
def env(make_store, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path / "out")
    store = make_store()
    p = projects.new_project("Bela")
    src = tmp_path / "chocolate.jpg"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=brown:s=900x1200", "-frames:v", "1", str(src)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        src.write_bytes(b"\xff\xd8\xff")
    asset = projects.add_asset(p, src, kind="reference", name="chocolate.jpg", label="chocolate on a wall")
    projects.save(store, p)
    yield store, p, asset
    store.close()


def _prof(**over):
    base = {"evaluate": {"enabled": False}, "face_swap": {"enabled": False}}
    base.update(over)
    return profiles.apply_overrides(profiles.default_profile(), base)


def _done_video_job(store, project):
    o = Orchestrator(store, profile=_prof())
    job = o.create_job("idea", project_id=project.id, autonomy="semi")
    job.status = JobStatus.DONE
    job.scenes = [Scene(index=0, keyframe_path="/kf.png", variants=[Variant("/kf.png", seed=4)],
                        raw_video_path="/raw.mp4", video_path="/clip.mp4", status=SceneStatus.COMPLETE)]
    store.save_job(job)
    for gate in ("storyboard", GATE_KEYFRAMES):
        store.log_decision(Decision(job_id=job.id, scene_index=None, phase=gate, attempt=1, action="approved"))
    return o, job


@needs_ffmpeg
def test_chosen_image_becomes_the_keyframe_at_the_jobs_frame_size(env):
    store, project, asset = env
    o, job = _done_video_job(store, project)
    o.reopen_scene(job, 0, what="keyframe", edits={"keyframe_asset_id": asset.id})
    s = job.scenes[0]

    assert s.keyframe_path.startswith(str(config.COMFY_OUTPUT / "vorch" / job.id[:8]))  # served by /media
    assert _dims(s.keyframe_path) == o.params.keyframe_size  # portrait photo cover-cropped to 16:9
    assert [v.source for v in s.variants] == ["generated", "user"]  # the generated one stays selectable
    assert s.variants[-1].asset_id == asset.id and s.keyframe_verdict is None
    assert (s.video_path, s.raw_video_path, s.status) == (None, None, SceneStatus.KEYFRAME)
    assert job.export is None
    m = provenance.read_manifest(s.keyframe_path)
    assert m["step"] == "user_keyframe" and m["job_id"] == job.id and m["params"]["asset_id"] == asset.id
    actions = [(d.phase, d.action) for d in store.decisions(job.id)]
    assert ("keyframe", "user_keyframe") in actions and (GATE_KEYFRAMES, "reopened") in actions


@needs_ffmpeg
def test_the_run_after_it_pauses_for_review_then_renders_only_that_clip(env, monkeypatch):
    store, project, asset = env
    o, job = _done_video_job(store, project)
    generated, animated = [], []
    monkeypatch.setattr(pipeline, "_fresh_keyframe", lambda *a, **k: generated.append(1) or ("/new.png", None))
    monkeypatch.setattr(pipeline, "animate_frame", lambda frame, *a, **k: animated.append(str(frame)) or "/v2.mp4")

    o.regenerate(job, 0, what="keyframe", edits={"keyframe_asset_id": asset.id})
    assert job.status is JobStatus.NEEDS_APPROVAL and generated == [] and animated == []
    o.approve(job, GATE_KEYFRAMES)
    assert job.status is JobStatus.DONE and generated == []
    assert animated == [job.scenes[0].keyframe_path] and job.scenes[0].video_path == "/v2.mp4"


@needs_ffmpeg
def test_a_stills_job_with_variants_never_generates_over_the_chosen_image(env, monkeypatch):
    store, project, asset = env
    o = Orchestrator(store, profile=profiles.apply_overrides(profiles.builtin_profiles()["stills_16x9"], {
        "evaluate": {"enabled": True}, "face_swap": {"enabled": False}}))
    job = o.create_job("idea", project_id=project.id, autonomy="full")
    job.scenes = [Scene(index=0, description="d", keyframe_prompt="kf")]
    store.save_job(job)
    for gate in ("storyboard",):
        store.log_decision(Decision(job_id=job.id, scene_index=None, phase=gate, attempt=1, action="approved"))
    judged = []
    monkeypatch.setattr(pipeline, "generate_keyframe_evaluated", lambda *a, **k: judged.append(1) or ("/g.png", None))
    monkeypatch.setattr(pipeline, "_fresh_keyframe", lambda *a, **k: judged.append(1) or ("/g.png", None))

    o.regenerate(job, 0, what="keyframe", edits={"keyframe_asset_id": asset.id})
    assert judged == []  # 4 variants per scene, yet nothing generated and nothing evaluated
    s = job.scenes[0]
    assert [v.source for v in s.variants] == ["user"] and job.status is JobStatus.DONE

    # A plain regenerate goes back to generating.
    o.reopen_scene(job, 0, what="keyframe")
    assert job.scenes[0].variants == [] and job.scenes[0].status is SceneStatus.PENDING


def test_bad_image_choices_are_rejected_before_the_job_starts(env):
    store, project, asset = env
    _o, job = _done_video_job(store, project)
    mgr = JobManager(store, orchestrator_factory=lambda s, j: pytest.fail("must not start"), auto_export=False)

    async def go():
        with pytest.raises(ValueError, match="unknown project asset"):
            await mgr.regenerate(job.id, 0, what="keyframe", edits={"keyframe_asset_id": "nope"})
        with pytest.raises(ValueError, match="what='keyframe'"):
            await mgr.regenerate(job.id, 0, what="video", edits={"keyframe_asset_id": asset.id})

    asyncio.run(go())
    no_project = new_job(idea="x")
    no_project.status = JobStatus.DONE
    no_project.scenes = [Scene(index=0)]
    store.save_job(no_project)
    with pytest.raises(ValueError, match="no project"):
        asyncio.run(mgr.regenerate(no_project.id, 0, what="keyframe", edits={"keyframe_asset_id": asset.id}))
    assert store.load_job(job.id).status is JobStatus.DONE and not mgr.is_active(job.id)


def test_scene_public_reports_a_user_keyframe(env):
    from vorch.api.public import scene_to_public

    s = Scene(index=0, keyframe_path="/u.png", variants=[Variant("/g.png"), Variant("/u.png", source="user", asset_id="a1")])
    out = scene_to_public(s)
    assert out.keyframe_source == "user" and out.keyframe_asset_id == "a1"
    assert [v.source for v in out.variants] == ["generated", "user"]
    s.keyframe_path = "/g.png"  # switched back to the generated candidate
    assert scene_to_public(s).keyframe_source == "generated"
