"""vorch/assembly.py: defaults, spec resolution/validation, runner, record roundtrip."""

from __future__ import annotations

import pathlib

import pytest

from vorch import assembly
from vorch import titles as titles_mod, config, media
from vorch.media import Cut, MediaInfo
from vorch.state import Scene, new_job


def _job(tmp_path, n=3, kind="video", raw_for=()):
    job = new_job(idea="asm", kind=kind, profile={"video": {"width": 832, "height": 480, "fps": 16, "length": 49}})
    scenes = []
    for i in range(n):
        p = tmp_path / f"s{i}.mp4"
        p.write_bytes(b"v")
        raw = None
        if i in raw_for:
            raw = tmp_path / f"s{i}_raw.mp4"
            raw.write_bytes(b"r")
        scenes.append(Scene(index=i, video_path=str(p), raw_video_path=str(raw) if raw else None))
    job.scenes = scenes
    return job


@pytest.fixture(autouse=True)
def _fake_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path / "out")
    monkeypatch.setattr(media, "transitions", lambda: ("none", "fade", "wipeleft"))
    monkeypatch.setattr(media, "probe", lambda p: MediaInfo(832, 480, 16.0, 3.0625, 49, "h264", False))


def test_default_spec_all_clips_fade(tmp_path):
    job = _job(tmp_path)
    spec = assembly.default_spec(job)
    assert [i.scene_index for i in spec.items] == [0, 1, 2]
    assert spec.cuts == [Cut("fade", 0.5), Cut("fade", 0.5)]
    assert assembly.output_path(job) == tmp_path / "out" / "vorch" / job.id[:8] / "assembly.mp4"
    st = assembly.status(job)
    assert st["available"] and st["transitions"][0] == "none" and st["profile_fps"] == 16
    assert [c["duration"] for c in st["clips"]] == [3.062] * 3 and st["record"] is None


def test_resolve_merges_partial_spec_and_validates(tmp_path):
    job = _job(tmp_path, raw_for=(1,))
    spec, paths = assembly.resolve(job, None)
    assert len(paths) == 3 and spec.encoder == "libx264"
    spec, paths = assembly.resolve(job, {"items": [{"scene_index": 2}, {"scene_index": 1, "source": "raw"}],
                                         "cuts": [{"transition": "none"}], "fps": 24, "crf": 20})
    assert [p.name for p in paths] == ["s2.mp4", "s1_raw.mp4"] and spec.fps == 24 and spec.crf == 20
    assert spec.cuts == [Cut("none", 0.5)]
    for bad, msg in [
        ({"items": []}, "at least one"),
        ({"items": [{"scene_index": 9}]}, "no scene 10"),
        ({"items": [{"scene_index": 0, "clip_id": "ghost"}]}, "no take"),
        ({"items": [{"scene_index": 0, "source": "weird"}]}, "source"),
        ({"cuts": [{"transition": "fade"}]}, "need 2 cut"),
        ({"cuts": [{"transition": "swoosh"}, {"transition": "fade"}]}, "unknown transition"),
        ({"cuts": [{"transition": "fade", "duration": 9}, {"transition": "fade"}]}, "within 0 and"),
        ({"cuts": [{"transition": "fade", "duration": 2.9}, {"transition": "fade"}]}, "exceed"),
        ({"cuts": [{"transition": "fade", "duration": 3.5}, {"transition": "none"}]}, "shorter than both"),
        ({"crf": 99}, "crf"),
        ({"encoder": "magic"}, "encoder"),
        ({"fps": -1}, "fps"),
    ]:
        with pytest.raises(ValueError, match=msg):
            assembly.resolve(job, bad)
    stills = _job(tmp_path, kind="stills")
    with pytest.raises(ValueError, match="only video"):
        assembly.resolve(stills, None)
    empty = new_job(idea="e")
    with pytest.raises(ValueError, match="no clips"):
        assembly.resolve(empty, None)


def test_run_records_failure_without_raising(tmp_path, monkeypatch):
    job = _job(tmp_path)
    spec, paths = assembly.resolve(job, None)
    events = []

    def boom(*a, **k):
        raise media.MediaError("ffmpeg assembly failed: xfade needs equal sizes")

    monkeypatch.setattr(media, "assemble", boom)
    rec = assembly.run(job, spec, paths, emit=events.append)
    assert rec.ok is False and "xfade" in rec.error and rec.n_clips == 3
    assert events[0]["type"] == "step" and events[0]["step"] == "assemble" and events[0]["seconds"] == pytest.approx(8.19, abs=0.01)
    assert events[0]["fps"] == 16.0 and (events[0]["width"], events[0]["height"]) == (832, 480)


def test_run_success_records_and_writes_sidecar(tmp_path, monkeypatch):
    job = _job(tmp_path)
    spec, paths = assembly.resolve(job, {"cuts": [{"transition": "none"}, {"transition": "wipeleft", "duration": 0.3}]})
    seen = {}

    def fake_assemble(clips, cuts, dest, **kw):
        seen.update(clips=[c.name for c in clips], cuts=cuts, kw=kw)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")
        return dest

    monkeypatch.setattr(media, "assemble", fake_assemble)
    rec = assembly.run(job, spec, paths)
    assert rec.ok and rec.path.endswith("/assembly_v1.mp4") and rec.version == 1 and rec.size_bytes == 3 and rec.n_clips == 3
    assert assembly.output_path(job).read_bytes() == b"mp4"  # current = hardlink of v1
    assert seen["cuts"] == [Cut("none", 0.5), Cut("wipeleft", 0.3)] and seen["kw"]["size"] == (832, 480) and seen["kw"]["fps"] == 16.0
    from vorch import provenance
    m = provenance.read_manifest(rec.path)
    assert m["kind"] == "assembly" and m["params"]["spec"]["cuts"][1]["transition"] == "wipeleft"
    assembly.register_version(job, rec)
    st = assembly.status(job, active=False)
    assert st["record"]["exists"] is True and st["record"]["exported"] is False
    assert [v["version"] for v in st["versions"]] == [1] and st["versions"][0]["current"] is True


def test_record_and_spec_roundtrip():
    spec = assembly.AssemblySpec(items=[assembly.AssemblyItem(0), assembly.AssemblyItem(2, "raw")],
                                 cuts=[Cut("fade", 0.4)], fps=24, crf=17)
    assert assembly.AssemblySpec.from_dict(spec.to_dict()) == spec
    rec = assembly.AssemblyRecord(path="/a.mp4", created_at="t", duration=5.2, n_clips=2, spec=spec.to_dict())
    assert assembly.AssemblyRecord.from_dict(rec.to_dict()) == rec
    assert assembly.AssemblyRecord.from_dict({"path": "/x", "ok": False, "error": "e"}).error == "e"


def _fake_assemble_factory(monkeypatch):
    calls = {"n": 0}

    def fake_assemble(clips, cuts, dest, **kw):
        calls["n"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f"v{calls['n']}:{cuts[0].transition}".encode())
        return dest

    monkeypatch.setattr(media, "assemble", fake_assemble)
    return calls


def test_versions_accumulate_select_and_delete(tmp_path, monkeypatch):
    job = _job(tmp_path)
    _fake_assemble_factory(monkeypatch)
    for tr in ("fade", "wipeleft", "none"):
        spec, paths = assembly.resolve(job, {"cuts": [{"transition": tr, "duration": 0.5}, {"transition": "fade", "duration": 0.5}]})
        rec = assembly.run(job, spec, paths)
        assembly.register_version(job, rec)
    versions = assembly.versions_of(job)
    assert [v["version"] for v in versions] == [1, 2, 3]
    assert job.assembly["version"] == 3 and job.assembly["path"].endswith("assembly_v3.mp4")
    assert assembly.output_path(job).read_bytes() == b"v3:none"
    assert all(pathlib.Path(v["path"]).is_file() for v in versions)  # every version keeps its file

    # Select v1 -> current file + record switch, versions list intact.
    assembly.select_version(job, 1)
    assert job.assembly["version"] == 1 and assembly.output_path(job).read_bytes() == b"v1:fade"
    assert [v["version"] for v in assembly.versions_of(job)] == [1, 2, 3]
    st = assembly.status(job)
    assert [v["current"] for v in st["versions"]] == [True, False, False]
    with pytest.raises(KeyError):
        assembly.select_version(job, 9)

    # Delete a non-current version: file gone, current untouched.
    assembly.delete_version(job, 2)
    assert not (tmp_path / "out" / "vorch" / job.id[:8] / "assembly_v2.mp4").exists()
    assert job.assembly["version"] == 1 and [v["version"] for v in assembly.versions_of(job)] == [1, 3]
    # Delete the current one: newest remaining (3) becomes current.
    assembly.delete_version(job, 1)
    assert job.assembly["version"] == 3 and assembly.output_path(job).read_bytes() == b"v3:none"
    # Delete the last: nothing left.
    assembly.delete_version(job, 3)
    assert job.assembly is None and not assembly.output_path(job).exists()
    with pytest.raises(KeyError):
        assembly.delete_version(job, 3)


def test_failed_run_does_not_become_a_version(tmp_path, monkeypatch):
    job = _job(tmp_path)
    _fake_assemble_factory(monkeypatch)
    spec, paths = assembly.resolve(job, None)
    assembly.register_version(job, assembly.run(job, spec, paths))
    monkeypatch.setattr(media, "assemble", lambda *a, **k: (_ for _ in ()).throw(media.MediaError("boom")))
    bad = assembly.run(job, spec, paths)
    assembly.register_version(job, bad)
    assert job.assembly["ok"] is False and job.assembly["version"] == 2
    assert [v["version"] for v in assembly.versions_of(job)] == [1]   # failures are not kept as versions
    assert assembly.next_version(job) == 2                             # the number is reused


def test_resolve_derives_speed_from_target_and_validates(tmp_path):
    job = _job(tmp_path)
    spec, paths = assembly.resolve(job, {
        "items": [{"scene_index": 0, "target_seconds": 6.125}, {"scene_index": 1}, {"scene_index": 2, "target_seconds": 1.53125}],
        "cuts": [{"transition": "fade"}, {"transition": "fade"}], "smooth": True,
    })
    assert spec.items[0].speed == pytest.approx(0.5) and spec.items[1].speed == 1.0 and spec.items[1].target_seconds is None
    assert spec.items[2].speed == pytest.approx(2.0) and spec.smooth is True
    assert spec.speeds == pytest.approx([0.5, 1.0, 2.0])
    for bad, msg in [
        ({"items": [{"scene_index": 0, "target_seconds": 20}]}, "outside 0.77–12.25 s"),
        ({"items": [{"scene_index": 0, "target_seconds": 0}]}, "positive"),
        # 2x speed leaves a 1.53 s clip: a 1.5 s fade must fail against the RETIMED length
        ({"items": [{"scene_index": 0, "target_seconds": 1.53125}, {"scene_index": 1}],
          "cuts": [{"transition": "fade", "duration": 1.5}]}, "exceed|shorter than both"),
    ]:
        with pytest.raises(ValueError, match=msg):
            assembly.resolve(job, bad)


def test_run_passes_speeds_and_reports_retimed_total(tmp_path, monkeypatch):
    job = _job(tmp_path)
    spec, paths = assembly.resolve(job, {
        "items": [{"scene_index": 0, "target_seconds": 6.125}, {"scene_index": 1}, {"scene_index": 2}],
        "cuts": [{"transition": "fade"}, {"transition": "none"}], "smooth": True,
    })
    seen, events = {}, []

    def fake_assemble(clips, cuts, dest, **kw):
        seen.update(kw)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")
        return dest

    monkeypatch.setattr(media, "assemble", fake_assemble)
    rec = assembly.run(job, spec, paths, emit=events.append)
    assert rec.ok and seen["speeds"] == pytest.approx([0.5, 1, 1]) and seen["smooth"] is True
    assert events[0]["seconds"] == pytest.approx(6.125 + 3.0625 + 3.0625 - 0.5, abs=0.01)
    assert events[0]["smooth"] is True and events[0]["retimed"] == 1
    assert rec.spec["items"][0]["target_seconds"] == 6.125 and rec.spec["smooth"] is True
    round_trip = assembly.AssemblySpec.from_dict(rec.spec)
    assert round_trip.items[0].speed == pytest.approx(0.5) and round_trip.smooth is True


def _assembled(tmp_path, monkeypatch, job):
    spec, paths = assembly.resolve(job, {})

    def fake_assemble(clips, cuts, dest, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")
        return dest

    monkeypatch.setattr(media, "assemble", fake_assemble)
    rec = assembly.run(job, spec, paths)
    assembly.register_version(job, rec)
    return rec


def test_assembly_goes_stale_when_a_joined_clip_changes(tmp_path, monkeypatch):
    job = _job(tmp_path)
    rec = _assembled(tmp_path, monkeypatch, job)
    assert rec.clips == [s.video_path for s in job.scenes]
    assert assembly.status(job)["record"]["stale"] is False

    new_clip = tmp_path / "s1_regenerated.mp4"
    new_clip.write_bytes(b"v2")
    job.scenes[1].video_path = str(new_clip)
    assert assembly.status(job)["record"]["stale"] is True

    job.scenes[1].video_path = None  # clip cleared by an edit, not rendered yet
    assert assembly.is_stale(job, assembly.AssemblyRecord.from_dict(job.assembly)) is True


def test_old_records_without_clips_fall_back_to_the_sidecar(tmp_path, monkeypatch):
    job = _job(tmp_path)
    _assembled(tmp_path, monkeypatch, job)
    job.assembly.pop("clips")  # a record written before clips were stored
    assert assembly.status(job)["record"]["stale"] is False
    job.scenes[0].video_path = str(tmp_path / "other.mp4")
    assert assembly.status(job)["record"]["stale"] is True


def test_unknown_clips_or_failed_records_are_never_flagged(tmp_path):
    job = _job(tmp_path)
    rec = assembly.AssemblyRecord(path=str(tmp_path / "gone.mp4"), created_at="", spec={"items": [{"scene_index": 0}]})
    assert assembly.is_stale(job, rec) is False  # no clips and no sidecar: cannot know
    failed = assembly.AssemblyRecord(path="x", created_at="", ok=False, clips=["/nope"], spec={"items": [{"scene_index": 0}]})
    assert assembly.is_stale(job, failed) is False


# --- delivery (what the assembly is mastered as) ---------------------------------------


def _job_profile(tmp_path, w, h, fps=16, purpose="cinematic", n=2, aspect="16:9"):
    job = _job(tmp_path, n=n)
    job.profile = {"purpose": purpose, "aspect": aspect,
                   "video": {"width": w, "height": h, "fps": fps, "length": 49}}
    if aspect is None:
        job.profile.pop("aspect")
    return job


def test_delivery_targets_follow_the_jobs_aspect(tmp_path):
    landscape = {d.id: d for d in assembly.deliveries(_job_profile(tmp_path, 832, 480))}
    assert (landscape["master_1080"].width, landscape["master_1080"].height) == (1920, 1080)
    assert landscape["master_1080"].fps == 24 and landscape["youtube_1080"].fps == 30
    assert landscape["source"].width is None  # source keeps whatever the clips are

    vertical = {d.id: d for d in assembly.deliveries(_job_profile(tmp_path, 480, 832, aspect="9:16"))}
    assert (vertical["social_1080"].width, vertical["social_1080"].height) == (1080, 1920)
    assert "youtube_1080" not in vertical  # a landscape target for a vertical film is noise

    square = {d.id: d for d in assembly.deliveries(_job_profile(tmp_path, 640, 640, aspect="1:1"))}
    assert (square["master_1080"].width, square["master_1080"].height) == (1080, 1080)

    # No declared aspect (a legacy job): fall back to the clip's own ratio.
    legacy = {d.id: d for d in assembly.deliveries(_job_profile(tmp_path, 832, 480, aspect=None))}
    assert (legacy["master_1080"].width, legacy["master_1080"].height) == (1920, 1108)


def test_social_jobs_default_to_a_social_target_and_others_to_a_master(tmp_path):
    assert assembly.delivery_default(_job_profile(tmp_path, 480, 832, purpose="social_vertical",
                                                  aspect="9:16")) == "social_1080"
    assert assembly.delivery_default(_job_profile(tmp_path, 832, 480)) == "master_1080"
    bare = _job(tmp_path)
    bare.profile = None
    assert assembly.delivery_default(bare) == "source"   # nothing to resolve an aspect from


def test_target_format_resolves_a_delivery_and_an_explicit_size_still_wins(tmp_path):
    job = _job_profile(tmp_path, 480, 832, aspect="9:16")
    first = MediaInfo(480, 832, 16.0, 3.0, 49, "h264", False)

    spec, _ = assembly.resolve(job, {"delivery": "social_1080"})
    assert assembly.target_format(job, spec, first) == (30.0, 1080, 1920)

    spec, _ = assembly.resolve(job, {"delivery": "social_1080", "width": 720, "height": 1280, "fps": 25})
    assert assembly.target_format(job, spec, first) == (25.0, 720, 1280)

    spec, _ = assembly.resolve(job, {})            # unchanged default: the clips' own format
    assert assembly.target_format(job, spec, first) == (16.0, 480, 832)

    with pytest.raises(ValueError, match="unknown delivery"):
        assembly.resolve(job, {"delivery": "imax"})


def test_status_offers_the_targets_with_the_default(tmp_path):
    st = assembly.status(_job_profile(tmp_path, 832, 480))
    assert [d["id"] for d in st["deliveries"]] == ["source", "master_1080", "social_1080", "youtube_1080"]
    assert st["delivery"] == "master_1080"


def test_the_record_and_spec_carry_the_mastering_settings(tmp_path, monkeypatch):
    job = _job_profile(tmp_path, 832, 480)
    spec, paths = assembly.resolve(job, {"delivery": "master_1080", "grain": 1.5, "max_mbps": 25})
    assert spec.to_dict()["delivery"] == "master_1080" and spec.to_dict()["grain"] == 1.5
    seen = {}

    def fake_assemble(clips, cuts, dest, **kw):
        seen.update(kw)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")
        return dest

    monkeypatch.setattr(media, "assemble", fake_assemble)
    assembly.run(job, spec, paths)
    assert seen["size"] == (1920, 1080) and seen["fps"] == 24.0
    assert seen["grain"] == 1.5 and seen["max_mbps"] == 25


# --- takes, repeats and trims ------------------------------------------------------------


def _with_takes(tmp_path, job):
    """Give scene 0 a second, older take on disk."""
    from vorch.state import Clip
    old = tmp_path / "s0_take1.mp4"
    old.write_bytes(b"v")
    job.scenes[0].clips = [Clip(video_path=str(old), id="take1"),
                           Clip(video_path=job.scenes[0].video_path, id="take2")]
    return old


def test_a_scene_may_be_placed_twice(tmp_path):
    """Two takes of one shot joined make a longer shot; the same clip twice is a repeat. Nothing
    downstream ever cared — the list is positional — so only the old guard forbade it."""
    job = _job(tmp_path)
    spec, paths = assembly.resolve(job, {"items": [{"scene_index": 0}, {"scene_index": 1}, {"scene_index": 0}],
                                         "cuts": [{"transition": "none"}, {"transition": "none"}]})
    assert [i.scene_index for i in spec.items] == [0, 1, 0]
    assert [p.name for p in paths] == ["s0.mp4", "s1.mp4", "s0.mp4"]


def test_a_take_is_chosen_by_id_and_an_unknown_one_is_refused(tmp_path):
    job = _job(tmp_path)
    old = _with_takes(tmp_path, job)
    spec, paths = assembly.resolve(job, {"items": [{"scene_index": 0, "clip_id": "take1"},
                                                   {"scene_index": 0, "clip_id": "take2"}],
                                         "cuts": [{"transition": "none"}]})
    assert [p.name for p in paths] == [old.name, "s0.mp4"]      # the older take, then the current one
    assert [i.clip_id for i in spec.items] == ["take1", "take2"]
    with pytest.raises(ValueError, match="clip 1 .scene 1.: no take"):
        assembly.resolve(job, {"items": [{"scene_index": 0, "clip_id": "nope"}]})


def test_a_pinned_take_does_not_go_stale_when_the_scene_is_re_rendered(tmp_path):
    """The take was named explicitly, so a newer render of that scene has not replaced it."""
    job = _job(tmp_path)
    _with_takes(tmp_path, job)
    rec = assembly.AssemblyRecord(
        path=str(tmp_path / "a.mp4"), created_at="", ok=True, clips=[str(tmp_path / "s0_take1.mp4"), str(tmp_path / "s1.mp4")],
        spec={"items": [{"scene_index": 0, "clip_id": "take1"}, {"scene_index": 1}]})
    assert assembly.is_stale(job, rec) is False
    job.scenes[1].video_path = str(tmp_path / "s1_new.mp4")     # scene 1 re-rendered, and it is not pinned
    assert assembly.is_stale(job, rec) is True


def test_a_trim_shortens_the_clip_and_is_snapped_to_its_frame_grid(tmp_path):
    """A person types milliseconds; a cut can only land on a frame. 16 fps = 62.5 ms."""
    job = _job(tmp_path)
    spec, _ = assembly.resolve(job, {"items": [{"scene_index": 0, "start": 1.234, "end": 2.0}],
                                     "cuts": []})
    assert (spec.items[0].start, spec.items[0].end) == (1.25, 2.0)   # snapped, and stored snapped
    # The kept length is what a target duration is measured against: 0.75 s, not the full 3.06 s.
    with pytest.raises(ValueError, match="outside"):
        assembly.resolve(job, {"items": [{"scene_index": 0, "start": 1.25, "end": 2.0, "target_seconds": 9}],
                               "cuts": []})
    with pytest.raises(ValueError, match="less than one frame"):
        assembly.resolve(job, {"items": [{"scene_index": 0, "start": 1.0, "end": 1.01}], "cuts": []})


def test_a_spec_saved_before_takes_and_trims_loads_unchanged(tmp_path):
    item = assembly._item_from_dict({"scene_index": 2, "source": "raw", "target_seconds": 6.0})
    assert (item.clip_id, item.start, item.end) == (None, None, None)
    assert item.scene_index == 2 and item.source == "raw" and item.target_seconds == 6.0


# --- keeping a cut ------------------------------------------------------------------------


@pytest.fixture
def _real_clip(tmp_path, monkeypatch):
    """A genuine 4 s file, since save_cut actually renders."""
    import shutil
    import subprocess
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg not installed")
    monkeypatch.undo()                       # the autouse fake probe would lie about the file
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path / "out")
    src = tmp_path / "real.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=64x64:rate=16:duration=4", "-pix_fmt", "yuv420p", str(src)], check=True)
    return src


def test_a_cut_becomes_a_take_and_can_become_the_scene_s_clip(tmp_path, _real_clip):
    """A trim in a timeline reaches nothing but assembly.mp4. Saved, it is a clip like any other —
    which is what lets the export folder carry the cut instead of the whole render."""
    job = new_job(idea="asm", kind="video", profile={"video": {"width": 64, "height": 64, "fps": 16}})
    job.scenes = [Scene(index=0, video_path=str(_real_clip), animation_prompt="drift")]

    clip = assembly.save_cut(job, 0, start=1.234, end=2.0)
    assert media.probe(clip.video_path).frame_duration == pytest.approx(0.75, abs=0.03)  # snapped 1.25
    assert clip in job.scenes[0].clips
    assert job.scenes[0].video_path == str(_real_clip), "the scene's own clip is untouched by default"
    assert clip.prompt == "drift"                    # the take carries what made the footage

    second = assembly.save_cut(job, 0, start=0.0, end=1.0, make_active=True)
    assert second.video_path != clip.video_path      # a second cut does not overwrite the first
    assert job.scenes[0].video_path == second.video_path, "make_active makes it the deliverable"
    assert len(job.scenes[0].clips) == 2


def test_a_saved_cut_can_be_cut_again_and_refuses_the_impossible(tmp_path, _real_clip):
    job = new_job(idea="asm", kind="video", profile={"video": {"width": 64, "height": 64, "fps": 16}})
    job.scenes = [Scene(index=0, video_path=str(_real_clip))]
    first = assembly.save_cut(job, 0, start=0.0, end=2.0)
    again = assembly.save_cut(job, 0, clip_id=first.id, start=0.5, end=1.0)
    assert media.probe(again.video_path).frame_duration == pytest.approx(0.5, abs=0.03)

    with pytest.raises(ValueError, match="nothing is trimmed"):
        assembly.save_cut(job, 0, start=None, end=None)
    with pytest.raises(ValueError, match="no take"):
        assembly.save_cut(job, 0, clip_id="ghost", start=0.0, end=1.0)
    with pytest.raises(KeyError):
        assembly.save_cut(job, 7, start=0.0, end=1.0)


# --- narration must not silently replace a clip's own audio -----------------------------


def test_audio_defaults_off_for_a_silent_timeline(tmp_path):
    """Every Wan clip is silent, so keeping audio would only add an empty track."""
    spec = assembly.default_spec(_job(tmp_path))
    assert spec.audio is False
    assert spec.narration is True
    assert all(c["has_audio"] is False for c in assembly.status(_job(tmp_path))["clips"])


def test_audio_defaults_on_when_a_clip_carries_a_track(tmp_path, monkeypatch):
    """The reported bug: an imported clip brings its own sound, narration was on by default, and
    `audio` was off — so the voice replaced the original instead of sitting over it."""
    monkeypatch.setattr(media, "probe",
                        lambda p: MediaInfo(832, 480, 16.0, 3.0625, 49, "h264", True))
    job = _job(tmp_path)
    assert assembly.any_clip_has_audio(job) is True
    assert assembly.default_spec(job).audio is True
    assert all(c["has_audio"] is True for c in assembly.status(job)["clips"])


def test_an_unreadable_clip_counts_as_silent(tmp_path, monkeypatch):
    """A guess must not switch audio on by accident."""
    def boom(p):
        raise media.MediaError("no such file")

    monkeypatch.setattr(media, "probe", boom)
    assert assembly.any_clip_has_audio(_job(tmp_path)) is False


def test_narration_is_chosen_by_the_take_that_exists_not_the_text(tmp_path):
    """Deliberate — an edited but unspoken line must still be heard as the take on disk. It is also
    why removing a take needs its own action: emptying the words cannot silence it."""
    job = _job(tmp_path, n=2)
    wav = tmp_path / "s0.wav"
    wav.write_bytes(b"RIFF")
    job.scenes[0].narration = ""                  # the words are gone
    job.scenes[0].narration_path = str(wav)       # the take is not
    spec = assembly.default_spec(job)
    voices = assembly._narration_for(job, spec, [3.0625, 3.0625])
    assert [p for p, _ in voices] == [wav]
    # Clearing the path is what removes it from the mix.
    job.scenes[0].narration_path = None
    assert assembly._narration_for(job, spec, [3.0625, 3.0625]) == []


# --- captions on the finished film -------------------------------------------------------


def _caption(**kw) -> dict:
    return {"text": "Открий своя аромат", "start": 0.0, "duration": 2.0, **kw}


def test_the_spec_round_trips_its_captions():
    """Guard one of the three sites that enumerate keys explicitly and drop anything unnamed."""
    spec = assembly.AssemblySpec(items=[], cuts=[], texts=[_caption()], titles=False)
    back = assembly.AssemblySpec.from_dict(spec.to_dict())
    assert back.texts == spec.texts and back.titles is False


def test_resolve_keeps_captions_the_api_sent(tmp_path):
    """The other site: `resolve()` builds the spec field by field, and it is the path the API and
    the CLI actually use — a round-trip test alone passes with this one missing."""
    job = _job(tmp_path, n=2)
    spec, _ = assembly.resolve(job, {"items": [{"scene_index": 0}, {"scene_index": 1}],
                                     "cuts": [{"transition": "fade"}],
                                     "texts": [_caption()], "titles": True})
    assert [t["text"] for t in spec.texts] == ["Открий своя аромат"]
    assert spec.titles is True
    assert assembly.resolve(job, {"titles": False})[0].titles is False


def test_a_timeline_caption_past_the_end_is_refused(tmp_path):
    """Authored against the finished length, so running past it is a spec error — unlike a
    scene's caption, which is clamped."""
    job = _job(tmp_path, n=2)
    with pytest.raises(ValueError, match="caption 1: .*past the end"):
        assembly.resolve(job, {"texts": [_caption(start=99.0)]})


def test_a_malformed_caption_is_refused_with_its_position(tmp_path):
    job = _job(tmp_path, n=2)
    with pytest.raises(ValueError, match="caption 2: .*unknown"):
        assembly.resolve(job, {"texts": [_caption(), _caption(effect="explode")]})


def test_too_many_captions_are_refused(tmp_path):
    job = _job(tmp_path, n=2)
    with pytest.raises(ValueError, match="at most"):
        assembly.resolve(job, {"texts": [_caption() for _ in range(titles_mod.MAX_ITEMS + 1)]})


# --- placing a scene's caption ------------------------------------------------------------


def test_a_scenes_caption_is_placed_through_the_transition_arithmetic(tmp_path):
    """The whole point: a caption cannot drift from the shot it annotates."""
    job = _job(tmp_path, n=3)
    job.scenes[1].texts = [_caption(start=1.2, duration=1.0)]
    spec, _ = assembly.resolve(job, {"items": [{"scene_index": i} for i in range(3)],
                                     "cuts": [{"transition": "fade"}, {"transition": "fade"}]})
    durations = [3.0625] * 3
    placed = assembly._texts_for(job, spec, durations)
    offsets = media.item_offsets(durations, spec.cuts, spec.speeds)
    assert len(placed) == 1
    assert placed[0].start == pytest.approx(offsets[1] + 1.2)
    assert placed[0].duration == pytest.approx(1.0)


def test_a_retimed_scene_carries_its_caption_with_it(tmp_path):
    """Start and duration both divide by the speed, so the caption stays glued to its frames."""
    job = _job(tmp_path, n=2)
    job.scenes[0].texts = [_caption(start=1.2, duration=1.0)]
    spec, _ = assembly.resolve(job, {
        "items": [{"scene_index": 0, "target_seconds": 3.0625 / 2}, {"scene_index": 1}],
        "cuts": [{"transition": "fade"}]})
    assert spec.items[0].speed == pytest.approx(2.0)
    placed = assembly._texts_for(job, spec, [3.0625, 3.0625])
    assert placed[0].start == pytest.approx(1.2 / 2)
    assert placed[0].duration == pytest.approx(0.5)


def test_a_scene_placed_twice_is_captioned_twice(tmp_path):
    job = _job(tmp_path, n=2)
    job.scenes[0].texts = [_caption(duration=1.0)]
    spec, _ = assembly.resolve(job, {"items": [{"scene_index": 0}, {"scene_index": 0}],
                                     "cuts": [{"transition": "fade"}]})
    placed = assembly._texts_for(job, spec, [3.0625, 3.0625])
    assert len(placed) == 2 and placed[0].start < placed[1].start


def test_a_scenes_caption_is_clamped_to_its_clip_not_refused(tmp_path):
    """A trim gets retuned constantly; failing a whole render over 0.3s is the wrong trade."""
    job = _job(tmp_path, n=1)
    job.scenes[0].texts = [_caption(start=2.5, duration=5.0), _caption(start=99.0)]
    spec, _ = assembly.resolve(job, {"items": [{"scene_index": 0}], "cuts": []})
    placed = assembly._texts_for(job, spec, [3.0625])
    assert len(placed) == 1, "the one starting past the clip is dropped, not raised"
    assert placed[0].duration == pytest.approx(3.0625 - 2.5), "the overrun is trimmed"


def test_the_titles_switch_renders_a_clean_pass(tmp_path):
    job = _job(tmp_path, n=1)
    job.scenes[0].texts = [_caption()]
    spec, _ = assembly.resolve(job, {"items": [{"scene_index": 0}], "cuts": [], "titles": False})
    assert assembly._texts_for(job, spec, [3.0625]) == []


def test_an_unusable_scene_caption_is_logged_and_skipped(tmp_path, caplog):
    job = _job(tmp_path, n=1)
    job.scenes[0].texts = [{"text": "   "}, _caption()]
    spec, _ = assembly.resolve(job, {"items": [{"scene_index": 0}], "cuts": []})
    placed = assembly._texts_for(job, spec, [3.0625])
    assert len(placed) == 1 and "nothing to draw" in caplog.text


# --- the file that gets written -----------------------------------------------------------


def test_run_writes_the_ass_at_the_resolved_output_size(tmp_path, monkeypatch):
    """The bug already paid for once: PlayRes must be the resolved size, not a clip's. A 9:16 job
    given 16:9 coordinates has every caption rescaled and the anchored ones pushed off frame."""
    job = _job(tmp_path, n=1)
    job.profile = {"video": {"width": 480, "height": 832, "fps": 16, "length": 49}}
    job.scenes[0].texts = [_caption(size=0.048)]
    monkeypatch.setattr(media, "probe", lambda p: MediaInfo(480, 832, 16.0, 3.0625, 49, "h264", False))
    seen = {}

    def fake_assemble(clips, cuts, dest, **kw):
        seen.update(kw)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")
        return dest

    monkeypatch.setattr(media, "assemble", fake_assemble)
    spec, paths = assembly.resolve(job, {"items": [{"scene_index": 0}], "cuts": []})
    rec = assembly.run(job, spec, paths)
    assert rec.ok
    ass = pathlib.Path(seen["titles"])
    assert ass.name.endswith(".ass") and ass.is_file()
    body = ass.read_text(encoding="utf-8")
    assert "PlayResX: 480" in body and "PlayResY: 832" in body
    assert ",40," in body, "0.048 of 832 is a 40px caption"
    assert "Открий своя аромат" in body


def test_no_captions_means_no_ass_and_no_filter(tmp_path, monkeypatch):
    job = _job(tmp_path, n=1)
    seen = {}

    def fake_assemble(clips, cuts, dest, **kw):
        seen.update(kw)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")
        return dest

    monkeypatch.setattr(media, "assemble", fake_assemble)
    spec, paths = assembly.resolve(job, {"items": [{"scene_index": 0}], "cuts": []})
    assembly.run(job, spec, paths)
    assert seen["titles"] is None
    assert not list(assembly.output_dir(job).glob("*.ass"))


def test_status_offers_the_caption_vocabulary(tmp_path):
    st = assembly.status(_job(tmp_path))
    assert "fade" in st["text_effects"] and "typewriter" in st["text_effects"]
    assert "lower-third" in st["text_presets"]
