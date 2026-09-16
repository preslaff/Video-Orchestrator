"""vorch/assembly.py: defaults, spec resolution/validation, runner, record roundtrip."""

from __future__ import annotations

import pathlib

import pytest

from vorch import assembly, config, media
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
        ({"items": [{"scene_index": 0}, {"scene_index": 0}]}, "twice"),
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
