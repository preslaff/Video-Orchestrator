"""vorch/assembly.py: defaults, spec resolution/validation, runner, record roundtrip."""

from __future__ import annotations

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
    assert rec.ok and rec.path.endswith("/assembly.mp4") and rec.size_bytes == 3 and rec.n_clips == 3
    assert seen["cuts"] == [Cut("none", 0.5), Cut("wipeleft", 0.3)] and seen["kw"]["size"] == (832, 480) and seen["kw"]["fps"] == 16.0
    from vorch import provenance
    m = provenance.read_manifest(rec.path)
    assert m["kind"] == "assembly" and m["params"]["spec"]["cuts"][1]["transition"] == "wipeleft"
    job.assembly = rec.to_dict()
    st = assembly.status(job, active=False)
    assert st["record"]["exists"] is True and st["record"]["exported"] is False


def test_record_and_spec_roundtrip():
    spec = assembly.AssemblySpec(items=[assembly.AssemblyItem(0), assembly.AssemblyItem(2, "raw")],
                                 cuts=[Cut("fade", 0.4)], fps=24, crf=17)
    assert assembly.AssemblySpec.from_dict(spec.to_dict()) == spec
    rec = assembly.AssemblyRecord(path="/a.mp4", created_at="t", duration=5.2, n_clips=2, spec=spec.to_dict())
    assert assembly.AssemblyRecord.from_dict(rec.to_dict()) == rec
    assert assembly.AssemblyRecord.from_dict({"path": "/x", "ok": False, "error": "e"}).error == "e"
