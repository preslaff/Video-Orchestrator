"""Handoff export folder: layout, names, manifest/README, idempotence, fallbacks."""

from __future__ import annotations

import errno
import json
import os

import pytest

from vorch import config, export, projects, provenance
from vorch.evaluator import Verdict
from vorch.state import Scene, Variant, new_job


def _v(score=7):
    return Verdict(action_correct=True, identity_match=True, style_match=True, artifacts=False,
                   composition_ok=True, score=score, verdict="accept", fix_suggestion="")


@pytest.fixture(autouse=True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPORT_ROOT", tmp_path / "export")
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")


def _media(tmp_path, name, manifest=None):
    p = tmp_path / "out" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(name.encode())
    if manifest:
        provenance.write_manifest(p, "keyframe", manifest)
    return str(p)


def _video_job(tmp_path):
    job = new_job(idea="A woman bakes bread at dawn", kind="video", profile_id="cinematic_16x9",
                  profile={"aspect": "16:9", "keyframe": {"width": 1024, "height": 576},
                           "video": {"width": 832, "height": 480, "length": 49, "fps": 16}})
    job.scenes = [
        Scene(index=0, description="kneading", keyframe_prompt="kf0", animation_prompt="an0",
              keyframe_path=_media(tmp_path, "s0kf.png", {"positive": "kf0", "seed": 11}),
              swapped_path=_media(tmp_path, "s0sw.png"),
              raw_video_path=_media(tmp_path, "video/s0raw.mp4"),
              video_path=_media(tmp_path, "video/s0final.mp4"),
              keyframe_verdict=_v(8)),
        Scene(index=1, description="oven", keyframe_prompt="kf1", animation_prompt="an1",
              keyframe_path=_media(tmp_path, "s1kf.png"), video_path=str(tmp_path / "missing.mp4")),
    ]
    return job


def test_layout_names_manifest_and_readme(store, tmp_path):
    job = _video_job(tmp_path)
    store.save_job(job)
    proj = projects.save(store, projects.new_project("Bakery Launch"))
    rec = export.export_job(store, job, proj)
    dest = tmp_path / "export" / "bakery-launch" / f"{job.id[:8]}_a-woman-bakes-bread-at-dawn"
    assert rec.path == str(dest) and rec.ok
    assert rec.files == ["s01_clip.mp4", "s01_clip_raw.mp4", "s01_keyframe.png", "s01_swapped.png", "s02_keyframe.png"]
    assert rec.missing == [f"s02_clip.mp4 <- {tmp_path / 'missing.mp4'}"]
    assert (dest / "s01_clip.mp4").read_bytes() == b"video/s0final.mp4"
    assert (dest / "provenance" / "s01_keyframe.png.json").is_file()
    m = json.loads((dest / "manifest.json").read_text())
    assert m["job_id"] == job.id and m["project"]["slug"] == "bakery-launch" and m["kind"] == "video"
    assert m["scenes"][0]["seeds"]["keyframe"] == 11 and m["scenes"][0]["verdicts"]["keyframe"]["score"] == 8
    assert m["scenes"][0]["files"]["s01_clip.mp4"].endswith("s0final.mp4")
    assert m["missing"] == rec.missing
    readme = (dest / "README.txt").read_text()
    assert "832x480 @ 16 fps, 49 frames (~3.06s each)" in readme and "01. kneading" in readme and "Shotcut" in readme
    sb = (dest / "storyboard.md").read_text()
    assert "## Scene 1" in sb and "an0" in sb and "score 8" in sb


def test_unassigned_and_stills_variants_named(tmp_path):
    job = new_job(idea="Posters", kind="stills")
    job.scenes = [Scene(index=0, keyframe_path=_media(tmp_path, "b.png"), variants=[
        Variant(_media(tmp_path, "a.png"), swapped_path=_media(tmp_path, "as.png")),
        Variant(_media(tmp_path, "b.png"), post_paths=[_media(tmp_path, "bp.png")]),
    ])]
    rec = export.export_job(None, job, None)
    assert rec.path.startswith(str(tmp_path / "export" / "_unassigned"))
    assert rec.files == ["s01_keyframe.png", "s01_keyframe_v2.png", "s01_post_v2_1.png", "s01_swapped.png"]
    assert "stills" in (tmp_path / "export" / "_unassigned").glob("*").__next__().joinpath("README.txt").read_text()


def test_idempotent_reexport_and_hardlink(tmp_path):
    job = _video_job(tmp_path)
    rec1 = export.export_job(None, job, None)
    dest = tmp_path / "export" / "_unassigned"
    first = next(dest.glob("*"))
    ino_before = (first / "s01_keyframe.png").stat().st_ino
    assert ino_before == os.stat(job.scenes[0].keyframe_path).st_ino  # hardlinked
    rec2 = export.export_job(None, job, None)
    assert rec2.files == rec1.files and (first / "s01_keyframe.png").stat().st_ino == ino_before


def test_hardlink_falls_back_to_copy(tmp_path, monkeypatch):
    def no_link(src, dst):
        raise OSError(errno.EXDEV, "cross-device")

    monkeypatch.setattr(export.os, "link", no_link)
    job = _video_job(tmp_path)
    rec = export.export_job(None, job, None)
    dest = next((tmp_path / "export" / "_unassigned").glob("*"))
    assert rec.ok and (dest / "s01_keyframe.png").read_bytes() == b"s0kf.png"
    assert (dest / "s01_keyframe.png").stat().st_ino != os.stat(job.scenes[0].keyframe_path).st_ino


def test_all_sources_missing_reported_not_raised(tmp_path):
    job = new_job(idea="ghost")
    job.scenes = [Scene(index=0, keyframe_path="/nope/k.png", video_path="/nope/v.mp4")]
    rec = export.export_job(None, job, None)
    assert rec.ok is False and len(rec.missing) == 2 and "no deliverables" in rec.error
    assert (tmp_path / "export" / "_unassigned").exists()


def test_status_and_stale(tmp_path):
    job = _video_job(tmp_path)
    assert export.status(job)["exported"] is False
    rec = export.export_job(None, job, None)
    job.export = rec.to_dict()
    st = export.status(job)
    assert st["exported"] and st["path"] == rec.path and st["stale"] is False and st["exists"]
    job.updated_at = "2999-01-01T00:00:00+00:00"
    assert export.status(job)["stale"] is True


def test_assembly_listed_in_plan_manifest_and_readme(store, tmp_path):
    job = _video_job(tmp_path)
    asm = _media(tmp_path, "vorch/abc/assembly.mp4")
    provenance.write_manifest(asm, "assemble", {"spec": {"cuts": []}}, {"kind": "assembly"})
    job.assembly = {"path": asm, "ok": True, "n_clips": 2, "duration": 5.62,
                    "spec": {"cuts": [{"transition": "fade", "duration": 0.5}, {"transition": "none", "duration": 0.5}],
                             "items": [{"scene_index": 0, "target_seconds": 6.125, "speed": 0.5}, {"scene_index": 1}],
                             "smooth": True}}
    rec = export.export_job(store, job, None)
    assert "assembly.mp4" in rec.files
    dest = next((tmp_path / "export" / "_unassigned").glob("*"))
    assert (dest / "assembly.mp4").read_bytes() == b"vorch/abc/assembly.mp4"
    assert (dest / "provenance" / "assembly.mp4.json").is_file()
    m = json.loads((dest / "manifest.json").read_text())
    assert m["assembly"]["n_clips"] == 2
    readme = (dest / "README.txt").read_text()
    assert "assembly.mp4: 2 clip(s), 5.62s, cuts: fade 0.5s, none, speed: s01 0.5× (smooth slow motion)" in readme
    assert "ready cut" in readme


def test_failed_assembly_not_exported(tmp_path):
    job = _video_job(tmp_path)
    job.assembly = {"path": str(tmp_path / "nope.mp4"), "ok": False, "error": "boom"}
    rec = export.export_job(None, job, None)
    assert "assembly.mp4" not in rec.files and not any(m.startswith("assembly") for m in rec.missing)
    dest = next((tmp_path / "export" / "_unassigned").glob("*"))
    assert json.loads((dest / "manifest.json").read_text())["assembly"] is None
