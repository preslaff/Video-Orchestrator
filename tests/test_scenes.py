"""Structural storyboard edits: insert, remove, reorder, renumber, assembly remap.

Pure functions over a Job — no store, no Gemma, no GPU. The invariant under test throughout:
``index`` is the position (renumbered on every change) and ``id`` is the identity (never
changes), so nothing built on a scene follows its number.
"""

from __future__ import annotations

import pytest

from vorch import scenes
from vorch.state import Clip, Scene, Variant, new_job


def _job(n: int = 3):
    job = new_job(idea="coast", n_scenes=n)
    job.scenes = [Scene(index=i, description=f"scene {i}") for i in range(n)]
    return job


def test_insert_renumbers_and_leaves_the_others_untouched():
    job = _job()
    ids = [s.id for s in job.scenes]
    added = scenes.insert(job, 1, description="the lamp at night", setting="Atlantic", writing_prompts=True)
    assert [s.index for s in job.scenes] == [0, 1, 2, 3]
    assert [s.id for s in job.scenes] == [ids[0], added.id, ids[1], ids[2]]   # identities kept
    assert added.description == "the lamp at night" and added.writing_prompts is True
    assert added.mode == job.mode and added.keyframe_path is None             # nothing generated yet
    assert job.n_scenes == 4                                                  # the cap follows reality


def test_insert_clamps_a_position_past_the_end():
    job = _job()
    assert scenes.insert(job, 99, description="last").index == 3
    assert scenes.insert(job, -5, description="first").index == 0


def test_remove_keeps_the_files_by_default_and_deletes_them_on_request(tmp_path):
    job = _job()
    kf, clip, old = tmp_path / "k.png", tmp_path / "c.mp4", tmp_path / "old.png"
    for f in (kf, clip, old):
        f.write_bytes(b"x")
    doomed = job.scenes[1]
    doomed.keyframe_path, doomed.video_path = str(kf), str(clip)
    doomed.variants = [Variant(path=str(old)), Variant(path=str(kf))]
    doomed.clips = [Clip(video_path=str(clip))]

    kept, was = scenes.remove(job, doomed.id, delete_files=False)
    assert kept is doomed and was == 1
    assert [s.index for s in job.scenes] == [0, 1] and [s.description for s in job.scenes] == ["scene 0", "scene 2"]
    assert kf.exists() and clip.exists() and old.exists()      # the job stops using them, that is all

    job2 = _job()
    victim = job2.scenes[0]
    victim.keyframe_path = str(kf)
    victim.variants = [Variant(path=str(kf)), Variant(path=str(old))]
    victim.clips = [Clip(video_path=str(clip))]
    scenes.remove(job2, victim.id, delete_files=True)
    assert not kf.exists() and not clip.exists() and not old.exists()   # every version, not just the active one


def test_remove_survives_a_file_that_is_already_gone(tmp_path):
    job = _job()
    job.scenes[0].keyframe_path = str(tmp_path / "never-existed.png")
    scenes.remove(job, job.scenes[0].id, delete_files=True)             # no exception
    assert len(job.scenes) == 2


def test_reorder_moves_scenes_and_rejects_anything_but_a_permutation():
    job = _job(4)
    ids = [s.id for s in job.scenes]
    scenes.reorder(job, [ids[3], ids[0], ids[2], ids[1]])
    assert [s.description for s in job.scenes] == ["scene 3", "scene 0", "scene 2", "scene 1"]
    assert [s.index for s in job.scenes] == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="missing"):
        scenes.reorder(job, ids[:2])
    with pytest.raises(ValueError, match="unknown"):
        scenes.reorder(job, [*ids, "ghost"])
    with pytest.raises(ValueError):
        scenes.reorder(job, [ids[0], ids[0], ids[1], ids[2]])          # a duplicate loses a scene


def test_unknown_scene_id_is_a_keyerror():
    with pytest.raises(KeyError):
        scenes.by_id(_job(), "ghost")


def test_remap_assembly_follows_moved_scenes_and_drops_removed_ones():
    job = _job(3)
    job.assembly = {"spec": {"items": [{"scene_index": 0, "source": "final"},
                                       {"scene_index": 1, "source": "final"},
                                       {"scene_index": 2, "source": "raw"}],
                             "fps": 16}}
    # scene 1 deleted: 2 becomes 1.
    assert scenes.remap_assembly(job, {2: 1}, removed=[1]) is True
    items = job.assembly["spec"]["items"]
    assert [i["scene_index"] for i in items] == [0, 1]
    assert items[1]["source"] == "raw"          # it is still *that* scene's clip, not the old scene 1
    assert job.assembly["spec"]["fps"] == 16    # the rest of the spec is untouched

    assert scenes.remap_assembly(job, {}, removed=[]) is False        # nothing to do
    assert scenes.remap_assembly(new_job(idea="no assembly"), {1: 0}) is False
