"""GPU-free checks for asset captioning — schema validation, retries, label building.

No live Gemma server: gemma.complete_vision is monkeypatched (as in test_evaluator)
so we exercise parsing, the retry loop, prompt construction and label_for.
"""

from __future__ import annotations

import json

import pytest

from vorch import asset_vision
from vorch.asset_vision import AssetNote, AssetVisionError, describe, label_for

FULL = {
    "caption": "A white stone lighthouse on a green headland above grey sea",
    "subjects": ["lighthouse", "stone wall", "sea"],
    "setting": "coastal headland",
    "people_count": 0,
    "is_portrait": False,
    "time_of_day": "afternoon",
    "weather": "overcast",
    "text_in_image": "",
    "landmark_guess": "",
    "landmark_confidence": 0,
    "tags": ["exterior", "lighthouse"],
    "quality_issues": [],
}


@pytest.fixture
def image(tmp_path):
    """A file that exists; media.downscale is stubbed so the bytes never matter."""
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"not really a jpeg")
    return p


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch):
    monkeypatch.setattr(asset_vision.media, "downscale", lambda src, **kw: src)


def _reply(monkeypatch, *responses):
    """Stub complete_vision with a sequence of replies; record the calls."""
    calls = []
    it = iter(responses)

    def fake(system, user_text, images, **kw):
        calls.append({"system": system, "user": user_text, "images": images, **kw})
        return next(it)

    monkeypatch.setattr(asset_vision.gemma, "complete_vision", fake)
    return calls


def test_clean_json_becomes_a_note(monkeypatch, image):
    _reply(monkeypatch, json.dumps(FULL))
    note = describe(image)
    assert note.caption.startswith("A white stone lighthouse")
    assert note.tags == ["exterior", "lighthouse"] and note.time_of_day == "afternoon"


def test_fenced_json_is_repaired(monkeypatch, image):
    _reply(monkeypatch, "```json\n" + json.dumps(FULL) + "\n```")
    assert describe(image).setting == "coastal headland"


def test_missing_optional_fields_still_validate(monkeypatch, image):
    _reply(monkeypatch, json.dumps({"caption": "a bicycle against a wall"}))
    note = describe(image)
    assert note.caption == "a bicycle against a wall"
    assert note.subjects == [] and note.people_count == 0 and note.time_of_day == "unknown"


def test_missing_caption_retries_then_succeeds(monkeypatch, image):
    calls = _reply(monkeypatch, json.dumps({"subjects": ["x"]}), json.dumps(FULL))
    assert describe(image).caption.startswith("A white stone")
    assert len(calls) == 2


def test_all_attempts_bad_raises(monkeypatch, image):
    _reply(monkeypatch, "not json", "still not json")
    with pytest.raises(AssetVisionError):
        describe(image)


def test_uses_evaluation_temperature_and_a_schema(monkeypatch, image):
    """NOTES.md §6: 0.2 for judging/describing, and constrain output at the source."""
    calls = _reply(monkeypatch, json.dumps(FULL))
    describe(image)
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["response_format"]["json_schema"]["name"] == "asset_note"


def test_hint_and_kind_steer_the_prompt(monkeypatch, image):
    calls = _reply(monkeypatch, json.dumps(FULL), json.dumps(FULL))
    describe(image, hint="Irish lighthouses; tag as lamp_room, fog_horn", kind="reference")
    describe(image, kind="master_face")
    assert "fog_horn" in calls[0]["user"] and "tags" in calls[0]["user"]
    assert "PERSON" in calls[1]["user"] and "fog_horn" not in calls[1]["user"]
    assert calls[0]["user"] != calls[1]["user"]


def test_video_samples_frames(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    frames = [tmp_path / f"f{i}.png" for i in range(3)]
    for f in frames:
        f.write_bytes(b"x")
    monkeypatch.setattr(asset_vision.media, "sample_frames", lambda p, n: frames[:n])
    calls = _reply(monkeypatch, json.dumps(FULL))
    describe(clip)
    assert len(calls[0]["images"]) == 3
    assert "one short video clip" in calls[0]["user"].lower() or "ONE short video" in calls[0]["user"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        describe(tmp_path / "nope.jpg")


# --- label_for ---------------------------------------------------------------------


def test_label_for_reference_leads_with_setting():
    label = label_for(AssetNote.model_validate(FULL), kind="reference")
    assert label.startswith("coastal headland — A white stone lighthouse")


def test_label_for_face_describes_the_person():
    note = AssetNote(caption="woman in her thirties, dark hair, freckles", is_portrait=True)
    assert label_for(note, kind="master_face") == "woman in her thirties, dark hair, freckles"


def test_label_for_face_flags_a_missing_face():
    note = AssetNote(caption="an empty beach at dusk", is_portrait=False)
    assert "no clear single face" in label_for(note, kind="master_face")


def test_label_for_uses_a_confident_landmark_only():
    sure = AssetNote(caption="a tower", setting="headland", landmark_guess="Hook Head",
                     landmark_confidence=9)
    unsure = AssetNote(caption="a tower", setting="headland", landmark_guess="Hook Head",
                       landmark_confidence=3)
    assert label_for(sure, kind="reference").startswith("Hook Head")
    assert "Hook Head" not in label_for(unsure, kind="reference")


def test_label_for_truncates_on_a_word_boundary():
    note = AssetNote(caption="word " * 60)
    label = label_for(note, kind="reference", max_len=40)
    assert len(label) <= 41 and label.endswith("…") and "  " not in label


def test_label_for_falls_back_to_the_filename():
    assert label_for(AssetNote(caption=""), kind="reference", fallback="780482127_18415") == "780482127_18415"


def test_label_for_keeps_cyrillic():
    note = AssetNote(caption="Жена с колело пред фара", setting="крайбрежие")
    assert "Жена с колело" in label_for(note, kind="reference")


def test_label_for_survives_a_chatty_setting():
    """The model is asked for a short phrase; a sentence must not eat the whole label."""
    note = AssetNote(
        caption="A white lighthouse stands above a choppy sea.",
        setting="A coastal environment featuring a white lighthouse tower and adjacent weathered "
                "stone buildings overlooking a large body of water.",
    )
    label = label_for(note, kind="reference")
    assert label.startswith("A coastal environment — A white lighthouse")
    assert "weathered" not in label


def test_phrase_keeps_short_settings_intact():
    assert asset_vision._phrase("rocky headland") == "rocky headland"
    assert asset_vision._phrase("spiral staircase, looking up") == "spiral staircase"
    assert asset_vision._phrase("") == ""
