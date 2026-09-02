"""Aspect presets, computed sizes, validation and VRAM warnings (vorch/formats.py)."""

from __future__ import annotations

import pytest

from vorch import formats


@pytest.mark.parametrize("aspect", list(formats.ASPECTS))
def test_presets_are_16_aligned_and_match_aspect(aspect):
    for w, h in (formats.keyframe_size(aspect), formats.wan_size(aspect)):
        assert w % 16 == 0 and h % 16 == 0
        assert abs((w / h) - formats.ratio(aspect)) / formats.ratio(aspect) < 0.06


def test_computed_sizes_for_unlisted_aspect():
    w, h = formats.keyframe_size("21:9")
    assert w % 16 == 0 and h % 16 == 0 and 0.5e6 <= w * h <= 0.7e6 and w > h
    w, h = formats.wan_size("2:3")
    assert (w, h) == (480, 720)


def test_orientation_and_hint():
    assert formats.orientation("16:9") == "landscape"
    assert formats.orientation("9:16") == "portrait"
    assert formats.orientation("1:1") == "square"
    assert "vertical" in formats.framing_hint("9:16")


def test_parse_aspect_rejects_garbage():
    for bad in ("wide", "16x9", "0:1", ""):
        with pytest.raises(ValueError):
            formats.parse_aspect(bad)


def test_validate_size_and_length():
    formats.validate_size(1024, 576)
    with pytest.raises(ValueError):
        formats.validate_size(1000, 576)
    with pytest.raises(ValueError):
        formats.validate_size(128, 576)
    formats.validate_length(81)
    with pytest.raises(ValueError):
        formats.validate_length(80)


def test_vram_warnings_thresholds():
    assert formats.vram_warnings(832, 480, 49) == []
    assert formats.vram_warnings(832, 480, 81) == []            # 1.65x landscape: fine
    assert len(formats.vram_warnings(480, 832, 81)) == 1        # 1.65x portrait: watch
    assert "OOM" in formats.vram_warnings(832, 480, 121)[0]     # 2.47x: loud
    assert formats.clip_seconds(49, 16) == 3.06


def test_aspects_match_tolerance():
    assert formats.aspects_match((1024, 576), (832, 480))
    assert formats.aspects_match((576, 1024), (480, 832))   # 2.5% apart after 16-alignment
    assert not formats.aspects_match((1024, 576), (480, 832))
    assert not formats.aspects_match((768, 768), (832, 480))
