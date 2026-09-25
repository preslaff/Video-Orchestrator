"""The ASS a caption compiles to (vorch/titles.py) — pure, no ffmpeg, no files.

The cases here are the ones that failed in a real render first: colour byte order, PlayRes against
the frame, and a typewriter that walks off centre while it types.
"""

from __future__ import annotations

import pytest

from vorch import titles
from vorch.titles import Placed, TextItem


def one(item: TextItem, start=0.0, duration=None, *, width=832, height=480) -> str:
    return titles.build([Placed(item, start, duration if duration is not None else item.duration)],
                        width=width, height=height)


def dialogues(ass: str) -> list[str]:
    return [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]


def styles(ass: str) -> list[str]:
    return [ln for ln in ass.splitlines() if ln.startswith("Style:")]


# --- escaping -------------------------------------------------------------------------


def test_braces_are_escaped_or_the_text_between_them_vanishes():
    """`{...}` is an ASS override block: an unescaped brace eats everything up to the next one."""
    assert titles.escape("Цена: 39,90 лв. {промо}") == r"Цена: 39,90 лв. \{промо\}"


def test_newlines_become_hard_breaks_and_commas_are_left_alone():
    assert titles.escape("едно\nдве") == r"едно\Nдве"
    assert titles.escape("едно\r\nдве") == r"едно\Nдве"
    # Text is the last field on the line and takes the rest of it, so a comma needs nothing.
    assert titles.escape("39,90") == "39,90"


def test_a_backslash_is_escaped_before_anything_else():
    assert titles.escape("a\\b") == "a\\\\b"
    assert titles.escape("\\N") == "\\\\N", "a literal backslash-N must not become a line break"


def test_escaped_text_reaches_the_event_line():
    ass = one(TextItem(text="Цена {промо}\nвтори ред", effect="none"))
    assert dialogues(ass)[0].endswith(r"Цена \{промо\}\Nвтори ред")


# --- colour ---------------------------------------------------------------------------


def test_colour_is_bgr_not_rgb():
    """The trap that rendered a gold kicker blue until the bytes were reversed."""
    assert titles.ass_colour("D8B26A") == "&H006AB2D8"
    assert titles.ass_colour("FFFFFF") == "&H00FFFFFF"
    assert titles.ass_colour("FF0000") == "&H000000FF", "red is BB=00 GG=00 RR=FF"


def test_opacity_is_transparency_and_accepts_a_hash():
    assert titles.ass_colour("FFFFFF", 1.0).startswith("&H00"), "opaque is 00"
    assert titles.ass_colour("FFFFFF", 0.0).startswith("&HFF"), "invisible is FF"
    assert titles.ass_colour("#d8b26a") == "&H006AB2D8"


def test_a_bad_colour_is_refused():
    with pytest.raises(ValueError, match="six hex digits"):
        titles.ass_colour("gold")


# --- the frame ------------------------------------------------------------------------


def test_playres_is_the_output_size():
    """libass scales its coordinate space to the frame; a mismatch moves and resizes everything."""
    ass = one(TextItem(text="x"), width=480, height=832)
    assert "PlayResX: 480" in ass and "PlayResY: 832" in ass


def test_sizes_are_fractions_of_height_so_both_orientations_read_the_same():
    item = TextItem(text="x", size=0.05, margin_x=0.1, margin_y=0.1)
    land = styles(one(item, width=832, height=480))[0].split(",")
    port = styles(one(item, width=480, height=832))[0].split(",")
    assert land[2] == "24" and port[2] == "42"          # 0.05 * height
    # Horizontal margins follow width, vertical follow height (MarginL, MarginR, MarginV).
    assert land[19:22] == ["83", "83", "48"]
    assert port[19:22] == ["48", "48", "83"]


def test_a_tiny_frame_still_gets_a_legible_floor():
    assert styles(one(TextItem(text="x", size=0.001), width=64, height=64))[0].split(",")[2] == "8"


# --- effects --------------------------------------------------------------------------


def test_none_draws_the_text_with_no_override():
    assert dialogues(one(TextItem(text="здравей", effect="none")))[0].endswith(",,здравей")


def test_fade_carries_both_ends():
    assert r"{\fad(350,350)}" in dialogues(one(TextItem(text="x", effect="fade")))[0]


def test_pop_scales_up_and_settles():
    line = dialogues(one(TextItem(text="x", effect="pop")))[0]
    assert r"\fscx60\fscy60" in line and r"\t(0,350,\fscx100\fscy100)" in line


def test_slide_rises_to_its_anchor():
    """`\\move` places the anchor the alignment names, so no font metrics are needed."""
    line = dialogues(one(TextItem(text="x", effect="slide", align=2, margin_y=0.1),
                         width=480, height=832))[0]
    # bottom-centre of a 480x832 frame with a 10% margin, rising 6% of height
    assert r"\move(240,799,240,749,0,350)" in line


def test_the_effect_never_outlasts_half_the_item():
    line = dialogues(one(TextItem(text="x", effect="fade", duration=0.4, effect_seconds=5)))[0]
    assert r"{\fad(200,200)}" in line


# --- typewriter -----------------------------------------------------------------------


def test_typewriter_reveals_one_character_at_a_time():
    ass = one(TextItem(text="абв", effect="typewriter", duration=3.0, effect_seconds=0.3))
    lines = dialogues(ass)
    assert len(lines) == 3
    assert lines[0].endswith("а") and lines[1].endswith("аб") and lines[2].endswith("абв")
    # The last one holds the finished line to the end; the others hand over.
    assert "0:00:00.00,0:00:00.10" in lines[0]
    assert "0:00:00.20,0:00:03.00" in lines[2]


def test_typewriter_is_anchored_left_so_it_does_not_crawl():
    """Centred, each revealed character re-centres the line and the text walks leftwards."""
    ass = one(TextItem(text="абв", effect="typewriter", align=2), width=480, height=832)
    assert styles(ass)[0].split(",")[18] == "1", "bottom-centre becomes bottom-left"
    assert r"\pos(" in dialogues(ass)[0]
    # Every event pins the same point, which is what stops the crawl.
    assert len({ln.split(r"\pos(")[1].split(")")[0] for ln in dialogues(ass)}) == 1


def test_typewriter_keeps_an_alignment_that_is_already_left():
    ass = one(TextItem(text="аб", effect="typewriter", align=7))
    assert styles(ass)[0].split(",")[18] == "7"


# --- times ----------------------------------------------------------------------------


def test_events_use_the_resolved_timeline_position_not_the_items_own_start():
    """A scene's caption is written in clip time and resolved by the caller; build only sees the
    answer, so it can never apply an offset twice."""
    item = TextItem(text="x", start=1.2, duration=2.0, effect="none")
    assert "0:00:09.50,0:00:11.50" in dialogues(one(item, start=9.5, duration=2.0))[0]


def test_timestamps_round_to_centiseconds_and_never_go_negative():
    item = TextItem(text="x", effect="none")
    assert "0:01:05.44" in dialogues(one(item, start=65.437, duration=1.0))[0]
    assert dialogues(one(item, start=-3.0, duration=1.0))[0].startswith("Dialogue: 0,0:00:00.00,")


# --- several items --------------------------------------------------------------------


def test_each_item_gets_its_own_style():
    ass = titles.build([
        Placed(TextItem(text="a", colour="D8B26A"), 0.0, 1.0),
        Placed(TextItem(text="b", size=0.1), 1.0, 1.0),
    ], width=832, height=480)
    assert [s.split(",")[0] for s in styles(ass)] == ["Style: t0", "Style: t1"]
    assert "&H006AB2D8" in styles(ass)[0] and styles(ass)[1].split(",")[2] == "48"
    assert len(dialogues(ass)) == 2


def test_no_items_is_a_valid_empty_script():
    ass = titles.build([], width=832, height=480)
    assert "[Events]" in ass and dialogues(ass) == []


# --- validation -----------------------------------------------------------------------


@pytest.mark.parametrize("kw,message", [
    ({"text": "   "}, "nothing to draw"),
    ({"effect": "explode"}, "unknown"),
    ({"start": -1}, "cannot be negative"),
    ({"duration": 0}, "must be positive"),
    ({"size": 0}, "fraction of the frame height"),
    ({"size": 2}, "fraction of the frame height"),
    ({"align": 0}, "ASS anchor"),
    ({"align": 10}, "ASS anchor"),
    ({"opacity": 1.5}, "0-1"),
    ({"colour": "gold"}, "six hex digits"),
    ({"outline": "12345"}, "six hex digits"),
    ({"margin_x": 0.6}, "0-0.5"),
])
def test_validate_refuses(kw, message):
    with pytest.raises(ValueError, match=message):
        titles.validate(TextItem(**{"text": "x", **kw}))


def test_validate_refuses_a_caption_that_starts_past_the_end():
    with pytest.raises(ValueError, match="past the end"):
        titles.validate(TextItem(text="x", start=4.0), limit=3.0)
    titles.validate(TextItem(text="x", start=2.9), limit=3.0)      # inside is fine


# --- parsing --------------------------------------------------------------------------


def test_from_dict_fills_every_default_so_an_old_job_loads():
    item = titles.from_dict({"text": "здравей"})
    assert item.effect == "fade" and item.align == 2 and item.size == 0.048
    titles.validate(item)


def test_from_dict_normalises_what_a_person_typed():
    item = titles.from_dict({"text": "x", "colour": "#d8b26a", "effect": "POP"})
    assert item.colour == "D8B26A" and item.effect == "pop"


def test_a_stored_item_round_trips_unchanged():
    item = titles.from_dict({"text": "x", "colour": "#d8b26a", "effect": "POP", "size": 0.06})
    assert titles.from_dict(item.to_dict()).to_dict() == item.to_dict()


def test_from_dict_rejects_a_number_that_is_not_one():
    with pytest.raises(ValueError, match="expected a number"):
        titles.from_dict({"text": "x", "start": "soon"})


def test_presets_are_all_usable_items():
    for name, preset in titles.PRESETS.items():
        item = TextItem(text=f"{name}", **preset)
        titles.validate(item)
        assert item.effect in titles.EFFECTS
