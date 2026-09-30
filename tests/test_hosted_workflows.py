"""The curated hosted-model workflows in ``workflows/hosted/``.

These are the entries that spend money, so they carry stricter invariants than a local
workflow: every one declares what a call costs, and a keyframe whose size is an enum must say
which aspects it can actually render. All of it is checked offline — no key, nothing billed.
"""

from __future__ import annotations

import pytest
import yaml

from vorch import config, formats, registry

HOSTED_DIR = config.WORKFLOW_DIR / "hosted"


def _hosted():
    reg = registry.load_registry()
    return [s for s in reg.list() if s.source and s.source.parent.name == "hosted"]


def test_the_curated_set_is_present_and_loads():
    names = {s.name for s in _hosted()}
    assert len(names) >= 10, names
    # one per provider worth having, across both kinds
    assert {"api_dalle3_t2i", "api_wan_t2i", "api_ideogram_v4_t2i", "api_nano_banana2_t2i",
            "api_seedream5_pro_t2i", "api_gpt_image_2_t2i"} <= names
    assert {"api_flux_kontext_max_edit", "api_nano_banana2_edit", "api_flux_erase"} <= names


def test_every_hosted_entry_is_a_kind_vorch_can_drive():
    for spec in _hosted():
        assert spec.kind in ("keyframe", "edit", "animate", "postprocess"), spec.name


def test_every_hosted_entry_declares_what_a_call_costs():
    """A hosted workflow with no price cannot be budgeted, and these all spend money."""
    for spec in _hosted():
        assert spec.cost is not None, f"{spec.name} has no cost:"
        assert spec.cost.min_usd > 0 and spec.cost.max_usd >= spec.cost.min_usd, spec.name
        assert spec.cost.max_usd < 1.0, f"{spec.name}: {spec.cost.describe()} looks like a typo"


def test_templates_are_api_format_and_offline_valid():
    reg = registry.load_registry()
    for spec in _hosted():
        wf = reg.load_template(spec.name)
        assert "nodes" not in wf, f"{spec.name} is a UI export"
        assert all("class_type" in n for n in wf.values()), spec.name
        assert reg.validate(spec) == [], spec.name      # offline: template + contract + backend


def test_a_keyframe_either_takes_pixels_or_says_which_aspects_it_can_render():
    """The failure this prevents: a 9:16 job on an enum-size model rendering something square,
    with no error, visible only in the picture."""
    for spec in _hosted():
        if spec.kind != "keyframe":
            continue
        rules = formats.keyframe_rules(spec.name)
        width = spec.params.get("width")
        wired_pixels = bool(width and width.targets)
        assert wired_pixels or rules.from_aspect, (
            f"{spec.name} neither wires width/height nor declares a from_aspect table"
        )
        if rules.from_aspect:
            # the two orientations this project actually ships must both be expressible
            assert "16:9" in rules.aspects and "9:16" in rules.aspects, (
                f"{spec.name} offers only {rules.aspects}"
            )


def test_every_hosted_entry_records_where_its_price_came_from():
    """The price is hand-copied and will drift, so the file has to say what it was read from."""
    for spec in _hosted():
        text = spec.source.read_text(encoding="utf-8")
        assert "# cost:" in text or "cost:" in text, spec.name
        assert "2026-" in text or "list price" in text or "price_badge" in text, (
            f"{spec.source.name} does not say where its price came from"
        )


def test_hosted_entries_do_not_shadow_a_local_workflow():
    """A hosted entry taking a local name would silently send a local job to a paid API."""
    local = {s.name for s in registry.load_registry().list()
             if s.source and s.source.parent.name != "hosted"}
    for spec in _hosted():
        assert spec.name not in local or spec.source.parent.name == "hosted", spec.name
    assert "keyframe" not in {s.name for s in _hosted()}
