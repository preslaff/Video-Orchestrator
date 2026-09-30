"""Converting a ComfyUI **UI export** into the API format the registry needs.

The two formats differ in more than shape: the UI form keeps widget values in a positional list
per node and wires sockets through a separate ``links`` array. Widget *order* is not recorded in
the export, so the node schemas from ``/object_info`` are what line the values up — which is why
importing needs a running ComfyUI. Everything here uses hand-built schemas instead.
"""

from __future__ import annotations

import pytest

from vorch import registry, registry_tools as rt

# A hosted text-to-image node: a prompt, a seed (followed in the UI by control_after_generate),
# a size enum and a count. Plus SaveImage, whose only widget is the filename prefix.
OI = {
    "FakeT2I": {
        "api_node": True,
        "output": ["IMAGE"],
        "input": {
            "required": {"prompt": ["STRING", {"default": ""}]},
            "optional": {
                "seed": ["INT", {"default": 0}],
                "size": ["COMBO", {"default": "1024x1024",
                                   "options": ["1024x1024", "1024x1792", "1792x1024"]}],
                "n": ["INT", {"default": 1}],
                "image": ["IMAGE", {}],
            },
        },
    },
    "SaveImage": {
        "output": ["IMAGE"],
        "input": {"required": {"images": ["IMAGE", {}],
                               "filename_prefix": ["STRING", {"default": "ComfyUI"}]}},
    },
    "LoadImage": {
        "output": ["IMAGE", "MASK"],
        "input": {"required": {"image": ["COMBO", {"options": ["a.png"]}]}},
    },
}


def _ui(nodes, links=()):
    return {"nodes": list(nodes), "links": [list(x) for x in links]}


def test_widget_values_line_up_and_the_control_string_is_stepped_over():
    doc = _ui(
        [{"id": 10, "type": "FakeT2I", "widgets_values": ["a cat", 42, "randomize", "1792x1024", 2]},
         {"id": 20, "type": "SaveImage", "widgets_values": ["SHOT"],
          "inputs": [{"name": "images", "type": "IMAGE", "link": 8}]}],
        links=[[8, 10, 0, 20, 0, "IMAGE"]],
    )
    api = rt.ui_to_api(doc, OI)
    assert api["10"]["inputs"] == {"prompt": "a cat", "seed": 42, "size": "1792x1024", "n": 2}
    assert api["20"]["inputs"] == {"images": ["10", 0], "filename_prefix": "SHOT"}
    assert "nodes" not in api


def test_an_absent_control_string_does_not_eat_the_next_value():
    doc = _ui([{"id": 1, "type": "FakeT2I", "widgets_values": ["p", 7, "1024x1024", 3]}])
    assert rt.ui_to_api(doc, OI)["1"]["inputs"] == {"prompt": "p", "seed": 7,
                                                    "size": "1024x1024", "n": 3}


def test_trailing_optional_widgets_may_be_omitted():
    doc = _ui([{"id": 1, "type": "FakeT2I", "widgets_values": ["p"]}])
    assert rt.ui_to_api(doc, OI)["1"]["inputs"] == {"prompt": "p"}


def test_unlinked_sockets_are_omitted_and_linked_ones_resolved():
    doc = _ui(
        [{"id": 1, "type": "LoadImage", "widgets_values": ["a.png"]},
         {"id": 2, "type": "FakeT2I", "widgets_values": ["p", 1, "fixed", "1024x1024", 1],
          "inputs": [{"name": "image", "type": "IMAGE", "link": 5}]}],
        links=[[5, 1, 0, 2, 0, "IMAGE"]],
    )
    api = rt.ui_to_api(doc, OI)
    assert api["2"]["inputs"]["image"] == ["1", 0]


def test_a_widget_dragged_into_a_socket_takes_its_value_from_the_link():
    doc = _ui(
        [{"id": 1, "type": "LoadImage", "widgets_values": ["a.png"]},
         # `prompt` was converted to an input: it is fed by a link and absent from widgets_values
         {"id": 2, "type": "FakeT2I", "widgets_values": [9, "fixed", "1024x1024", 1],
          "inputs": [{"name": "prompt", "type": "STRING", "link": 5,
                      "widget": {"name": "prompt"}}]}],
        links=[[5, 1, 0, 2, 0, "STRING"]],
    )
    got = rt.ui_to_api(doc, OI)["2"]["inputs"]
    assert got["prompt"] == ["1", 0] and got["seed"] == 9 and got["size"] == "1024x1024"


def test_muted_and_bypassed_nodes_are_dropped():
    doc = _ui([{"id": 1, "type": "FakeT2I", "widgets_values": ["p"], "mode": 4},
               {"id": 2, "type": "FakeT2I", "widgets_values": ["q"], "mode": 0}])
    assert sorted(rt.ui_to_api(doc, OI)) == ["2"]


def test_annotation_nodes_are_skipped_not_reported_missing():
    """Note/MarkdownNote never execute, so they are absent from /object_info by design."""
    doc = _ui([{"id": 1, "type": "MarkdownNote", "widgets_values": ["hello"]},
               {"id": 2, "type": "Note", "widgets_values": ["hi"]},
               {"id": 3, "type": "FakeT2I", "widgets_values": ["p"]}])
    assert sorted(rt.ui_to_api(doc, OI)) == ["3"]


def test_a_genuinely_missing_class_is_reported_with_what_to_do():
    doc = _ui([{"id": 1, "type": "SomeCustomNode", "widgets_values": []}])
    with pytest.raises(rt.ConversionError, match="install the custom nodes"):
        rt.ui_to_api(doc, OI)


def test_a_subgraph_reference_is_refused_with_its_own_message():
    doc = _ui([{"id": 1, "type": "e20a7fb5-3d72-41c9-a78c-fdf287ec46ec", "widgets_values": []}])
    with pytest.raises(rt.ConversionError, match="subgraph/component"):
        rt.ui_to_api(doc, OI)


def test_an_api_format_document_passes_through_untouched():
    api = {"1": {"class_type": "FakeT2I", "inputs": {"prompt": "p"}}}
    assert rt.ui_to_api(api, OI) is api


# --- dynamic combos and autogrow -------------------------------------------------

OI_DYN = {
    "FakeDyn": {
        "api_node": True,
        "output": ["IMAGE"],
        "input": {"required": {
            "prompt": ["STRING", {}],
            "model": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                {"key": "small", "inputs": {"required": {"width": ["INT", {}]}}},
                {"key": "big", "inputs": {"required": {
                    "size": ["COMBO", {"options": ["2048x2048"]}],
                    "images": ["COMFY_AUTOGROW_V3", {"template": {
                        "input": {"required": {"image": ["IMAGE", {}]}},
                        "names": ["image_1", "image_2"]}}],
                }}},
            ]}],
        }},
    },
}


def test_a_dynamic_combo_emits_flat_dotted_keys_for_the_selected_option_only():
    doc = _ui([{"id": 1, "type": "FakeDyn", "widgets_values": ["p", "big", "2048x2048"]}])
    got = rt.ui_to_api(doc, OI_DYN)["1"]["inputs"]
    assert got == {"prompt": "p", "model": "big", "model.size": "2048x2048"}
    # the other option's inputs must not appear
    assert "model.width" not in got


def test_the_other_option_is_used_when_it_is_the_one_selected():
    doc = _ui([{"id": 1, "type": "FakeDyn", "widgets_values": ["p", "small", 512]}])
    assert rt.ui_to_api(doc, OI_DYN)["1"]["inputs"]["model.width"] == 512


def test_an_autogrow_slot_resolves_to_its_template_type():
    """`model.images.image_1` has no key in the schema; its type comes from the autogrow
    template, which is what lets an image wired there mark the graph as an edit."""
    declared = OI_DYN["FakeDyn"]["input"]["required"]
    assert rt._declared_type(declared, "model.images.image_1") == "IMAGE"
    assert rt._declared_type(declared, "model.size") == "COMBO"
    assert rt._declared_type(declared, "nope.nothing") is None


# --- kind detection ---------------------------------------------------------------

def _api(cls, inputs):
    return {"1": {"class_type": cls, "inputs": inputs}}


def test_kind_falls_back_to_custom_without_object_info():
    """A hosted graph has no Empty*Latent and no sampler, so the local heuristics cannot see it."""
    assert rt.guess_kind(_api("FakeT2I", {"prompt": "p"})) == "custom"
    assert rt.guess_kind(_api("FakeT2I", {"prompt": "p"}), OI) == "keyframe"


def test_a_wired_image_makes_it_an_edit_even_though_the_node_also_does_t2i():
    assert rt.guess_kind(_api("FakeT2I", {"prompt": "p", "image": ["9", 0]}), OI) == "edit"


def test_only_the_first_output_counts():
    """A 3D generator trails an IMAGE preview after the real product; it is not a keyframe."""
    oi = {"Gen3D": {"api_node": True, "output": ["STRING", "FILE_3D_GLB", "IMAGE"],
                    "input": {"required": {"prompt": ["STRING", {}]}}}}
    assert rt.guess_kind(_api("Gen3D", {"prompt": "p"}), oi) == "custom"


def test_wired_video_means_postprocess_not_keyframe():
    oi = {"Matte": {"api_node": True, "output": ["IMAGE", "MASK"],
                    "input": {"required": {"video": ["VIDEO", {}]}}}}
    assert rt.guess_kind(_api("Matte", {"video": ["9", 0]}), oi) == "postprocess"


def test_video_out_with_an_image_in_is_animate():
    oi = {"I2V": {"api_node": True, "output": ["VIDEO"],
                  "input": {"required": {"start_frame": ["IMAGE", {}]}}}}
    assert rt.guess_kind(_api("I2V", {"start_frame": ["9", 0]}), oi) == "animate"
    # text-to-video wires nothing: vorch's animate kind animates a keyframe, so it does not fit
    assert rt.guess_kind(_api("I2V", {}), oi) == "custom"


# --- the aspect table -------------------------------------------------------------

@pytest.mark.parametrize("options,expect", [
    (["1024x1024", "1024x1792", "1792x1024"],
     {"1:1": "1024x1024", "9:16": "1024x1792", "16:9": "1792x1024"}),
    (["1920:1080", "1080:1920", "1080:1080"],                      # colon-separated pixels
     {"16:9": "1920:1080", "9:16": "1080:1920", "1:1": "1080:1080"}),
    (["auto", "1:1", "16:9", "9:16", "Custom"],                    # bare ratios, junk ignored
     {"1:1": "1:1", "16:9": "16:9", "9:16": "9:16"}),
    (["2560x1440 (16:9)", "1440x2560 (9:16)"],                     # decorated labels
     {"16:9": "2560x1440 (16:9)", "9:16": "1440x2560 (9:16)"}),
])
def test_aspect_table_reads_every_label_dialect(options, expect):
    assert rt.aspect_table(options) == expect


def test_aspect_table_prefers_the_largest_matching_label():
    assert rt.aspect_table(["896x512", "1792x1024"])["16:9"] == "1792x1024"


def test_aspect_table_omits_aspects_the_node_cannot_express():
    table = rt.aspect_table(["1024x1024"])
    assert table == {"1:1": "1024x1024"}
    assert "16:9" not in table  # the workflow will refuse a 16:9 job rather than render square


def test_label_ratio_ignores_what_it_cannot_parse():
    assert rt.label_ratio("auto") is None and rt.label_ratio("Custom") is None
    assert rt.label_ratio(None) is None


# --- the generated YAML -----------------------------------------------------------

def test_a_hosted_keyframe_spec_is_loadable_and_carries_the_aspect_table():
    wf = _api("FakeT2I", {"prompt": "", "seed": 0, "size": "1024x1024", "n": 1})
    wf["2"] = {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}
    doc, _ = rt.guess_spec(wf, name="fake", kind="keyframe", template="fake.json", object_info=OI)
    p = doc["params"]
    assert p["positive"]["field"] == "prompt"
    assert p["aspect"]["from_aspect"] == {"1:1": "1024x1024", "9:16": "1024x1792",
                                          "16:9": "1792x1024"}
    # size is an enum here, so the contract's pixel params are declared but wired nowhere
    assert p["width"]["targets"] == [] and p["height"]["targets"] == []
    # and the result actually satisfies the keyframe contract, which is the point of the fillers
    assert registry._contract_problems(_parsed(doc)) == []


def _parsed(doc: dict):
    """Parse a guessed doc. ``parse_spec`` resolves the template beside its source, so point
    both at a real shipped template — the graph's contents are irrelevant to the contract."""
    from vorch import config

    tpl = config.WORKFLOW_DIR / "image_flux2_klein_text_to_image.json"
    return registry.parse_spec({**doc, "template": tpl.name}, source=tpl.with_suffix(".yaml"))
