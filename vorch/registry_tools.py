"""Helpers for registering a *new* ComfyUI workflow without writing Python.

``inspect_template`` lists every node with its literal inputs and marks the
usual suspects (prompts, seeds, sizes, loaders, save nodes); ``guess_spec``
turns those guesses into a registry YAML document with the canonical parameter
names each kind expects (see ``registry.KIND_CONTRACTS``). Both are heuristics:
``vorch registry validate --online`` is the safety net.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import registry

INPUT_CLASSES = {  # class_type -> (field, media type)
    "LoadImage": ("image", "image"),
    "LoadImageMask": ("image", "image"),
    "LoadVideo": ("file", "video"),
    "VHS_LoadVideo": ("video", "video"),
    "VHS_LoadVideoPath": ("video", "video"),
}
SAVE_CLASSES = {  # class_type -> output media type
    "SaveImage": "image",
    "SaveAnimatedWEBP": "image",
    "SaveAnimatedPNG": "image",
    "SaveVideo": "video",
    "VHS_VideoCombine": "video",
}
SAMPLER_CLASSES = ("KSampler", "SamplerCustom")
SIZE_CLASS_RE = re.compile(r"(Empty.*Latent|ImageToVideo|Scheduler|EmptyImage)", re.I)
NEGATIVE_RE = re.compile(
    r"\b(distorted|deformed|blur|blurry|ugly|worst quality|low quality|jitter|morph|"
    r"bad anatomy|watermark|airbrushed)\b", re.I,
)
LOADER_PARAM = {
    "unet_name": "unet", "ckpt_name": "checkpoint", "clip_name": "clip", "clip_name1": "clip_1",
    "clip_name2": "clip_2", "vae_name": "vae", "lora_name": "lora", "model_name": "model",
    "swap_model": "swap_model", "face_restore_model": "restore_model", "upscale_model": "upscale_model",
}


# --- UI -> API format conversion -------------------------------------------------

class ConversionError(ValueError):
    """The UI export cannot be turned into an API-format graph."""


# Primitive input types are drawn as widgets and their values live positionally in
# ``widgets_values``. Everything else (IMAGE, MASK, MODEL, LATENT, …) is a socket fed by a link.
_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})
# The frontend inserts a "control_after_generate" widget right after a seed. It is UI-only and
# has no counterpart in the API format, so its value has to be stepped over.
_CONTROL_VALUES = frozenset({"randomize", "increment", "decrement", "fixed"})
_SEED_NAMES = frozenset({"seed", "noise_seed", "rand_seed"})
_SKIP_MODES = frozenset({2, 4})  # muted / bypassed nodes never run
# Frontend-only nodes: they carry annotations for the human reading the graph and never execute,
# so they are absent from /object_info by design rather than missing.
_ANNOTATION_CLASSES = frozenset({"Note", "MarkdownNote"})
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _is_widget(spec: Any) -> bool:
    if not isinstance(spec, (list, tuple)) or not spec:
        return False
    t = spec[0]
    if isinstance(t, list):        # V1 combo: the options are inlined
        return True
    return isinstance(t, str) and (t in _WIDGET_TYPES or "DYNAMICCOMBO" in t)


def _dynamic_options(spec: Any) -> list[dict] | None:
    """The option list of a COMFY_DYNAMICCOMBO_V3 input, else None."""
    if not (isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], str)):
        return None
    if "DYNAMICCOMBO" not in spec[0] or len(spec) < 2 or not isinstance(spec[1], dict):
        return None
    opts = spec[1].get("options")
    return opts if isinstance(opts, list) else None


def _ordered_inputs(info: dict) -> list[tuple[str, Any]]:
    inp = info.get("input") or {}
    out: list[tuple[str, Any]] = []
    for section in ("required", "optional"):
        out += list((inp.get(section) or {}).items())
    return out


def _consume_widgets(
    entries: list[tuple[str, Any]], values: list, cursor: list[int],
    into: dict, *, prefix: str, skip: set[str],
) -> None:
    """Assign ``values`` positionally to the widget inputs in ``entries``.

    ``cursor`` is a one-element list used as a shared index. Dynamic combos recurse: the
    selected option's nested widgets follow the parent inline, and are submitted as flat
    dotted keys (``moderation.visual_output_moderation``), which is what ComfyUI expects.
    """
    for name, spec in entries:
        if not _is_widget(spec) or name in skip:
            continue
        if cursor[0] >= len(values):
            return  # trailing optional widgets were left off the export
        value = values[cursor[0]]
        cursor[0] += 1
        key = f"{prefix}{name}"
        into[key] = value
        if name in _SEED_NAMES and cursor[0] < len(values) and values[cursor[0]] in _CONTROL_VALUES:
            cursor[0] += 1  # step over control_after_generate
        options = _dynamic_options(spec)
        if options is None:
            continue
        chosen = next((o for o in options if isinstance(o, dict) and o.get("key") == value), None)
        if chosen is None:
            continue  # an option this ComfyUI no longer offers; leave the parent value alone
        nested = chosen.get("inputs") or {}
        entries_n: list[tuple[str, Any]] = []
        for section in ("required", "optional"):
            entries_n += list((nested.get(section) or {}).items())
        _consume_widgets(entries_n, values, cursor, into, prefix=f"{key}.", skip=set())


def ui_to_api(doc: dict, object_info: dict) -> dict:
    """Convert a ComfyUI **UI export** into the API format vorch's registry needs.

    The two differ in more than shape: the UI form keeps widget values in a positional list
    per node and wires sockets through a separate ``links`` array, while the API form is a flat
    dict keyed by node id whose ``inputs`` name every value. Widget *order* is not recorded in
    the export at all, so the node's schema from ``/object_info`` is required to line the
    values up — which is why importing needs a running ComfyUI.
    """
    if "nodes" not in doc:
        return doc  # already API format
    if not isinstance(doc.get("nodes"), list):
        raise ConversionError("`nodes` is not a list; this is not a ComfyUI UI export")
    links: dict[int, list] = {}
    for link in doc.get("links") or []:
        if isinstance(link, (list, tuple)) and len(link) >= 5:
            links[link[0]] = [str(link[1]), link[2]]

    out: dict[str, dict] = {}
    unknown: list[str] = []
    for node in doc["nodes"]:
        if not isinstance(node, dict) or node.get("mode") in _SKIP_MODES:
            continue
        cls, nid = node.get("type"), node.get("id")
        if not cls or nid is None or cls in _ANNOTATION_CLASSES:
            continue
        info = object_info.get(cls)
        if info is None:
            unknown.append(str(cls))
            continue
        inputs: dict[str, Any] = {}
        widget_slots: set[str] = set()
        for sock in node.get("inputs") or []:
            if not isinstance(sock, dict):
                continue
            # A widget dragged out into a socket is fed by its link, not by widgets_values.
            if sock.get("widget"):
                widget_slots.add(str(sock["widget"].get("name") or sock.get("name") or ""))
            lid = sock.get("link")
            if lid is not None and lid in links:
                inputs[str(sock.get("name"))] = list(links[lid])
        raw = node.get("widgets_values")
        if isinstance(raw, dict):           # some exports use a mapping instead of a list
            inputs.update({k: v for k, v in raw.items() if k not in widget_slots})
        else:
            _consume_widgets(_ordered_inputs(info), list(raw or []), [0], inputs,
                             prefix="", skip=widget_slots)
        entry: dict[str, Any] = {"class_type": cls, "inputs": inputs}
        if node.get("title"):
            entry["_meta"] = {"title": str(node["title"])}
        out[str(nid)] = entry

    if unknown:
        names = sorted(set(unknown))
        if any(_UUID_RE.match(n) for n in names):
            raise ConversionError(
                "this export references a subgraph/component by id "
                f"({', '.join(n for n in names if _UUID_RE.match(n))}) — vorch needs a flat graph, "
                "so open it in ComfyUI, expand the subgraph and export again"
            )
        raise ConversionError(
            "ComfyUI has no node class " + ", ".join(names)
            + " — install the custom nodes this workflow needs, then import it again"
        )
    if not out:
        raise ConversionError("the export has no runnable nodes")
    return out


@dataclass
class NodeInfo:
    node: str
    class_type: str
    title: str
    literals: dict[str, Any]
    candidates: dict[str, str] = field(default_factory=dict)  # field -> role
    options: dict[str, list] = field(default_factory=dict)    # field -> combo options (online)


def _title(node: dict) -> str:
    return str((node.get("_meta") or {}).get("title") or "")


def inspect_template(wf: dict, object_info: dict | None = None) -> list[NodeInfo]:
    """Per-node literal inputs + role guesses (input/prompt/seed/size/…)."""
    rows: list[NodeInfo] = []
    prompt_nodes: list[tuple[NodeInfo, str, str]] = []
    for node_id, node in wf.items():
        cls = node.get("class_type", "")
        lits = {k: v for k, v in (node.get("inputs") or {}).items() if not isinstance(v, list)}
        info = NodeInfo(str(node_id), cls, _title(node), lits)
        if cls in INPUT_CLASSES:
            fld, media = INPUT_CLASSES[cls]
            if fld in lits:
                info.candidates[fld] = f"input:{media}"
        if cls in SAVE_CLASSES:
            info.candidates["__output__"] = f"output:{SAVE_CLASSES[cls]}"
            if "filename_prefix" in lits:
                info.candidates["filename_prefix"] = "filename_prefix"
        # A hosted node carries the prompt and the size on itself, so the local rules below
        # (CLIPTextEncode, Empty*Latent, samplers) never fire for it.
        is_api = bool(object_info) and bool((object_info.get(cls) or {}).get("api_node"))
        if is_api:
            for fld, val in lits.items():
                if fld in API_FIELD_PARAM:
                    info.candidates[fld] = API_FIELD_PARAM[fld]
            size_field = next((f for f in API_SIZE_FIELDS if f in lits), None)
            if size_field and "width" not in lits:
                # Size is one of the node's own labels; mark it so guess_spec can build the
                # aspect table from the enum's options rather than inventing pixel params.
                info.candidates[size_field] = "aspect_enum"
        for fld, val in lits.items():
            if fld in info.candidates:
                continue
            if cls == "PrimitiveStringMultiline" and fld == "value" or cls == "CLIPTextEncode" and fld == "text":
                prompt_nodes.append((info, fld, str(val)))
            elif fld in ("noise_seed", "seed") and (cls == "RandomNoise" or cls.startswith(SAMPLER_CLASSES) or fld == "seed"):
                info.candidates[fld] = "seed"
            elif fld in ("width", "height") and SIZE_CLASS_RE.search(cls):
                info.candidates[fld] = fld
            elif cls == "PrimitiveInt" and fld == "value" and info.title.lower() in ("width", "height"):
                info.candidates[fld] = info.title.lower()
            elif fld == "length" and "ImageToVideo" in cls:
                info.candidates[fld] = "length"
            elif fld in ("steps", "cfg", "sampler_name", "scheduler", "denoise"):
                info.candidates[fld] = fld
            elif fld in ("fps", "frame_rate"):
                info.candidates[fld] = "fps"
            elif fld in registry.MODEL_FIELDS:
                info.candidates[fld] = f"loader:{fld}"
            elif fld == "batch_size":
                info.candidates[fld] = "batch_size"
            elif fld == "end_at_step" and lits.get("start_at_step") == 0:
                info.candidates[fld] = "split_step"      # first sampler of a cascade: 0..split
            elif fld == "start_at_step" and (lits.get("end_at_step") or 0) >= 1000:
                info.candidates[fld] = "split_step"      # last sampler: split..end (10000 sentinel)
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                info.candidates[fld] = "numeric"
        if object_info and cls in object_info:
            for fld in lits:
                opts = registry._combo_options(object_info, cls, fld)
                if opts is not None:
                    info.options[fld] = opts
        rows.append(info)
    # Positive / negative prompts: title, negative vocabulary, or "the empty one".
    nonempty = [t for _, _, t in prompt_nodes if t.strip()]
    positive_taken = False
    for info, fld, text in prompt_nodes:
        is_neg = ("negative" in info.title.lower()) or bool(NEGATIVE_RE.search(text)) or (
            not text.strip() and bool(nonempty)
        )
        if is_neg:
            info.candidates[fld] = "negative"
        elif not positive_taken:
            info.candidates[fld] = "positive"
            positive_taken = True
        else:
            info.candidates[fld] = "prompt"
    return rows


_MEDIA_TYPES = frozenset({"IMAGE", "MASK", "VIDEO", "AUDIO"})


def _nested_entries(spec: Any) -> list[tuple[str, Any]]:
    """The inputs a composite input can contribute, flattened.

    Two composites exist. A dynamic combo carries one nested schema per option. An autogrow
    (``COMFY_AUTOGROW_V3``, e.g. GPT-Image's 16 reference slots) carries a single ``template``
    schema plus the generated names it repeats it under — so ``images.image_1`` has the type of
    the template's lone input.
    """
    out: list[tuple[str, Any]] = []
    for opt in _dynamic_options(spec) or []:
        nested = opt.get("inputs") or {} if isinstance(opt, dict) else {}
        for section in ("required", "optional"):
            out += list((nested.get(section) or {}).items())
    if (isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[0], str)
            and "AUTOGROW" in spec[0] and isinstance(spec[1], dict)):
        tpl = spec[1].get("template") or {}
        inner = tpl.get("input") or {}
        specs = [v for section in ("required", "optional")
                 for v in (inner.get(section) or {}).values()]
        if specs:
            out += [(str(n), specs[0]) for n in (tpl.get("names") or [])]
    return out


def _declared_type(declared: dict, field: str) -> str | None:
    """The declared type of ``field``, which may be a dotted dynamic-combo path.

    A dynamic combo submits its selected option's inputs as flat dotted keys
    (``model.images.image_1``), so the plain schema has no such key. Resolve by walking the
    option trees for the leaf name — the nesting can be several levels deep and an autogrow
    input invents numbered names, so searching beats reconstructing the exact path.
    """
    spec = declared.get(field)
    if spec is None and "." in field:
        leaf = field.rsplit(".", 1)[1]
        queue = list(declared.items())
        seen = 0
        while queue and seen < 2000:
            name, candidate = queue.pop(0)
            seen += 1
            if name == leaf:
                spec = candidate
                break
            queue += _nested_entries(candidate)
    if not (isinstance(spec, (list, tuple)) and spec) or not isinstance(spec[0], str):
        return None
    return spec[0].split(",")[0].strip() or None   # some inputs accept a union of types


def price_badge_cost(info: dict | None) -> "registry.Cost | None":
    """The price a node declares, but only when it declares a plain number.

    ComfyUI expresses this as a JSONata expression evaluated by its own frontend. Roughly a
    third are literals; the rest do megapixel maths or branch on the chosen model, and vorch has
    no JSONata engine (adding one for price estimation would dwarf its six runtime deps). So the
    literals are read and everything else is left for a human to declare — which is honest, and
    is why a missing cost is a warning rather than a silent zero.
    """
    expr = ((info or {}).get("price_badge") or {}).get("expr")
    if not isinstance(expr, str):
        return None
    text = expr.strip()
    if any(tok in text for tok in ("widgets", "inputs", "$", "?")):
        return None  # an expression, not a literal
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        if data.get("type") == "usd" and isinstance(data.get("usd"), (int, float)):
            return registry.parse_cost("price_badge", {"usd": float(data["usd"])})
        if data.get("type") == "range_usd":
            lo, hi = data.get("min_usd"), data.get("max_usd")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                return registry.parse_cost("price_badge", {"min_usd": float(lo), "max_usd": float(hi)})
    except registry.RegistryError:
        return None
    return None


API_FIELD_PARAM = {
    "prompt": "positive",
    "negative_prompt": "negative",
    "width": "width",
    "height": "height",
    "seed": "seed",
}
# Size-as-an-enum fields, in the order we would rather use them: a literal WxH list beats a
# bare ratio, because it also tells us the resolution.
API_SIZE_FIELDS = ("resolution", "size", "size_preset", "aspect_ratio", "ratio")
_LABEL_RE = re.compile(r"(\d{2,5})\s*[x:×]\s*(\d{2,5})")
_RATIO_RE = re.compile(r"^(\d{1,2})\s*:\s*(\d{1,2})$")


def label_ratio(label: str) -> float | None:
    """The aspect ratio a node's own size label describes, or None.

    The dialects seen across providers: ``1792x1024``, ``1080:1920``, ``16:9``,
    ``2560x1440 (16:9)``, ``9:19.5``. A pixel pair is read as pixels; a small pair as a ratio.
    """
    if not isinstance(label, str):
        return None
    if m := _LABEL_RE.search(label):
        w, h = int(m.group(1)), int(m.group(2))
        return w / h if h else None
    if m := _RATIO_RE.match(label.strip()):
        w, h = int(m.group(1)), int(m.group(2))
        return w / h if h else None
    return None


def aspect_table(options: Sequence[str], *, tolerance: float = 0.02) -> dict[str, str]:
    """Map vorch's aspects onto a node's own size labels.

    For each aspect, the label whose ratio matches within ``tolerance`` and which has the most
    pixels (so 1792x1024 wins over 896x512). Aspects the node cannot express are simply absent,
    which is what makes the workflow refuse them later instead of rendering something square.
    """
    from . import formats

    out: dict[str, str] = {}
    for aspect, (aw, ah) in formats.ASPECTS.items():
        want = aw / ah
        best: tuple[int, str] | None = None
        for label in options:
            got = label_ratio(label)
            if got is None or abs(got - want) / want > tolerance:
                continue
            m = _LABEL_RE.search(label)
            size = int(m.group(1)) * int(m.group(2)) if m else 0
            if best is None or size > best[0]:
                best = (size, label)
        if best is not None:
            out[aspect] = best[1]
    return out


def api_node_role(wf: dict, object_info: dict | None) -> str | None:
    """The kind implied by a hosted API node, or None if the graph has none.

    A hosted graph is unlike a local one: no Empty*Latent, no sampler, and the prompt sits on
    the API node itself rather than a CLIPTextEncode — so the local heuristics below all fall
    through to "custom". What it does have is a node ComfyUI flags ``api_node``.

    Two things decide the kind, and both are read from data rather than from names:

    * the node's **first** output, which is its product. Several nodes trail an IMAGE preview
      after the real thing — a 3D generator returns ``[STRING, FILE_3D_GLB, FILE_3D_OBJ, IMAGE]``
      and is not an image generator at all.
    * which media inputs the graph actually **wires**. Many of these classes do several jobs:
      IdeogramV3 declares optional ``image``/``mask`` yet its t2i template connects none, and
      the names vary (Kling calls its still ``start_frame``). A wired image is what separates an
      edit from a text-to-image; a wired video means the graph post-processes footage.
    """
    if not object_info:
        return None
    for node in wf.values():
        info = object_info.get(node.get("class_type", "")) or {}
        if not info.get("api_node"):
            continue
        inp = info.get("input") or {}
        declared = {**(inp.get("required") or {}), **(inp.get("optional") or {})}
        wired = set()
        for field, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and (t := _declared_type(declared, str(field))) in _MEDIA_TYPES:
                wired.add(t)
        outputs = [str(o) for o in (info.get("output") or [])]
        primary = outputs[0] if outputs else None
        if primary == "VIDEO":
            if wired & {"IMAGE"}:
                return "animate"
            return "postprocess" if "VIDEO" in wired else "custom"
        if primary == "IMAGE":
            if "VIDEO" in wired:
                return "postprocess"       # footage in, frames/matte out
            return "edit" if wired & {"IMAGE", "MASK"} else "keyframe"
        return "custom"                    # 3D, audio, text: not a kind vorch models
    return None


def guess_kind(wf: dict, object_info: dict | None = None) -> str:
    classes = {n.get("class_type", "") for n in wf.values()}
    has_load_image = bool(classes & {"LoadImage", "LoadImageMask"})
    has_load_video = bool(classes & {"LoadVideo", "VHS_LoadVideo", "VHS_LoadVideoPath"})
    has_sampler = any(c.startswith(SAMPLER_CLASSES) for c in classes)
    if (role := api_node_role(wf, object_info)) is not None:
        return role
    if any("ImageToVideo" in c for c in classes):
        return "animate"
    if "ReActorFaceSwap" in classes and has_load_image and not has_sampler:
        return "faceswap"
    if "ReferenceLatent" in classes:
        return "edit"
    if has_load_video:
        return "postprocess"
    if any(re.search(r"Empty.*Latent", c) for c in classes) and not has_load_image:
        return "keyframe"
    if has_load_image and (classes & set(SAVE_CLASSES)) and not has_sampler:
        return "postprocess"
    return "custom"


def _input_names(kind: str, media: str, index: int) -> str:
    if media == "video":
        return "input_video" if index == 0 else f"input_video_{index + 1}"
    if kind == "animate":
        return "start_image" if index == 0 else f"image_{index + 1}"
    if kind == "faceswap":
        return ("input_image", "source_image")[index] if index < 2 else f"image_{index + 1}"
    if kind == "edit":
        return "reference" if index == 0 else f"reference_{index + 1}"
    if kind == "postprocess":
        return "input_image" if index == 0 else f"image_{index + 1}"
    return "image" if index == 0 else f"image_{index + 1}"


def guess_spec(wf: dict, *, name: str, kind: str | None = None, template: str = "",
               object_info: dict | None = None) -> tuple[dict, list[str]]:
    """A registry YAML document guessed from the template, plus notes for the human.

    Returns ``(doc, notes)``; ``notes`` lists literal numeric fields that were not
    mapped (the user can promote them to params by hand).
    """
    kind = kind or guess_kind(wf, object_info)
    rows = inspect_template(wf, object_info)
    params: dict[str, dict] = {}
    inputs: dict[str, str] = {}
    out_type: str | None = None
    out_nodes: list[str] = []
    notes: list[str] = []
    fanout: dict[str, list[dict]] = {}
    fanout_meta: dict[str, dict] = {}
    n_img = n_vid = 0
    used_names: set[str] = set()

    def add(pname: str, node: str, fld: str, **meta) -> None:
        if pname in used_names or pname in params:
            i = 2
            while f"{pname}_{i}" in params or f"{pname}_{i}" in used_names:
                i += 1
            pname = f"{pname}_{i}"
        used_names.add(pname)
        params[pname] = {"node": node, "field": fld, **meta}

    def add_fanout(pname: str, node: str, fld: str, **meta) -> None:
        fanout.setdefault(pname, []).append({"node": node, "field": fld})
        fanout_meta.setdefault(pname, meta)

    for info in rows:
        for fld, role in info.candidates.items():
            if role.startswith("input:"):
                media = role.split(":", 1)[1]
                idx = n_vid if media == "video" else n_img
                pname = _input_names(kind, media, idx)
                if media == "video":
                    n_vid += 1
                else:
                    n_img += 1
                add(pname, info.node, fld, type="str", required=True)
                inputs[pname] = media
            elif role.startswith("output:"):
                out_type = out_type or role.split(":", 1)[1]
                out_nodes.append(info.node)
            elif role == "filename_prefix":
                if "filename_prefix" not in params:
                    add("filename_prefix", info.node, fld, type="str")
            elif role in ("positive", "negative", "prompt"):
                add(role, info.node, fld, type="str", **({"required": True} if role == "positive" else {}))
            elif role == "seed":
                add_fanout("seed", info.node, fld, type="int", required=kind in ("keyframe", "edit"))
            elif role in ("width", "height"):
                # A hosted node states its own range; the local 16-alignment is not a given.
                meta = {"type": "int"} if info.class_type in (object_info or {}) and (
                    (object_info or {}).get(info.class_type, {}).get("api_node")
                ) else {"type": "int", "constraint": "multiple_of_16"}
                add_fanout(role, info.node, fld, **meta)
            elif role == "aspect_enum":
                table = aspect_table(info.options.get(fld) or [])
                meta: dict[str, Any] = {"type": "enum"}
                if info.class_type:
                    meta["enum_from"] = {"class_type": info.class_type, "input": fld}
                if table:
                    meta["from_aspect"] = table
                add("aspect", info.node, fld, **meta)
            elif role == "length":
                add("length", info.node, fld, type="int", constraint="4n+1", required=True)
            elif role in ("steps", "cfg", "denoise"):
                add_fanout(role, info.node, fld, type="int" if role == "steps" else "float")
            elif role in ("sampler_name", "scheduler"):
                if role == "sampler_name":
                    add("sampler", info.node, fld, type="enum",
                        enum_from={"class_type": info.class_type, "input": fld})
                else:
                    add("scheduler", info.node, fld, type="enum",
                        enum_from={"class_type": info.class_type, "input": fld})
            elif role == "fps":
                add("fps", info.node, fld, type="int", min=1, max=60)
            elif role == "batch_size":
                add("batch_size", info.node, fld, type="int", min=1)
            elif role == "split_step":
                add_fanout("split_step", info.node, fld, type="int", min=1)
            elif role.startswith("loader:"):
                lf = role.split(":", 1)[1]
                add(LOADER_PARAM.get(lf, lf), info.node, fld, type="enum",
                    enum_from={"class_type": info.class_type, "input": fld})
            elif role == "numeric":
                notes.append(f"node {info.node} ({info.class_type}).{fld} = {info.literals[fld]!r}")
    for pname, targets in fanout.items():
        meta = fanout_meta[pname]
        if len(targets) == 1:
            params[pname] = {**targets[0], **meta}
        else:
            params[pname] = {"targets": targets, **meta}
    # Whatever the kind requires and the graph cannot offer is declared with no target: it
    # keeps the entry loadable and still type-checked, and is the honest record that this
    # model has no such control (a hosted node often has no seed, and none has pixel sizes
    # when it takes a size enum). Without this a generated hosted entry fails to load at all.
    required = registry.KIND_CONTRACTS[kind].required if kind in registry.KIND_CONTRACTS else frozenset()
    for missing in sorted(required - set(params)):
        if missing in ("width", "height", "seed"):
            params[missing] = {"targets": [], "type": "int"}
        elif missing == "positive":
            params[missing] = {"targets": [], "type": "str"}

    # Contract-driven ordering: required names first, then the rest as found.
    doc: dict[str, Any] = {"name": name, "kind": kind, "template": template or "<template>.json"}
    doc["description"] = "TODO: describe what this workflow does"
    doc["tags"] = []
    if inputs:
        doc["inputs"] = inputs
    doc["outputs"] = {"type": out_type or ("video" if kind == "animate" else "image"), "nodes": out_nodes}
    doc["params"] = params
    return doc, notes


def format_inspect(rows: list[NodeInfo]) -> str:
    lines = []
    for r in rows:
        head = f"{r.node:>8}  {r.class_type:<28} {r.title!r}"
        lines.append(head)
        for fld, val in r.literals.items():
            role = r.candidates.get(fld, "")
            if fld == "__output__":
                continue
            shown = repr(val)
            if len(shown) > 48:
                shown = shown[:45] + "...'"
            extra = f"   <- {role}" if role else ""
            if fld in r.options:
                opts = r.options[fld]
                extra += f"   [{len(opts)} installed option(s)" + (
                    "" if val in opts else ", VALUE NOT INSTALLED") + "]"
            lines.append(f"            .{fld} = {shown}{extra}")
        if "__output__" in r.candidates:
            lines.append(f"            -> {r.candidates['__output__']}")
    return "\n".join(lines)
