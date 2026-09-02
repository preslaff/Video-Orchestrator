"""Helpers for registering a *new* ComfyUI workflow without writing Python.

``inspect_template`` lists every node with its literal inputs and marks the
usual suspects (prompts, seeds, sizes, loaders, save nodes); ``guess_spec``
turns those guesses into a registry YAML document with the canonical parameter
names each kind expects (see ``registry.KIND_CONTRACTS``). Both are heuristics:
``vorch registry validate --online`` is the safety net.
"""

from __future__ import annotations

import re
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


def guess_kind(wf: dict) -> str:
    classes = {n.get("class_type", "") for n in wf.values()}
    has_load_image = bool(classes & {"LoadImage", "LoadImageMask"})
    has_load_video = bool(classes & {"LoadVideo", "VHS_LoadVideo", "VHS_LoadVideoPath"})
    has_sampler = any(c.startswith(SAMPLER_CLASSES) for c in classes)
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


def guess_spec(wf: dict, *, name: str, kind: str | None = None, template: str = "") -> tuple[dict, list[str]]:
    """A registry YAML document guessed from the template, plus notes for the human.

    Returns ``(doc, notes)``; ``notes`` lists literal numeric fields that were not
    mapped (the user can promote them to params by hand).
    """
    kind = kind or guess_kind(wf)
    rows = inspect_template(wf)
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
                add_fanout(role, info.node, fld, type="int", constraint="multiple_of_16")
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
