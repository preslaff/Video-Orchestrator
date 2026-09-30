"""Declarative ComfyUI workflow registry — node surgery driven by YAML, not Python.

Every workflow is an API-format template (``*.json``, flat dict keyed by node
id) plus a sidecar ``*.yaml`` that names the fields the orchestrator may touch::

    name: animate
    kind: animate                      # keyframe|animate|faceswap|edit|postprocess|custom
    template: wan22_i2v_gguf.json      # relative to the YAML's directory
    backend: local                     # which ComfyUI runs it (vorch.backends; default local)
    inputs: {start_image: image}       # params that are files staged via /upload/image
    outputs: {type: video, nodes: ["13"]}
    params:
      positive: {node: "6", field: text, type: str, required: true}
      length:   {node: "8", field: length, type: int, constraint: 4n+1}
      seed:     {targets: [{node: "9", field: noise_seed}, {node: "10", field: noise_seed}], type: int}

Search order: ``config.WORKFLOW_DIR`` (built-ins shipped with the repo), then
``config.USER_WORKFLOW_DIR`` (``~/vorch/workflows``) where a same-named entry
overrides the built-in. Dropping a JSON + YAML pair there is all it takes to
use a newly installed ComfyUI workflow (``vorch registry inspect/add`` helps
write the YAML). Templates are never restructured — only listed literal fields
are replaced (SKILL.md invariant 2).

Each *kind* has a contract (``KIND_CONTRACTS``): the parameter names the
pipeline relies on, so any workflow of that kind can be driven the same way.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import backends as backends_mod
from . import config

log = logging.getLogger("vorch.registry")

KINDS = ("keyframe", "animate", "faceswap", "edit", "postprocess", "custom")
PARAM_TYPES = ("int", "float", "str", "bool", "enum", "any")
INPUT_TYPES = ("image", "video")
OUTPUT_TYPES = ("image", "video")

VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".gif"}

# Loader inputs whose literal value names a model file (env check: installed?).
MODEL_FIELDS = {
    "unet_name", "clip_name", "clip_name1", "clip_name2", "vae_name", "ckpt_name",
    "lora_name", "model_name", "swap_model", "face_restore_model", "upscale_model",
}


class RegistryError(ValueError):
    """A workflow YAML is invalid (aggregated problems in the message)."""


# --- constraints ----------------------------------------------------------------


def _c_kn1(k: int, example: str) -> Callable[[Any], None]:
    def check(v: Any) -> None:
        if not isinstance(v, int) or v < 1 or v % k != 1:
            raise ValueError(f"length must be {k}n+1 ({example}), got {v!r}")
    return check


def _c_mult(n: int) -> Callable[[Any], None]:
    def check(v: Any) -> None:
        if not isinstance(v, int) or v <= 0 or v % n != 0:
            raise ValueError(f"must be a positive multiple of {n}, got {v!r}")
    return check


def _c_positive(v: Any) -> None:
    if not isinstance(v, (int, float)) or v <= 0:
        raise ValueError(f"must be > 0, got {v!r}")


CONSTRAINTS: dict[str, Callable[[Any], None]] = {
    "4n+1": _c_kn1(4, "49, 81, 121, ..."),      # Wan
    "8n+1": _c_kn1(8, "97, 121, 145, ..."),     # LTX-2.x
    "multiple_of_32": _c_mult(32),              # LTX-2.x sizes
    "multiple_of_16": _c_mult(16),
    "multiple_of_8": _c_mult(8),
    "positive": _c_positive,
}
LENGTH_CONSTRAINTS = ("4n+1", "8n+1")
SIZE_CONSTRAINTS = {"multiple_of_8": 8, "multiple_of_16": 16, "multiple_of_32": 32}


@dataclass(frozen=True)
class Contract:
    required: frozenset[str]
    optional: frozenset[str]
    min_image_inputs: int = 0
    min_inputs: int = 0


KIND_CONTRACTS: dict[str, Contract] = {
    "keyframe": Contract(
        frozenset({"positive", "seed", "width", "height"}),
        frozenset({"aspect", "negative", "steps", "cfg", "unet", "batch_size", "sampler",
                   "filename_prefix"}),
    ),
    "animate": Contract(
        frozenset({"start_image", "positive", "length"}),
        frozenset({"negative", "width", "height", "fps", "seed", "steps", "cfg",
                   "split_step", "filename_prefix"}),
        min_image_inputs=1,
    ),
    "faceswap": Contract(
        frozenset({"input_image", "source_image"}),
        frozenset({"restore_model", "filename_prefix"}),
        min_image_inputs=2,
    ),
    "edit": Contract(
        frozenset({"positive", "seed"}),
        frozenset({"width", "height", "steps", "cfg", "unet", "negative", "batch_size",
                   "sampler", "filename_prefix"}),
        min_image_inputs=1,
    ),
    "postprocess": Contract(frozenset(), frozenset({"fps", "seed", "filename_prefix"}), min_inputs=1),
    "custom": Contract(frozenset(), frozenset()),
}

# Cross-parameter checks per kind, run after coercion (log-only; never raise here).
KIND_HOOKS: dict[str, Callable[["WorkflowSpec", dict], None]] = {}


# --- spec dataclasses ------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    node: str
    field: str


@dataclass
class ParamSpec:
    name: str
    targets: list[Target]
    type: str = "any"
    default: Any = None
    required: bool = False
    constraint: str | None = None
    min: float | None = None
    max: float | None = None
    enum_from: dict | None = None  # {class_type, input}
    # Hosted API nodes usually express size as an enum of their own dialect ("1792x1024",
    # "16:9", "1080:1920") rather than width/height ints. This maps the job's aspect onto
    # whatever label this node uses; an aspect the table has no entry for is refused.
    from_aspect: dict[str, str] | None = None

    def coerce(self, value: Any) -> Any:
        if self.from_aspect is not None:
            key = str(value)
            if key not in self.from_aspect:
                raise ValueError(
                    f"param {self.name!r}: this workflow cannot render aspect {key!r} "
                    f"(it offers {', '.join(sorted(self.from_aspect))})"
                )
            value = self.from_aspect[key]
        try:
            if self.type == "int":
                if isinstance(value, bool):
                    raise TypeError
                value = int(value)
            elif self.type == "float":
                value = float(value)
            elif self.type == "str":
                value = str(value)
            elif self.type == "bool":
                if isinstance(value, str):
                    value = value.strip().lower() in ("1", "true", "yes", "on")
                else:
                    value = bool(value)
        except (TypeError, ValueError):
            raise ValueError(f"param {self.name!r}: expected {self.type}, got {value!r}")
        if self.min is not None and value < self.min:
            raise ValueError(f"param {self.name!r}: {value!r} < min {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"param {self.name!r}: {value!r} > max {self.max}")
        if self.constraint:
            try:
                CONSTRAINTS[self.constraint](value)
            except ValueError as e:
                raise ValueError(f"param {self.name!r}: {e}") from None
        return value


@dataclass
class WorkflowSpec:
    name: str
    kind: str
    template: Path
    params: dict[str, ParamSpec]
    inputs: dict[str, str] = field(default_factory=dict)      # param -> image|video
    output_type: str | None = None
    output_nodes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ""
    timeout: int | None = None
    backend: str = backends_mod.LOCAL                       # ComfyUI that runs it (backends.yaml)
    source: Path | None = None
    builtin: bool = True
    requires_nodes: set[str] = field(default_factory=set)      # all class_types in the template
    requires_models: list[tuple[str, str, str, str]] = field(default_factory=list)  # (node, class, field, value)

    def image_inputs(self) -> list[str]:
        return [p for p, t in self.inputs.items() if t == "image"]

    def video_inputs(self) -> list[str]:
        return [p for p, t in self.inputs.items() if t == "video"]

    def accepts(self, name: str) -> bool:
        return name in self.params

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "template": self.template.name,
            "description": self.description,
            "tags": list(self.tags),
            "timeout": self.timeout,
            "backend": self.backend,
            "builtin": self.builtin,
            "source": str(self.source) if self.source else None,
            "inputs": dict(self.inputs),
            "outputs": {"type": self.output_type, "nodes": list(self.output_nodes)},
            "params": {
                n: {
                    "targets": [{"node": t.node, "field": t.field} for t in p.targets],
                    "type": p.type,
                    "default": p.default,
                    "required": p.required,
                    "constraint": p.constraint,
                    "min": p.min,
                    "max": p.max,
                    "enum_from": p.enum_from,
                    "from_aspect": p.from_aspect,
                }
                for n, p in self.params.items()
            },
            "requires": {
                "nodes": sorted(self.requires_nodes),
                "models": [
                    {"node": n, "class_type": c, "field": f, "value": v}
                    for n, c, f, v in self.requires_models
                ],
            },
        }


_ASPECT_RE = re.compile(r"^\d+:\d+$")


# --- parsing ---------------------------------------------------------------------


def _parse_param(name: str, raw: Any) -> ParamSpec:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:  # shorthand ["node", "field"]
        raw = {"node": raw[0], "field": raw[1]}
    if not isinstance(raw, dict):
        raise RegistryError(f"param {name!r}: expected a mapping")
    if "targets" in raw:
        targets = [Target(str(t["node"]), str(t["field"])) for t in raw["targets"]]
    elif "node" in raw and "field" in raw:
        targets = [Target(str(raw["node"]), str(raw["field"]))]
    else:
        raise RegistryError(f"param {name!r}: needs node+field or targets")
    ptype = str(raw.get("type", "any"))
    if ptype not in PARAM_TYPES:
        raise RegistryError(f"param {name!r}: unknown type {ptype!r}")
    from_aspect = raw.get("from_aspect")
    if from_aspect is not None:
        if not isinstance(from_aspect, dict) or not from_aspect:
            raise RegistryError(f"param {name!r}: from_aspect must be a non-empty mapping")
        # Keys are vorch aspects ("16:9"); values are this node's own labels. Checked by shape
        # rather than against formats.ASPECTS, because formats imports this module.
        for k, v in from_aspect.items():
            if not _ASPECT_RE.match(str(k)):
                raise RegistryError(f"param {name!r}: from_aspect key {k!r} is not an aspect (want W:H)")
            if not isinstance(v, str) or not v.strip():
                raise RegistryError(f"param {name!r}: from_aspect[{k!r}] must be a non-empty string")
        from_aspect = {str(k): v for k, v in from_aspect.items()}
    return ParamSpec(
        name=name,
        targets=targets,
        type=ptype,
        default=raw.get("default"),
        required=bool(raw.get("required", False)),
        constraint=raw.get("constraint"),
        min=raw.get("min"),
        max=raw.get("max"),
        enum_from=raw.get("enum_from"),
        from_aspect=from_aspect,
    )


def parse_spec(doc: dict, *, source: Path | None = None, builtin: bool = True) -> WorkflowSpec:
    """Build a WorkflowSpec from a YAML document (no template validation yet)."""
    if not isinstance(doc, dict):
        raise RegistryError(f"{source}: YAML root must be a mapping")
    name = str(doc.get("name") or (source.stem if source else "")).strip()
    if not name:
        raise RegistryError(f"{source}: missing name")
    kind = str(doc.get("kind", "custom"))
    if kind not in KINDS:
        raise RegistryError(f"{name}: unknown kind {kind!r} (want one of {KINDS})")
    tpl = doc.get("template")
    if not tpl:
        raise RegistryError(f"{name}: missing template")
    template = Path(tpl)
    if not template.is_absolute():
        template = (source.parent if source else Path(config.WORKFLOW_DIR)) / template
    params = {str(k): _parse_param(str(k), v) for k, v in (doc.get("params") or {}).items()}
    for p in params.values():
        if p.constraint and p.constraint not in CONSTRAINTS:
            raise RegistryError(f"{name}.{p.name}: unknown constraint {p.constraint!r}")
    inputs = {str(k): str(v) for k, v in (doc.get("inputs") or {}).items()}
    for k, v in inputs.items():
        if k not in params:
            raise RegistryError(f"{name}: input {k!r} is not a declared param")
        if v not in INPUT_TYPES:
            raise RegistryError(f"{name}: input {k!r} has unknown type {v!r}")
    backend = str(doc.get("backend") or backends_mod.LOCAL)
    if not backends_mod.NAME_RE.match(backend):
        raise RegistryError(f"{name}: invalid backend name {backend!r}")
    outputs = doc.get("outputs") or {}
    out_type = outputs.get("type")
    if out_type is not None and out_type not in OUTPUT_TYPES:
        raise RegistryError(f"{name}: outputs.type {out_type!r} not in {OUTPUT_TYPES}")
    spec = WorkflowSpec(
        name=name,
        kind=kind,
        template=template,
        params=params,
        inputs=inputs,
        output_type=out_type,
        output_nodes=[str(n) for n in (outputs.get("nodes") or [])],
        tags=[str(t) for t in (doc.get("tags") or [])],
        description=str(doc.get("description") or ""),
        timeout=int(doc["timeout"]) if doc.get("timeout") is not None else None,
        backend=backend,
        source=source,
        builtin=builtin,
    )
    req = doc.get("requires") or {}
    spec.requires_nodes.update(str(n) for n in (req.get("nodes") or []))
    spec._explicit_models = [(str(m["node"]), str(m["field"])) for m in (req.get("models") or [])]  # type: ignore[attr-defined]
    return spec


def load_template_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def derive_model_requirements(wf: dict) -> list[tuple[str, str, str, str]]:
    """(node, class_type, field, value) for every literal model-file input in a template."""
    out = []
    for node_id, node in wf.items():
        for fld, val in (node.get("inputs") or {}).items():
            if fld in MODEL_FIELDS and isinstance(val, str):
                out.append((str(node_id), node.get("class_type", ""), fld, val))
    return out


def _contract_problems(spec: WorkflowSpec) -> list[str]:
    c = KIND_CONTRACTS[spec.kind]
    problems = [f"kind {spec.kind!r} requires param {r!r}" for r in sorted(c.required - set(spec.params))]
    if len(spec.image_inputs()) < c.min_image_inputs:
        problems.append(f"kind {spec.kind!r} needs {c.min_image_inputs} image input(s) in `inputs`")
    if len(spec.inputs) < c.min_inputs:
        problems.append(f"kind {spec.kind!r} needs at least {c.min_inputs} input in `inputs`")
    return problems


def check_template(spec: WorkflowSpec, wf: dict | None = None) -> list[str]:
    """Problems with spec vs its template (existence, literal targets, outputs)."""
    problems: list[str] = []
    if wf is None:
        if not spec.template.exists():
            return [f"template not found: {spec.template}"]
        try:
            wf = load_template_file(spec.template)
        except (OSError, json.JSONDecodeError) as e:
            return [f"template unreadable: {e}"]
    if not isinstance(wf, dict) or "nodes" in wf:
        return ["template is not ComfyUI API format (flat dict keyed by node id)"]
    for p in spec.params.values():
        for t in p.targets:
            node = wf.get(t.node)
            if node is None:
                problems.append(f"param {p.name!r}: node {t.node!r} not in template")
                continue
            inputs = node.get("inputs") or {}
            if t.field not in inputs:
                problems.append(f"param {p.name!r}: node {t.node} has no input {t.field!r}")
            elif isinstance(inputs[t.field], list):
                problems.append(f"param {p.name!r}: {t.node}.{t.field} is a link, not a literal")
    for n in spec.output_nodes:
        if n not in wf:
            problems.append(f"output node {n!r} not in template")
    for n, f in getattr(spec, "_explicit_models", []):
        if n not in wf or f not in (wf[n].get("inputs") or {}):
            problems.append(f"requires.models: {n}.{f} not in template")
    return problems


def _finalize(spec: WorkflowSpec) -> None:
    """Fill template-derived fields (class types, model files). Raises on problems."""
    problems = check_template(spec)
    problems += _contract_problems(spec)
    if problems:
        raise RegistryError(f"workflow {spec.name!r} ({spec.source}):\n  - " + "\n  - ".join(problems))
    wf = load_template_file(spec.template)
    spec.requires_nodes.update(n.get("class_type", "") for n in wf.values())
    spec.requires_nodes.discard("")
    spec.requires_models = derive_model_requirements(wf)
    for n, f in getattr(spec, "_explicit_models", []):
        val = wf[n]["inputs"][f]
        entry = (n, wf[n].get("class_type", ""), f, str(val))
        if entry not in spec.requires_models:
            spec.requires_models.append(entry)


# --- registry ----------------------------------------------------------------------


class Registry:
    def __init__(self, specs: dict[str, WorkflowSpec], errors: dict[str, str] | None = None):
        self._specs = specs
        self.errors = errors or {}  # user entries that failed to load: name -> problem

    def get(self, name: str) -> WorkflowSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"Unknown workflow {name!r}. Known: {sorted(self._specs)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def list(self, kind: str | None = None) -> list[WorkflowSpec]:
        return [s for s in self._specs.values() if kind is None or s.kind == kind]

    def names(self) -> list[str]:
        return sorted(self._specs)

    def load_template(self, name: str) -> dict:
        return load_template_file(self.get(name).template)

    def apply(self, name: str, wf: dict | None = None, *, strict: bool = True, **values: Any) -> dict:
        """A fresh workflow dict for ``name`` with ``values`` written by node id.

        ``strict=False`` drops params the workflow doesn't declare **if** the
        kind contract lists them as optional (so a leaner user workflow still
        runs); unknown names otherwise raise KeyError.
        """
        spec = self.get(name)
        wf = copy.deepcopy(wf) if wf is not None else self.load_template(name)
        clean: dict[str, Any] = {}
        for pname, value in values.items():
            p = spec.params.get(pname)
            if p is None:
                if not strict and pname in KIND_CONTRACTS[spec.kind].optional:
                    log.warning("workflow %r ignores param %r (not declared)", name, pname)
                    continue
                raise KeyError(f"{name!r} has no param {pname!r}. Known: {sorted(spec.params)}")
            clean[pname] = p.coerce(value)
        hook = KIND_HOOKS.get(spec.kind)
        if hook:
            hook(spec, clean)
        for pname, value in clean.items():
            for t in spec.params[pname].targets:
                wf[t.node]["inputs"][t.field] = value
        return wf

    def validate(
        self, spec: WorkflowSpec, *, object_info: dict | None = None,
        backends: dict[str, Any] | None = None,
    ) -> list[str]:
        """Problems for one spec: template, contract, its backend being configured
        (when ``backends`` is given), and (online) node classes + model files —
        ``object_info`` must be **that backend's**."""
        problems = check_template(spec) + _contract_problems(spec)
        if backends is not None and spec.backend not in backends:
            problems.append(
                f"backend {spec.backend!r} is not configured (add it to {config.BACKENDS_FILE})"
            )
        if object_info is not None and not problems:
            for cls in sorted(spec.requires_nodes):
                if cls not in object_info:
                    problems.append(f"ComfyUI has no node class {cls!r} (custom node not installed?)")
            for node, cls, fld, val in spec.requires_models:
                opts = _combo_options(object_info, cls, fld)
                if opts is not None and val not in opts:
                    problems.append(f"model file {val!r} ({cls}.{fld}, node {node}) is not installed")
            classes: dict[str, str] | None = None
            for p in spec.params.values():
                if p.enum_from and p.enum_from.get("class_type") not in object_info:
                    problems.append(f"param {p.name!r}: enum_from class {p.enum_from.get('class_type')!r} unknown")
                if not p.from_aspect:
                    continue
                # A mistyped size label ("1972x1024") would otherwise surface as a paid API call
                # that fails, or worse succeeds at the wrong size. Check it against the node's enum.
                if classes is None:
                    wf = load_template_file(spec.template)
                    classes = {n: node.get("class_type", "") for n, node in wf.items()}
                for t in p.targets:
                    opts = _combo_options(object_info, classes.get(t.node, ""), t.field)
                    if opts is None:
                        continue
                    for aspect, label in sorted(p.from_aspect.items()):
                        if label not in opts:
                            problems.append(
                                f"param {p.name!r}: from_aspect[{aspect!r}] = {label!r} is not an "
                                f"option of {classes.get(t.node, '')}.{t.field}"
                            )
        return problems

    def requirements(self, name: str) -> dict:
        spec = self.get(name)
        return {
            "nodes": sorted(spec.requires_nodes),
            "models": [{"class_type": c, "field": f, "value": v} for _, c, f, v in spec.requires_models],
        }

    def enum_options(self, name: str, param: str, object_info: dict) -> list | None:
        p = self.get(name).params[param]
        if not p.enum_from:
            return None
        return _combo_options(object_info, p.enum_from.get("class_type"), p.enum_from.get("input"))


def _combo_options(object_info: dict, cls: str | None, fld: str | None) -> list | None:
    """The installed options of a combo input, or None if it is not a combo.

    /object_info encodes combos two ways. Local (V1) nodes inline the list:
    ``[["a.safetensors", "b.safetensors"]]``. Newer schema (V3) nodes, which is every
    ``api_node``, name the type and carry the list in the metadata:
    ``["COMBO", {"options": ["1024x1024", ...]}]``. Both must be read, or an API node's
    enums look empty and nothing can be checked against them."""
    node = object_info.get(cls or "")
    if not node:
        return None
    for section in ("required", "optional"):
        spec = (node.get("input") or {}).get(section, {}).get(fld)
        if not spec:
            continue
        if isinstance(spec[0], list):
            return spec[0]
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            opts = spec[1].get("options")
            if isinstance(opts, list):
                return opts
    return None


def _default_dirs() -> list[tuple[Path, bool]]:
    return [(Path(config.WORKFLOW_DIR), True), (Path(config.USER_WORKFLOW_DIR), False)]


def load_dirs(dirs: Iterable[tuple[Path, bool]]) -> Registry:
    specs: dict[str, WorkflowSpec] = {}
    errors: dict[str, str] = {}
    for d, builtin in dirs:
        if not d.is_dir():
            continue
        for y in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            try:
                with open(y, encoding="utf-8") as f:
                    doc = yaml.safe_load(f)
                spec = parse_spec(doc, source=y, builtin=builtin)
                _finalize(spec)
            except (RegistryError, yaml.YAMLError, OSError) as e:
                if builtin:
                    raise
                log.error("skipping user workflow %s: %s", y, e)
                errors[y.stem] = str(e)
                continue
            if spec.name in specs:
                log.info("workflow %r from %s overrides %s", spec.name, y, specs[spec.name].source)
            specs[spec.name] = spec
    return Registry(specs, errors)


_cache: dict[tuple, tuple[tuple, Registry]] = {}  # dirs -> (signature, registry)


def _signature(pairs: Iterable[tuple[Path, bool]]) -> tuple:
    """Cheap fingerprint of the workflow dirs (names + mtimes + sizes of YAML/JSON
    files) so a long-running server notices a dropped-in workflow without a restart."""
    sig = []
    for d, _ in pairs:
        try:
            entries = sorted(
                (e.name, e.stat().st_mtime_ns, e.stat().st_size)
                for e in os.scandir(d) if e.is_file() and e.name.endswith((".yaml", ".yml", ".json"))
            )
        except OSError:
            entries = []
        sig.append((str(d), tuple(entries)))
    return tuple(sig)


def load_registry(dirs: Sequence[Path] | None = None, *, force: bool = False) -> Registry:
    """Cached registry. ``dirs`` overrides the search path (first dir = built-ins).
    The cache is invalidated when a workflow file appears, changes or disappears."""
    if dirs is None:
        pairs = _default_dirs()
    else:
        pairs = [(Path(d), i == 0) for i, d in enumerate(dirs)]
    key = tuple(str(p) for p, _ in pairs)
    sig = _signature(pairs)
    hit = _cache.get(key)
    if force or hit is None or hit[0] != sig:
        if hit is not None and hit[0] != sig:
            log.info("registry: workflow files changed; reloading")
        _cache[key] = (sig, load_dirs(pairs))
    return _cache[key][1]


def install(reg: Registry) -> None:
    """Make ``reg`` the process-wide default registry (tests, tooling) until the
    default workflow dirs change on disk."""
    pairs = _default_dirs()
    _cache[tuple(str(p) for p, _ in pairs)] = (_signature(pairs), reg)


def reset_cache() -> None:
    _cache.clear()
