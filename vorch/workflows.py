"""Compatibility shim over the declarative registry (``vorch.registry``).

The node-id maps now live in ``workflows/*.yaml``; this module keeps the old
function names (``load_template``, ``apply_params``, ``output_node``,
``_validate_wan_length``) and a read-only ``REGISTRY`` view for callers and
tests written against the Python-dict era.
"""

from __future__ import annotations

from typing import Any

from . import registry as _registry


def load_template(step: str) -> dict:
    """Load a fresh copy of the workflow JSON for ``step``."""
    return _registry.load_registry().load_template(step)


def apply_params(step: str, wf: dict | None = None, **values: Any) -> dict:
    """Return a workflow dict for ``step`` with ``values`` applied by node ID."""
    return _registry.load_registry().apply(step, wf, **values)


def output_node(step: str) -> str:
    """The first node ID whose history outputs hold the result file(s)."""
    nodes = _registry.load_registry().get(step).output_nodes
    if not nodes:
        raise KeyError(f"workflow {step!r} declares no output nodes")
    return nodes[0]


def _validate_wan_length(length: int) -> None:
    """Wan WanImageToVideo accepts only length = 4n+1 (SKILL.md invariant 3)."""
    _registry.CONSTRAINTS["4n+1"](length)


def __getattr__(name: str):
    if name == "REGISTRY":  # legacy view: step -> {template, output_node, params: {name: (node, field)}}
        reg = _registry.load_registry()
        return {
            spec.name: {
                "template": spec.template.name,
                "output_node": spec.output_nodes[0] if spec.output_nodes else None,
                "params": {p: (ps.targets[0].node, ps.targets[0].field) for p, ps in spec.params.items()},
            }
            for spec in reg.list()
        }
    raise AttributeError(name)
