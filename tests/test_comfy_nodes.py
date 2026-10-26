"""vorch's own ComfyUI node (``comfy_nodes/vorch_matte``).

The node lives in ComfyUI's world, not vorch's: it needs torch, and vorch's runtime is six
packages by design (``pyproject.toml``). So the helper tests are marked and **skip here**. To run
them you need ComfyUI's interpreter *and* a pytest in it, which it does not have by default:

    ~/comfy/venv-next/bin/python -m pip install pytest     # not installed by default
    ~/comfy/venv-next/bin/python -m pytest -q tests/test_comfy_nodes.py

Be honest about what that means: as the machine stands, those two skip everywhere. Their
assertions were verified by running the node directly in the comfy venv against a synthetic moving
disc (dilate 9x9 -> 121 px, feather centre > 0.99 with a ramped edge, and a propagated matte that
is 0.000 at a box corner outside the shape). They are kept because they are the regression net the
moment anyone does add pytest there.

The two tests that **always run** need no imports at all, and they guard the two failures that are
otherwise invisible: a renamed ``NODE_CLASS_MAPPINGS`` (the node just never appears in
``/object_info``) and a checkpoint-to-config typo (a hydra error at render time).
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

NODES = Path(__file__).resolve().parent.parent / "comfy_nodes" / "vorch_matte"
HAVE_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch lives in ComfyUI's venv, not this one")


def _exported(path: Path) -> set[str]:
    """Names a module binds at top level, read without importing it."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def test_the_package_exports_what_comfyui_loads():
    """ComfyUI imports a custom-node package and reads ``NODE_CLASS_MAPPINGS`` off it.

    Static, so it holds even where torch is absent: this is the check that catches a renamed
    mapping or a module that stopped re-exporting it, which otherwise shows up only as a node
    silently missing from ``/object_info``.
    """
    assert "NODE_CLASS_MAPPINGS" in _exported(NODES / "__init__.py")
    assert "NODE_CLASS_MAPPINGS" in _exported(NODES / "sam2_box_matte.py")


def test_every_checkpoint_name_maps_to_a_config_that_sam2_ships():
    """The config table is the one place that must track upstream's filenames.

    Impact Pack's equivalent table knows only the SAM 2.0 names, which is why this node has its
    own; a typo here surfaces as a hydra error at render time instead.
    """
    src = (NODES / "sam2_box_matte.py").read_text()
    table = ast.literal_eval(src.split("CONFIGS = ", 1)[1].split("\n}\n", 1)[0] + "\n}")
    assert len(table) == 8, "four checkpoints x two SAM2 generations"
    for ckpt, cfg in table.items():
        assert ckpt.endswith(".pt")
        gen = "sam2.1" if ckpt.startswith("sam2.1") else "sam2"
        assert cfg.startswith(f"configs/{gen}/{gen}_hiera_"), f"{ckpt} -> {cfg}"
        assert cfg.endswith(".yaml")

    if importlib.util.find_spec("sam2") is not None:      # the real check, in the comfy venv
        import sam2
        root = Path(sam2.__file__).parent
        missing = [c for c in table.values() if not (root / c).is_file()]
        assert not missing, f"configs not shipped by the installed sam2: {missing}"


@needs_torch
def test_the_helpers_shape_a_mask_the_way_the_blend_wants():
    """``dilate`` covers a seam; ``feather`` softens the edge without touching the middle."""
    import sys
    import torch
    sys.path.insert(0, str(NODES.parent))
    from vorch_matte.sam2_box_matte import _dilate, _feather

    m = torch.zeros(1, 41, 41)
    m[0, 16:25, 16:25] = 1.0                                       # a 9x9 blob

    assert float(_dilate(m, 1).sum()) == 121.0                     # 9x9 -> 11x11
    assert float(_dilate(m, 0).sum()) == 81.0                      # 0 is a no-op

    f = _feather(m, 3)
    assert float(f[0, 20, 20]) > 0.99, "the middle of the mask must stay opaque"
    assert 0.0 < float(f[0, 20, 15]) < 1.0, "the edge must ramp, not step"
    assert float(f.max()) <= 1.0 and float(f.min()) >= 0.0
    assert float(_feather(m, 0).sum()) == 81.0                     # 0 is a no-op
