"""The Modal deployment assets stay consistent with the registry and the app code
(no Modal account needed: files are parsed and compiled, never deployed)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from vorch import registry

ROOT = Path(__file__).resolve().parent.parent
MODAL = ROOT / "modal"


def test_modal_scripts_compile_and_pin_the_local_comfy_version():
    src = (MODAL / "vorch_comfy.py").read_text()
    ast.parse(src)
    ast.parse((MODAL / "download_models.py").read_text())
    servers = [l for l in src.splitlines() if l.startswith("@modal.web_server(")]
    assert len(servers) == 2 and all("requires_proxy_auth=True" in l for l in servers)
    assert "min_containers=" not in src  # cost: never keep-warm in the committed file
    assert 'COMFY_VERSION = "v0.34.0"' in src


def test_cloud_workflows_load_and_use_core_nodes_only():
    reg = registry.load_dirs([(MODAL / "workflows", False)])
    assert reg.errors == {}
    assert reg.names() == ["animate_wan22_fp16", "animate_wan22_fp8"]
    for spec in reg.list():
        assert spec.kind == "animate" and spec.backend.startswith("modal-")
        assert set(spec.params) >= set(registry.load_registry().get("animate").params)
        assert not (spec.requires_nodes & {"UnetLoaderGGUF", "CLIPLoaderGGUF"})
        wf = json.loads(spec.template.read_text())
        assert "nodes" not in wf and wf["1"]["class_type"] == "UNETLoader"
    fp16 = reg.get("animate_wan22_fp16")
    assert fp16.backend == "modal-h100" and fp16.params["width"].max == 1280
    files = {v for _, _, _, v in fp16.requires_models}
    assert "wan2.2_i2v_high_noise_14B_fp16.safetensors" in files


def test_downloader_sets_cover_the_cloud_workflow_models():
    ns: dict = {}
    src = (MODAL / "download_models.py").read_text()
    # evaluate only the constants (no modal import needed)
    tree = ast.parse(src)
    def names(n):
        if isinstance(n, ast.Assign):
            return [t.id for t in n.targets if isinstance(t, ast.Name)]
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            return [n.target.id]
        return []

    consts = [n for n in tree.body if set(names(n)) & {"WAN22", "SETS"}]
    exec(compile(ast.Module(body=consts, type_ignores=[]), "sets", "exec"), ns)  # noqa: S102
    available = {Path(p).name for items in ns["SETS"].values() for _, p, _ in items}
    reg = registry.load_dirs([(MODAL / "workflows", False)])
    for spec in reg.list():
        for _, _, _, val in spec.requires_models:
            assert val in available, f"{spec.name} needs {val} but no download set provides it"


def test_cloud_profiles_import_shape_and_reference_cloud_workflows():
    reg = registry.load_dirs([(MODAL / "workflows", False)])
    for f in sorted((MODAL / "profiles").glob("*.yaml")):
        doc = yaml.safe_load(f.read_text())
        assert doc["name"] and doc["base"] in ("cinematic_16x9", "social_9x16", "social_1x1", "stills_16x9")
        assert doc["overrides"]["video"]["workflow"] in reg
        assert doc["overrides"]["video"]["length"] % 4 == 1
