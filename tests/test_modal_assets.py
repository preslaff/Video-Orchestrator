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


EXPECTED = {  # name -> (kind, backend)
    "animate_wan22_fp16": ("animate", "modal-h100"),
    "animate_wan22_fp8": ("animate", "modal-l40s"),
    "animate_ltx25": ("animate", "modal-h100"),
    "keyframe_flux2_4b": ("keyframe", "modal-l40s"),
    "keyframe_flux2_9b": ("keyframe", "modal-l40s"),
    "animate_wan_animate2": ("postprocess", "modal-l40s"),
    "faceswap_video_remote": ("postprocess", "modal-l40s"),
}
LOCAL_UPLOAD = {"faceswap_video_remote"}  # models come from modal/upload_local_models.sh, not a download set


def test_cloud_workflows_load_with_core_nodes_and_declared_backends():
    reg = registry.load_dirs([(MODAL / "workflows", False)])
    assert reg.errors == {}
    assert set(reg.names()) == set(EXPECTED)
    builtin = registry.load_registry()
    for spec in reg.list():
        kind, backend = EXPECTED[spec.name]
        assert (spec.kind, spec.backend) == (kind, backend)
        assert not (spec.requires_nodes & {"UnetLoaderGGUF", "CLIPLoaderGGUF"})
        wf = json.loads(spec.template.read_text())
        assert "nodes" not in wf                                   # API format, not a UI export
        if spec.name == "faceswap_video_remote":
            assert spec.template.read_bytes() == (ROOT / "workflows" / "reactor_video_faceswap.json").read_bytes()
        if spec.kind in ("animate", "keyframe") and "ltx" not in spec.name:
            # same params as the built-in of that kind (a profile can swap them freely)
            assert set(spec.params) >= set(builtin.get("animate" if spec.kind == "animate" else "keyframe").params)
    fp16 = reg.get("animate_wan22_fp16")
    assert fp16.params["width"].max == 1280
    assert "wan2.2_i2v_high_noise_14B_fp16.safetensors" in {v for _, _, _, v in fp16.requires_models}
    ltx = reg.get("animate_ltx25")
    assert ltx.params["length"].constraint == "8n+1" and ltx.params["width"].constraint == "multiple_of_32"
    assert {"LTXVConditioning", "LTXVLatentUpsampler", "LTXVAudioVAEDecode"} <= ltx.requires_nodes
    a2 = reg.get("animate_wan_animate2")
    assert a2.accepts("source_image") and a2.video_inputs() == ["input_video"]
    assert "WanAnimate2ToVideo" in a2.requires_nodes
    assert "license:non-commercial" in reg.get("keyframe_flux2_9b").tags


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

    consts = [n for n in tree.body if names(n) and all(x.isupper() for x in names(n))]  # WAN22, KLEIN4B, ..., SETS
    exec(compile(ast.Module(body=consts, type_ignores=[]), "sets", "exec"), ns)  # noqa: S102
    available = {Path(p).name for items in ns["SETS"].values() for _, p, _ in items}
    reg = registry.load_dirs([(MODAL / "workflows", False)])
    uploader = (MODAL / "upload_local_models.sh").read_text()
    for spec in reg.list():
        for _, _, _, val in spec.requires_models:
            if spec.name in LOCAL_UPLOAD:
                assert val in uploader, f"{spec.name} needs {val} but upload_local_models.sh does not copy it"
            else:
                assert val in available, f"{spec.name} needs {val} but no download set provides it"


def test_cloud_profiles_import_shape_and_validate(tmp_path, monkeypatch):
    from vorch import config, profiles

    # make the cloud workflows visible so the profiles validate against their real rules
    for f in (MODAL / "workflows").iterdir():
        (config.USER_WORKFLOW_DIR / f.name).write_bytes(f.read_bytes())
    registry.reset_cache()
    reg = registry.load_registry()
    for f in sorted((MODAL / "profiles").glob("*.yaml")):
        doc = yaml.safe_load(f.read_text())
        assert doc["name"] and doc["base"] in ("cinematic_16x9", "social_9x16", "social_1x1", "stills_16x9")
        p = profiles.clone(profiles.builtin_profiles()[doc["base"]], doc["name"], overrides=doc["overrides"])
        assert set(p.workflows_used()) <= set(reg.names()), (f.name, p.workflows_used())
        rules = registry.load_registry().get(p.video.workflow)
        assert p.video.workflow.startswith(("animate_",))
        if "ltx" in p.video.workflow:
            assert p.video.length % 8 == 1 and p.video.width % 32 == 0 and p.video.fps == 24
        else:
            assert p.video.length % 4 == 1
        assert rules.backend.startswith("modal-")
