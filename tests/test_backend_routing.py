"""Per-workflow backend routing (docs/modal-backend.md §3): the registry
``backend`` field, run_step dispatch + download for a remote backend, uploads
going to the *consuming* workflow's backend, provenance recording the backend,
and the VRAM guard skipping remote backends."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

from vorch import backends, config, pipeline, provenance, registry, steps, vram
from vorch.backends import ComfyBackend
from vorch.clients import comfy
from vorch.genparams import DEFAULT, PostStep
from vorch.steps import postprocess

REMOTE = ComfyBackend(name="box", url="http://box:8188", remote=True, gpu="H100")


def _user_copy(name: str, builtin: str, **extra) -> Path:
    """Register a user workflow = the built-in's YAML with an absolute template + extras."""
    doc = yaml.safe_load((config.WORKFLOW_DIR / f"{builtin}.yaml").read_text())
    doc["name"] = name
    doc["template"] = str(config.WORKFLOW_DIR / doc["template"])
    doc.update(extra)
    config.USER_WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    out = config.USER_WORKFLOW_DIR / f"{name}.yaml"
    out.write_text(yaml.safe_dump(doc))
    return out


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A configured remote backend 'box' + remote copies of animate/faceswap."""
    (tmp_path / "backends.yaml").write_text(
        "backends:\n  box:\n    url: http://box:8188\n    cold_start_grace: 0\n    gpu: H100\n    cost_per_hour: 3.95\n"
    )
    monkeypatch.setattr(config, "BACKENDS_FILE", tmp_path / "backends.yaml")
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path / "out")
    _user_copy("animate_remote", "animate", backend="box")
    _user_copy("faceswap_remote", "faceswap", backend="box")
    backends.reset_cache()
    registry.reset_cache()
    return backends.get("box")


class _Resp:
    def __init__(self, status=200, data=None, content=b""):
        self.status_code, self._data, self.content, self.text = status, data, content, ""

    def json(self):
        return self._data

    def iter_content(self, chunk_size=1):
        yield self.content

    def close(self):
        pass


class _Comfy:
    """A whole fake ComfyUI behind comfy._session: records host + path per call."""

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []
        self.headers: dict = {}

    def request(self, method, url, **kw):
        u = urlsplit(url)
        self.calls.append((method, u.netloc, u.path))
        if u.path == "/system_stats":
            return _Resp(200, {"system": {"comfyui_version": "0.34.0"}})
        if u.path == "/prompt":
            return _Resp(200, {"prompt_id": "p1"})
        if u.path.startswith("/history/"):
            return _Resp(200, {"p1": {"status": {"status_str": "success"}, "outputs": {
                "13": {"gifs": [{"filename": "s00_clip_00001_.mp4", "subfolder": "vorch/job1", "type": "output"}]},
                "11": {"images": [{"filename": "preview_00001_.png", "subfolder": "", "type": "output"}]},
            }}})
        if u.path == "/view":
            return _Resp(200, content=b"VIDEO")
        if u.path == "/upload/image":
            return _Resp(200, {"name": kw["files"]["image"][0]})
        raise AssertionError(f"unexpected {method} {url}")


# --- registry field ---------------------------------------------------------------------


def test_backend_field_default_parse_and_to_dict(tmp_path):
    base = {"name": "x", "kind": "custom", "template": str(config.WORKFLOW_DIR / "wan22_i2v_gguf.json")}
    assert registry.parse_spec(base).backend == "local"
    spec = registry.parse_spec({**base, "backend": "modal-h100"})
    assert spec.backend == "modal-h100" and spec.to_dict()["backend"] == "modal-h100"
    assert registry.load_registry().get("animate").to_dict()["backend"] == "local"
    with pytest.raises(registry.RegistryError, match="invalid backend name"):
        registry.parse_spec({**base, "backend": "Not Valid"})


def test_validate_flags_an_unconfigured_backend(box):
    reg = registry.load_registry()
    spec = replace(reg.get("animate_remote"), backend="nope")
    assert not [p for p in reg.validate(spec) if "backend" in p]  # offline: not checked
    problems = reg.validate(spec, backends=backends.load())
    assert len(problems) == 1 and "backend 'nope' is not configured" in problems[0]
    assert reg.validate(reg.get("animate_remote"), backends=backends.load()) == []
    assert backends.for_workflow("animate_remote") is box


# --- run_step on a remote backend ---------------------------------------------------------


def test_run_step_on_remote_backend_downloads_into_local_layout(box, monkeypatch, tmp_path):
    fake = _Comfy()
    monkeypatch.setattr(comfy, "_session", lambda b: fake)
    monkeypatch.setattr(comfy, "_sleep", lambda s: None)
    paths = steps.run_step("animate_remote", start_image="f.png", positive="pan", length=49, seed=3)
    assert paths == [tmp_path / "out" / "vorch/job1" / "s00_clip_00001_.box.mp4"]
    assert paths[0].read_bytes() == b"VIDEO"
    assert {host for _, host, _ in fake.calls} == {"box:8188"}
    assert [p for _, _, p in fake.calls] == ["/system_stats", "/prompt", "/history/p1", "/view"]
    manifest = json.loads(provenance.manifest_path(paths[0]).read_text())
    assert manifest["backend"] == "box" and manifest["gpu"] == "H100" and manifest["params"]["seed"] == 3


def test_local_run_step_records_local_backend(monkeypatch, tmp_path):
    out = tmp_path / "k.png"
    out.write_bytes(b"i")
    seen = {}

    def fake_run(wf, timeout=None, *, backend=None):
        seen["backend"] = backend
        return [{"filename": out.name, "subfolder": "", "node_id": "9"}]

    monkeypatch.setattr(comfy, "run", fake_run)
    monkeypatch.setattr(comfy, "output_path", lambda d, backend=None: out)
    steps.run_step("keyframe", positive="p", seed=1, width=512, height=512)
    assert seen["backend"].name == "local" and seen["backend"].is_local
    manifest = json.loads(provenance.manifest_path(out).read_text())
    assert manifest["backend"] == "local" and manifest["gpu"] is None


# --- uploads follow the consuming workflow ----------------------------------------------


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    uploads: list[tuple[str, str]] = []
    reclaims: list[str | None] = []

    def upload(path, *, backend=None):
        uploads.append((Path(path).name, (backend or backends.local()).name))
        return f"staged:{Path(path).name}"

    def run(wf, timeout=None, *, backend=None):
        return [{"filename": "o.png", "subfolder": "", "node_id": "4"}]

    monkeypatch.setattr(comfy, "upload_image", upload)
    monkeypatch.setattr(comfy, "run", run)
    monkeypatch.setattr(comfy, "output_path", lambda d, backend=None: tmp_path / f"{(backend or backends.local()).name}_{d['filename']}")
    monkeypatch.setattr(vram, "reclaim", lambda *a, backend=None, **k: reclaims.append(backend.name if backend else None))
    monkeypatch.setattr(vram, "log_state", lambda *a, **k: None)
    return uploads, reclaims


def test_swap_and_animate_upload_to_their_workflows_backend(box, fakes, tmp_path):
    uploads, reclaims = fakes
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"i")
    params = replace(DEFAULT, faceswap_workflow="faceswap_remote", animate_workflow="animate_remote")
    assert pipeline.upload_master(frame, params) == "staged:frame.png"
    pipeline.swap_master_onto(frame, "staged:master", params=params)
    pipeline.animate_frame(frame, "pan", seed=1, params=params)
    assert uploads == [("frame.png", "box")] * 3
    assert reclaims == ["box"]
    # the default (local) profile is unchanged
    uploads.clear(); reclaims.clear()
    pipeline.swap_master_onto(frame, "m", params=DEFAULT)
    pipeline.animate_frame(frame, "pan", seed=1, params=DEFAULT)
    assert uploads == [("frame.png", "local")] * 2 and reclaims == ["local"]


def test_stage_references_go_to_edit_backend_and_cache_per_backend(box, fakes, tmp_path):
    uploads, _ = fakes
    _user_copy("edit_remote", "identity_edit", backend="box")
    registry.reset_cache()
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"i")
    cache: dict = {}
    remote = replace(DEFAULT, edit_workflow="edit_remote")
    assert pipeline.stage_references([ref], remote, fit=False, cache=cache) == ["staged:ref.png"]
    pipeline.stage_references([ref], remote, fit=False, cache=cache)      # cached
    pipeline.stage_references([ref], DEFAULT, fit=False, cache=cache)     # other backend: uploaded again
    assert uploads == [("ref.png", "box"), ("ref.png", "local")]
    assert set(cache) == {("box", str(ref)), ("local", str(ref))}


def test_run_sequence_uploads_master_to_faceswap_backend(box, fakes, tmp_path, monkeypatch):
    uploads, _ = fakes
    master = tmp_path / "master.jpg"
    master.write_bytes(b"i")
    monkeypatch.setattr(pipeline, "keyframe", lambda prompt, **kw: tmp_path / "kf.png")
    monkeypatch.setattr(pipeline, "faceswap", lambda a, b, **kw: tmp_path / "sw.png")
    monkeypatch.setattr(pipeline, "animate", lambda a, b, **kw: tmp_path / "clip.mp4")
    params = replace(DEFAULT, faceswap_workflow="faceswap_remote")
    pipeline.run_sequence([pipeline.Scene("k", "a")], master, params=params)
    assert uploads == [("master.jpg", "box"), ("kf.png", "box"), ("sw.png", "local")]


def test_post_chain_routes_each_step(box, fakes, tmp_path):
    uploads, reclaims = fakes
    tpl = tmp_path / "post.json"
    tpl.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
                               "4": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "p"}}}))
    for name, backend in (("up_remote", "box"), ("up_local", "local")):
        (config.USER_WORKFLOW_DIR / f"{name}.yaml").write_text(yaml.safe_dump({
            "kind": "postprocess", "template": str(tpl), "backend": backend,
            "inputs": {"input_image": "image"}, "outputs": {"type": "image", "nodes": ["4"]},
            "params": {"input_image": {"node": "1", "field": "image", "type": "str", "required": True}},
        }))
    registry.reset_cache()
    still = tmp_path / "still.png"
    still.write_bytes(b"i")
    outs = postprocess.run_chain(still, [PostStep("up_remote"), PostStep("up_local")])
    assert [n for n, _ in uploads] == ["still.png", "box_o.png"]
    assert [b for _, b in uploads] == ["box", "local"] and reclaims == ["box", "local"]
    assert [p.name for p in outs] == ["box_o.png", "local_o.png"]


# --- VRAM guard -------------------------------------------------------------------------


def test_reclaim_is_a_noop_for_remote_backends(monkeypatch):
    freed: list = []
    monkeypatch.setattr(vram.comfy, "free", lambda *a, **k: freed.append(k.get("backend")))
    monkeypatch.setattr(vram, "gpu_free_mb", lambda: 3000)
    assert vram.reclaim(backend=REMOTE, settle=0) is None
    assert freed == []
    local = backends.local()
    assert vram.reclaim(backend=local, settle=0) == 3000
    assert freed == [local]
    vram.reclaim(settle=0)
    assert freed == [local, None]
