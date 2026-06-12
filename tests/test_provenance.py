"""GPU-free checks for reproducibility provenance — seed materialization + manifests."""

from __future__ import annotations

import json

from vorch import config, provenance, steps
from vorch.clients import comfy


def test_resolve_seed_passes_fixed_through():
    assert config.resolve_seed(42) == 42


def test_resolve_seed_materializes_random_for_sentinel():
    s = config.resolve_seed(config.SEED_RANDOM)
    assert s != config.SEED_RANDOM and config._SEED_MIN <= s <= config._SEED_MAX


def test_manifest_path_appends_json():
    assert provenance.manifest_path("/out/Flux2-Klein_00029_.png").name == (
        "Flux2-Klein_00029_.png.json"
    )


def test_write_manifest_records_prompt_seed_and_output(tmp_path):
    media = tmp_path / "Flux2-Klein_00029_.png"
    media.write_bytes(b"img")
    dest = provenance.write_manifest(
        media, "keyframe", {"positive": "oil painting, kneading bread", "seed": 7, "cfg": 5}
    )
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["step"] == "keyframe"
    assert data["template"] == "image_flux2_klein_text_to_image.json"  # from REGISTRY
    assert data["output"] == "Flux2-Klein_00029_.png"
    assert data["params"]["positive"] == "oil painting, kneading bread"
    assert data["params"]["seed"] == 7


def test_run_step_writes_sidecar(monkeypatch, tmp_path):
    """run_step writes a manifest beside each produced output."""
    out = tmp_path / "Wan22_00010_.mp4"
    out.write_bytes(b"vid")

    monkeypatch.setattr(steps.workflows, "apply_params", lambda step, **p: {"wf": True})
    monkeypatch.setattr(comfy, "run", lambda wf, timeout=None: [{"filename": out.name}])
    monkeypatch.setattr(comfy, "output_path", lambda d: out)

    paths = steps.run_step("animate", positive="slow pan", seed=99, length=49)
    assert paths == [out]
    manifest = json.loads(provenance.manifest_path(out).read_text(encoding="utf-8"))
    assert manifest["params"]["seed"] == 99 and manifest["params"]["positive"] == "slow pan"
    assert manifest["step"] == "animate"
