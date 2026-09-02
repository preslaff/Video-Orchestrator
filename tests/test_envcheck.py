"""Environment check: every external touchpoint is monkeypatched; the DB check is real."""

from __future__ import annotations

import shutil

import pytest
import requests

from vorch import config, envcheck, profiles, registry, vram
from vorch.clients import comfy


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._p


def _object_info_for_builtins() -> dict:
    """An /object_info that satisfies every built-in workflow (classes + model files)."""
    info: dict = {}
    for spec in registry.load_registry().list():
        for cls in spec.requires_nodes:
            info.setdefault(cls, {"input": {"required": {}}})
        for _, cls, fld, val in spec.requires_models:
            req = info.setdefault(cls, {"input": {"required": {}}})["input"]["required"]
            req.setdefault(fld, [[], {}])[0].append(val)
        for p in spec.params.values():
            if p.enum_from:
                info.setdefault(p.enum_from["class_type"], {"input": {"required": {}}})
    return info


@pytest.fixture
def healthy(monkeypatch, tmp_path):
    """All services up; returns the mutable object_info so tests can break things."""
    info = _object_info_for_builtins()

    def fake_get(url, timeout=None):
        if url.endswith("/system_stats"):
            return _Resp({"system": {"comfyui_version": "0.34.0"}, "devices": [{"name": "RTX", "vram_total": 16 * 2**30}]})
        if url.endswith("/v1/models"):
            return _Resp({"data": [{"id": "gemma-4"}]})
        raise AssertionError(url)

    monkeypatch.setattr(envcheck.requests, "get", fake_get)
    monkeypatch.setattr(comfy, "object_info", lambda refresh=False: info)
    monkeypatch.setattr(vram, "snapshot", lambda: {"used": 500, "free": 15000, "total": 16000})
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    for name in ("COMFY_INPUT", "COMFY_OUTPUT"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(config, name, d)
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(config, "EXPORT_ROOT", tmp_path / "export")
    envcheck.clear_cache()
    yield info
    envcheck.clear_cache()


def _by_name(report):
    return {c.name: c for c in report.checks}


def test_all_ok_when_services_up(healthy, store):
    prof = profiles.default_profile()
    rep = envcheck.run_checks(profile=prof, store=store, force=True)
    by = _by_name(rep)
    assert rep.ok and rep.status == "ok", [c for c in rep.checks if c.status != "ok"]
    assert by["database"].data["version"] >= 2
    assert by["comfy"].data["version"] == "0.34.0"
    assert by["workflow:animate"].data["used"] is True and by["workflow:kv_edit"].data["used"] is False
    assert by["profile"].detail.startswith("cinematic_16x9")
    d = rep.to_dict()
    assert d["ok"] and d["checks"][0]["name"] == "database"
    assert (config.EXPORT_ROOT).is_dir()  # created by the check


def test_comfy_down_is_fail_and_workflows_go_offline(healthy, store, monkeypatch):
    def down(url, timeout=None):
        if url.endswith("/system_stats"):
            raise requests.ConnectionError("refused")
        return _Resp({"data": []})

    monkeypatch.setattr(envcheck.requests, "get", down)
    rep = envcheck.run_checks(profile=profiles.default_profile(), store=store, force=True)
    by = _by_name(rep)
    assert not rep.ok and by["comfy"].status == "fail"
    assert by["workflow:keyframe"].status == "ok" and "offline" in by["workflow:keyframe"].detail
    with pytest.raises(envcheck.EnvironmentNotReady) as ei:
        envcheck.gate(profiles.default_profile(), store=store)
    assert "comfy" in str(ei.value)


def test_missing_model_file_fails_only_for_profile_workflows(healthy, store):
    healthy["UnetLoaderGGUF"]["input"]["required"]["unet_name"][0].remove("Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf")
    video = envcheck.run_checks(profile=profiles.default_profile(), store=store, force=True)
    assert _by_name(video)["workflow:animate"].status == "fail" and not video.ok
    stills = envcheck.run_checks(profile=profiles.builtin_profiles()["stills_16x9"], store=store, force=True)
    assert _by_name(stills)["workflow:animate"].status == "warn" and stills.ok


def test_missing_custom_node_class_fails(healthy, store):
    del healthy["ReActorFaceSwap"]
    rep = envcheck.run_checks(profile=profiles.default_profile(), store=store, force=True)
    c = _by_name(rep)["workflow:faceswap"]
    assert c.status == "fail" and "ReActorFaceSwap" in c.detail


def test_ffmpeg_missing_gpu_low_and_gemma_down(healthy, store, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(vram, "snapshot", lambda: {"used": 15000, "free": 800, "total": 16000})

    def gemma_down(url, timeout=None):
        if url.endswith("/v1/models"):
            raise requests.ConnectionError("refused")
        return _Resp({"system": {"comfyui_version": "0.34.0"}, "devices": []})

    monkeypatch.setattr(envcheck.requests, "get", gemma_down)
    rep = envcheck.run_checks(store=store, force=True)
    by = _by_name(rep)
    assert by["ffmpeg"].status == "warn" and by["ffprobe"].status == "warn" and by["gpu"].status == "warn"
    assert by["gemma"].status == "fail" and by["profile"].status == "ok"
    assert "gemma_vision" not in by  # not deep


def test_deep_vision_probe_warns_without_mmproj(healthy, store, monkeypatch):
    from vorch.clients import gemma

    monkeypatch.setattr(gemma, "complete_vision", lambda *a, **k: "NO, I cannot see any image")
    rep = envcheck.run_checks(store=store, deep=True, force=True)
    c = _by_name(rep)["gemma_vision"]
    assert c.status == "warn" and "--mmproj" in c.detail and rep.ok
    monkeypatch.setattr(gemma, "complete_vision", lambda *a, **k: "YES")
    assert _by_name(envcheck.run_checks(store=store, deep=True, force=True))["gemma_vision"].status == "ok"


def test_profile_problems(healthy, store):
    bad = profiles.apply_overrides(profiles.default_profile(), {"video": {"workflow": "nope"}})
    rep = envcheck.run_checks(profile=bad, store=store, force=True)
    assert _by_name(rep)["profile"].status == "fail"
    heavy = profiles.apply_overrides(profiles.default_profile(), {"video": {"length": 121}})
    rep = envcheck.run_checks(profile=heavy, store=store, force=True)
    assert _by_name(rep)["profile"].status == "warn" and rep.ok


def test_cache_ttl_and_force(healthy, store, monkeypatch):
    calls = {"n": 0}
    real = envcheck._check_comfy

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(envcheck, "_check_comfy", counting)
    envcheck.run_checks(store=store, force=True)
    envcheck.run_checks(store=store)
    assert calls["n"] == 1
    envcheck.run_checks(store=store, force=True)
    assert calls["n"] == 2
    envcheck.run_checks(store=store, ttl=0)
    assert calls["n"] == 3


def test_database_check_without_store(healthy, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql:///definitely_not_a_db_xyz")
    rep = envcheck.run_checks(force=True)
    assert _by_name(rep)["database"].status == "fail" and not rep.ok
