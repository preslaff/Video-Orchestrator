"""Environment check: every external touchpoint is monkeypatched; the DB check is real."""

from __future__ import annotations

import shutil

import pytest
import requests

from vorch import config, envcheck, profiles, registry, supervisor, vram
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
            files = req.setdefault(fld, [[], {}])[0]
            if val not in files:  # built-ins share model files; a real /object_info lists each once
                files.append(val)
        for p in spec.params.values():
            if p.enum_from:
                info.setdefault(p.enum_from["class_type"], {"input": {"required": {}}})
    return info


@pytest.fixture
def healthy(monkeypatch, tmp_path):
    """All services up; returns the mutable object_info so tests can break things."""
    info = _object_info_for_builtins()

    def fake_get(url, timeout=None):
        if url.endswith("/v1/models"):
            return _Resp({"data": [{"id": "gemma-4"}]})
        if url.endswith("/health"):
            # The narration server is not running, which for an on-demand service is correct.
            raise requests.ConnectionError("refused")
        raise AssertionError(url)

    stats = {"system": {"comfyui_version": "0.34.0"}, "devices": [{"name": "RTX", "vram_total": 16 * 2**30}]}
    monkeypatch.setattr(envcheck.requests, "get", fake_get)
    monkeypatch.setattr(comfy, "system_stats", lambda backend=None, **kw: stats)
    monkeypatch.setattr(comfy, "wait_ready", lambda backend=None, **kw: stats)
    monkeypatch.setattr(comfy, "object_info", lambda refresh=False, backend=None: info)
    monkeypatch.setattr(vram, "snapshot", lambda: {"used": 500, "free": 15000, "total": 16000})
    # Per-process attribution and the service rows must not read this machine's real state.
    monkeypatch.setattr(vram, "holders", lambda: [])
    empty_proc = tmp_path / "proc"
    empty_proc.mkdir()
    monkeypatch.setattr(supervisor, "PROC", empty_proc)
    monkeypatch.setattr(config, "RUN_DIR", tmp_path / "run")
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
    assert by["comfy:local"].data["version"] == "0.34.0" and by["comfy:local"].data["probed"] is True
    assert by["backends"].detail.startswith("local only")
    assert [c.name for c in rep.checks if c.name.startswith("comfy:")] == ["comfy:local"]
    assert by["workflow:animate"].data["used"] is True and by["workflow:kv_edit"].data["used"] is False
    assert by["workflow:animate"].data["backend"] == "local"
    assert by["profile"].detail.startswith("cinematic_16x9")
    d = rep.to_dict()
    assert d["ok"] and d["checks"][0]["name"] == "database"
    assert (config.EXPORT_ROOT).is_dir()  # created by the check


def test_comfy_down_is_fail_and_workflows_go_offline(healthy, store, monkeypatch):
    def down(backend=None, **kw):
        raise comfy.ComfyError("local: GET /system_stats failed (refused)")

    monkeypatch.setattr(comfy, "system_stats", down)
    rep = envcheck.run_checks(profile=profiles.default_profile(), store=store, force=True)
    by = _by_name(rep)
    assert not rep.ok and by["comfy:local"].status == "fail"
    assert by["workflow:keyframe"].status == "ok" and "offline" in by["workflow:keyframe"].detail
    with pytest.raises(envcheck.EnvironmentNotReady) as ei:
        envcheck.gate(profiles.default_profile(), store=store)
    assert "comfy:local" in str(ei.value)


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
        raise requests.ConnectionError("refused")

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

    def counting(backend, **kw):
        calls["n"] += 1
        return real(backend, **kw)

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


def test_probe_png_is_truecolor_and_decodable(tmp_path):
    import struct
    import subprocess

    data = envcheck.probe_png()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h, depth, color = struct.unpack(">IIBB", data[16:26])
    assert (w, h, depth, color) == (64, 64, 8, 2)  # 8-bit RGB, not palette (llama-server rejects pal8)
    if shutil.which("ffprobe"):
        f = tmp_path / "p.png"
        f.write_bytes(data)
        out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,pix_fmt",
                                       "-of", "csv=p=0", str(f)]).decode().strip()
        assert out == "64,64,rgb24"


# --- backends (docs/modal-backend.md §3) --------------------------------------------------


def _remote_backend(tmp_path, monkeypatch, name="box", **extra):
    import yaml

    from vorch import backends

    doc = {"backends": {name: {"url": "http://box:8188", "cold_start_grace": 0, "gpu": "H100", **extra}}}
    (tmp_path / "backends.yaml").write_text(yaml.safe_dump(doc))
    monkeypatch.setattr(config, "BACKENDS_FILE", tmp_path / "backends.yaml")
    backends.reset_cache()


def _remote_animate(tmp_path, monkeypatch, backend="box"):
    import yaml

    doc = yaml.safe_load((config.WORKFLOW_DIR / "animate.yaml").read_text())
    doc.update(name="animate_remote", template=str(config.WORKFLOW_DIR / doc["template"]), backend=backend)
    config.USER_WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    (config.USER_WORKFLOW_DIR / "animate_remote.yaml").write_text(yaml.safe_dump(doc))
    registry.reset_cache()
    return profiles.apply_overrides(profiles.default_profile(), {"video": {"workflow": "animate_remote"}})


def test_remote_backend_not_probed_unless_used_or_deep(healthy, store, tmp_path, monkeypatch):
    _remote_backend(tmp_path, monkeypatch)
    probed: list[str] = []
    stats = {"system": {"comfyui_version": "0.35.0"}, "devices": [{"name": "H100", "vram_total": 80 * 2**30}]}
    monkeypatch.setattr(comfy, "wait_ready", lambda backend=None, **kw: probed.append(backend.name) or stats)
    infos: list[str] = []
    monkeypatch.setattr(comfy, "object_info", lambda refresh=False, backend=None: infos.append(backend.name) or healthy)

    # local profile: the remote row exists but is not contacted, and no object_info is fetched from it
    rep = envcheck.run_checks(profile=profiles.default_profile(), store=store, force=True)
    by = _by_name(rep)
    assert rep.ok and by["backends"].detail == "local + box"
    assert by["comfy:box"].status == "ok" and by["comfy:box"].data["probed"] is False and "not probed" in by["comfy:box"].detail
    assert probed == [] and infos == ["local"]

    # a profile using a remote workflow probes it (the job follows right away) and validates against its object_info
    prof = _remote_animate(tmp_path, monkeypatch)
    probed.clear(); infos.clear()
    rep = envcheck.run_checks(profile=prof, store=store, force=True)
    by = _by_name(rep)
    assert probed == ["box"] and sorted(infos) == ["box", "local"]
    assert by["comfy:box"].detail == "ComfyUI 0.35.0 on H100" and by["comfy:box"].data["probed"] is True
    assert by["workflow:animate_remote"].status == "ok" and by["workflow:animate_remote"].data == {"used": True, "backend": "box"}
    assert "@box" in by["workflow:animate_remote"].detail and "offline" not in by["workflow:animate_remote"].detail
    assert by["workflow:animate"].data["used"] is False

    # deep probes everything
    probed.clear()
    envcheck.run_checks(profile=profiles.default_profile(), store=store, deep=True, force=True)
    assert probed == ["box"]


def test_remote_workflow_is_checked_against_its_own_object_info(healthy, store, tmp_path, monkeypatch):
    _remote_backend(tmp_path, monkeypatch)
    prof = _remote_animate(tmp_path, monkeypatch)
    monkeypatch.setattr(comfy, "wait_ready", lambda backend=None, **kw: {"system": {}, "devices": []})
    monkeypatch.setattr(comfy, "object_info", lambda refresh=False, backend=None: healthy if backend.is_local else {})
    rep = envcheck.run_checks(profile=prof, store=store, force=True)
    by = _by_name(rep)
    assert by["workflow:animate"].status == "ok"                # local ComfyUI has everything
    assert by["workflow:animate_remote"].status == "fail" and not rep.ok  # the remote box has no nodes
    assert "UnetLoaderGGUF" in by["workflow:animate_remote"].detail


def test_remote_backend_down_fails_and_unconfigured_backend_is_flagged(healthy, store, tmp_path, monkeypatch):
    _remote_backend(tmp_path, monkeypatch)
    prof = _remote_animate(tmp_path, monkeypatch)

    def down(backend=None, **kw):
        raise comfy.ComfyError("box: GET /system_stats failed (ConnectionError)")

    monkeypatch.setattr(comfy, "wait_ready", down)
    rep = envcheck.run_checks(profile=prof, store=store, force=True)
    by = _by_name(rep)
    assert by["comfy:box"].status == "fail" and "remote ComfyUI unreachable" in by["comfy:box"].detail
    assert by["workflow:animate_remote"].status == "ok" and "offline" in by["workflow:animate_remote"].detail

    _remote_animate(tmp_path, monkeypatch, backend="ghost")
    rep = envcheck.run_checks(profile=prof, store=store, force=True)
    c = _by_name(rep)["workflow:animate_remote"]
    assert c.status == "fail" and "backend 'ghost' is not configured" in c.detail


def test_comfy_dirs_relaxed_when_only_remote_workflows_are_used(healthy, store, tmp_path, monkeypatch):
    _remote_backend(tmp_path, monkeypatch)
    prof = _remote_animate(tmp_path, monkeypatch)
    monkeypatch.setattr(comfy, "wait_ready", lambda backend=None, **kw: {"system": {}, "devices": []})
    monkeypatch.setattr(config, "COMFY_INPUT", tmp_path / "nope_in")
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path / "nope_out")
    # the default profile still uses local keyframe/faceswap workflows -> hard fail
    rep = envcheck.run_checks(profile=prof, store=store, force=True)
    assert _by_name(rep)["comfy_dirs"].status == "fail"
    # a profile whose every workflow is remote: output dir created, input dir only a warning
    remote_only = profiles.apply_overrides(prof, {"keyframe": {"workflow": "animate_remote"}, "face_swap": {"enabled": False}})
    rep = envcheck.run_checks(profile=remote_only, store=store, force=True)
    c = _by_name(rep)["comfy_dirs"]
    assert c.status == "warn" and "input dir missing" in c.detail and (tmp_path / "nope_out").is_dir()
    # no profile at all: the local install is assumed
    rep = envcheck.run_checks(store=store, force=True)
    assert _by_name(rep)["comfy_dirs"].status == "fail"


def test_backends_file_error_is_a_fail_and_local_still_checked(healthy, store, tmp_path, monkeypatch):
    from vorch import backends

    monkeypatch.delenv("VORCH_TEST_MISSING_KEY", raising=False)
    (tmp_path / "backends.yaml").write_text(
        "backends:\n  m:\n    url: https://x\n    auth: modal-proxy\n    key_env: VORCH_TEST_MISSING_KEY\n    secret_env: VORCH_TEST_MISSING_KEY\n"
    )
    monkeypatch.setattr(config, "BACKENDS_FILE", tmp_path / "backends.yaml")
    backends.reset_cache()
    rep = envcheck.run_checks(store=store, force=True)
    by = _by_name(rep)
    assert by["backends"].status == "fail" and "VORCH_TEST_MISSING_KEY" in by["backends"].detail
    assert by["comfy:local"].status == "ok" and "comfy:m" not in by and not rep.ok


# --- the gpu row judges against a real number when it can ------------------------------


def _low_vram(monkeypatch, free=2001, holders=None):
    monkeypatch.setattr(vram, "snapshot", lambda: {"used": 16303 - free, "free": free, "total": 16303})
    monkeypatch.setattr(vram, "holders", lambda: holders or [])
    envcheck.clear_cache()


def _holder(name, mb, evictable):
    return vram.Holder(pid=1000 + mb, used_mb=mb, service=name, cmdline=name, cwd=None,
                       evictable=evictable, started_at=None)


def test_gpu_stays_advisory_without_a_profile(healthy, store, monkeypatch):
    """No profile means no concrete requirement, so a fail would be a guess."""
    _low_vram(monkeypatch, holders=[_holder("tts", 11268, True)])
    rep = envcheck.run_checks(store=store, force=True)
    row = _by_name(rep)["gpu"]
    assert row.status == "warn" and rep.ok
    assert "held by tts" in row.detail


def test_gpu_warns_when_a_render_needs_freeing_first(healthy, store, monkeypatch):
    """Reclaimable is not a problem: it happens on its own before the render."""
    _low_vram(monkeypatch, holders=[_holder("tts", 11268, True), _holder("comfy", 222, True)])
    rep = envcheck.run_checks(profile=profiles.default_profile(), store=store, force=True)
    row = _by_name(rep)["gpu"]
    assert row.status == "warn" and rep.ok, "a submit must not be blocked by something we can fix"
    assert "automatically before the render" in row.detail
    assert row.data["admission"]["ok"] is True


def test_gpu_fails_when_the_deficit_is_real_and_blocks_submit(healthy, store, monkeypatch):
    """The case the old warn-only row let through: a job would be admitted and then OOM."""
    _low_vram(monkeypatch, free=500, holders=[_holder("mystery", 13000, False)])
    rep = envcheck.run_checks(profile=profiles.default_profile(), store=store, force=True)
    row = _by_name(rep)["gpu"]
    assert row.status == "fail" and not rep.ok
    assert "short" in row.detail
    with pytest.raises(envcheck.EnvironmentNotReady):
        envcheck.gate(profiles.default_profile(), store=store)


def test_gpu_is_ok_when_it_simply_fits(healthy, store, monkeypatch):
    _low_vram(monkeypatch, free=13000, holders=[_holder("gemma", 1838, False)])
    row = _by_name(envcheck.run_checks(profile=profiles.default_profile(), store=store,
                                      force=True))["gpu"]
    assert row.status == "ok"
