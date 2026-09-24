"""GPU-free checks for vram.py — nvidia-smi parsing and the reclaim guard.

subprocess and comfy.free are monkeypatched so there is no real GPU/sleep.
"""

from __future__ import annotations

import subprocess

import pytest

from vorch import vram


def _fake_smi(mapping):
    """Return a fake check_output that maps a query field to a CSV line."""
    def check_output(cmd, **kw):
        query = next(a.split("=", 1)[1] for a in cmd if a.startswith("--query-gpu="))
        if query not in mapping:
            raise subprocess.CalledProcessError(1, cmd)
        return (mapping[query] + "\n").encode()
    return check_output


def test_gpu_free_mb_parses(monkeypatch):
    monkeypatch.setattr(vram.subprocess, "check_output", _fake_smi({"memory.free": "13744"}))
    assert vram.gpu_free_mb() == 13744


def test_gpu_free_mb_none_without_nvidia_smi(monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(vram.subprocess, "check_output", boom)
    assert vram.gpu_free_mb() is None
    assert vram.snapshot() is None


def test_snapshot_parses_triplet(monkeypatch):
    monkeypatch.setattr(
        vram.subprocess, "check_output",
        _fake_smi({"memory.used,memory.free,memory.total": "2091, 13744, 16303"}),
    )
    assert vram.snapshot() == {"used": 2091, "free": 13744, "total": 16303}


def test_reclaim_unconditional_frees(monkeypatch):
    freed = []
    monkeypatch.setattr(vram.comfy, "free", lambda *a, **k: freed.append(True))
    monkeypatch.setattr(vram, "gpu_free_mb", lambda: 5000)
    vram.reclaim(settle=0)
    assert freed == [True]


def test_reclaim_skips_when_above_guard(monkeypatch):
    freed = []
    monkeypatch.setattr(vram.comfy, "free", lambda *a, **k: freed.append(True))
    monkeypatch.setattr(vram, "gpu_free_mb", lambda: 14000)
    out = vram.reclaim(min_mb=10000, settle=0)
    assert freed == [] and out == 14000  # enough headroom, no reclaim


def test_reclaim_frees_when_below_guard(monkeypatch):
    freed = []
    monkeypatch.setattr(vram.comfy, "free", lambda *a, **k: freed.append(True))
    monkeypatch.setattr(vram, "gpu_free_mb", lambda: 3000)
    vram.reclaim(min_mb=10000, settle=0)
    assert freed == [True]


def test_log_state_handles_missing_smi(monkeypatch):
    monkeypatch.setattr(vram, "snapshot", lambda: None)
    vram.log_state("test")  # must not raise


# --- per-process attribution and admission -------------------------------------------
#
# A fake /proc tree plus a fake nvidia-smi, so the whole arbiter is checkable with no GPU.

import os
from pathlib import Path

import yaml

from vorch import services


def _fake_gpu(mapping, apps=""):
    """check_output covering both --query-gpu= and --query-compute-apps=."""
    def check_output(cmd, **kw):
        if any(a.startswith("--query-compute-apps=") for a in cmd):
            return (apps + "\n").encode()
        query = next(a.split("=", 1)[1] for a in cmd if a.startswith("--query-gpu="))
        if query not in mapping:
            raise subprocess.CalledProcessError(1, cmd)
        return (mapping[query] + "\n").encode()
    return check_output


def _fake_proc(root: Path, procs: dict[int, tuple[str, Path, float]], boot_offset=100_000.0):
    """Build /proc/<pid>/{cmdline,cwd,stat} and /proc/uptime for the given pids.

    ``procs`` maps pid -> (cmdline, cwd, uptime_seconds).
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "uptime").write_text(f"{boot_offset} 0.0\n", encoding="utf-8")
    ticks = float(os.sysconf("SC_CLK_TCK"))
    for pid, (cmdline, cwd, uptime) in procs.items():
        d = root / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode() + b"\0")
        cwd.mkdir(parents=True, exist_ok=True)
        (d / "cwd").symlink_to(cwd)
        # field 22 (starttime) is the 20th entry after the parenthesised comm
        start_ticks = (boot_offset - uptime) * ticks
        after_comm = ["S"] + ["0"] * 18 + [str(int(start_ticks))]
        (d / "stat").write_text(f"{pid} (fake) " + " ".join(after_comm) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Today's actual mess, in miniature: a narration server squatting on the card."""
    comfy_cwd, tts_cwd, gemma_cwd = tmp_path / "comfy", tmp_path / "omni", tmp_path / "llama"
    manifest = tmp_path / "services.yaml"
    manifest.write_text(yaml.safe_dump({"services": {
        "gemma": {"manage": "full", "argv": ["llama-server"], "cwd": str(gemma_cwd),
                  "match": {"cmdline_contains": ["llama-server"], "cwd": str(gemma_cwd)},
                  "vram": {"class": "resident", "holds_mb": 1838}, "lifetime": "long"},
        "comfy": {"manage": "full", "argv": ["python", "main.py"], "cwd": str(comfy_cwd),
                  "match": {"cmdline_contains": ["main.py"], "cwd": str(comfy_cwd)},
                  "vram": {"class": "elastic", "needs_mb": 12000, "holds_mb": 222,
                           "free_endpoint": "/free"}, "lifetime": "long"},
        "tts": {"manage": "full", "argv": ["bash", "serve.sh"], "cwd": str(tts_cwd),
                "match": {"cmdline_contains": ["omnivoice.cli.serve"], "cwd": str(tts_cwd)},
                "vram": {"class": "exclusive", "needs_mb": 11500, "holds_mb": 11268},
                "lifetime": "warm", "idle_timeout": 600, "evict_adopted": True},
    }}), encoding="utf-8")
    monkeypatch.setenv("VORCH_SERVICES_FILE", str(manifest))
    monkeypatch.setattr(services.config, "SERVICES_FILE", manifest)
    services.reset_cache()

    monkeypatch.setattr(vram, "PROC", _fake_proc(tmp_path / "proc", {
        251592: ("python -m omnivoice.cli.serve --model x", tts_cwd, 41_000.0),   # up ~11h
        8536: ("./build/bin/llama-server -m model.gguf", gemma_cwd, 82_000.0),
        409909: ("python main.py --enable-manager", comfy_cwd, 21_000.0),
        999: ("/usr/bin/some-other-cuda-thing", tmp_path / "elsewhere", 60.0),    # not in the manifest
    }))
    monkeypatch.setattr(vram.subprocess, "check_output", _fake_gpu(
        {"memory.used,memory.free,memory.total": "13833, 2001, 16303",
         "memory.free": "2001"},
        apps="251592, 11268\n8536, 1960\n409909, 222\n999, 100",
    ))
    yield
    services.reset_cache()


def test_holders_name_processes_from_the_manifest(world):
    hs = {h.service: h for h in vram.holders()}
    assert hs["tts"].used_mb == 11268 and hs["gemma"].used_mb == 1960
    # Largest first, so the culprit is the first thing anyone reads.
    assert vram.holders()[0].service == "tts"
    # Uptime comes from /proc/<pid>/stat: the point is spotting a server left running for hours.
    assert 11 * 3600 < hs["tts"].uptime_sec < 12 * 3600
    assert hs["tts"].uptime_str.startswith("11h")


def test_a_process_the_manifest_does_not_know_is_never_evictable(world):
    unknown = next(h for h in vram.holders() if h.service is None)
    assert unknown.pid == 999 and unknown.evictable is False
    # It still shows up, with its command line, so a person can decide.
    assert "some-other-cuda-thing" in unknown.cmdline
    assert unknown.label == "some-other-cuda-thing"


def test_evictable_follows_the_manifest_not_the_size(world):
    hs = {h.service: h for h in vram.holders()}
    assert hs["comfy"].evictable, "elastic: frees without stopping"
    assert hs["tts"].evictable, "warm + evict_adopted: may be stopped"
    assert not hs["gemma"].evictable, "resident: the evaluator, a fixed cost on purpose"


def test_attribute_accounts_for_driver_overhead(world):
    info = vram.attribute()
    assert info["unattributed_mb"] == 13833 - (11268 + 1960 + 222 + 100)


def test_admit_when_it_already_fits(world):
    adm = vram.admit(1500)
    assert adm.ok and adm.clear and not adm.evict


def test_admit_frees_cheapest_first(world):
    """ComfyUI's /free costs seconds; stopping the narration model costs a 90 s reload."""
    adm = vram.admit(12000)
    assert adm.ok and adm.evict == ("comfy", "tts")


def test_admit_refuses_and_says_what_is_missing(world):
    adm = vram.admit(30000)
    assert not adm.ok
    assert adm.deficit_mb == 30000 - (2001 + 11268 + 222)
    # Gemma is reported as a fixed cost, never as something to go and kill.
    assert "fixed cost and stays" in adm.reason
    assert "vorch down gemma" not in adm.reason


def test_refusal_names_a_squatter_and_the_command_to_stop_it(world):
    """The narration server is excluded (we are about to use it), so it cannot free itself —
    but its memory must not silently vanish from the arithmetic."""
    adm = vram.admit(11500, exclude=["tts"])
    assert not adm.ok
    assert "tts" not in adm.evict
    assert "11268 MB" in adm.reason and "vorch down tts" in adm.reason


def test_exclude_keeps_a_service_out_of_the_eviction_plan(world):
    assert vram.admit(12000, exclude=["tts"]).evict == ("comfy",)


def test_admit_does_not_arbitrate_without_nvidia_smi(monkeypatch):
    """Same stance reclaim already takes: nothing to measure, so nothing to refuse."""
    def boom(*a, **k):
        raise OSError("no nvidia-smi")
    monkeypatch.setattr(vram.subprocess, "check_output", boom)
    adm = vram.admit(99999)
    assert adm.ok and adm.free_mb is None and "not arbitrating" in adm.reason


def test_admit_decides_but_never_frees_anything(world, monkeypatch):
    """vram measures and decides; only the supervisor acts. Guard the boundary explicitly.

    admit() returning an eviction plan must not *be* the eviction — otherwise a diagnostic command
    like `vorch vram --by-process --need N`, or the envcheck row, would stop a running service as a
    side effect of being asked a question.
    """
    def must_not_run(*a, **k):
        raise AssertionError("admit() freed VRAM; that is the supervisor's job")

    monkeypatch.setattr(vram.comfy, "free", must_not_run)
    adm = vram.admit(12000)
    assert adm.ok and adm.evict == ("comfy", "tts")     # a plan, not an action
    # Still exactly as it was before the call.
    assert {h.service for h in vram.holders()} == {"tts", "gemma", "comfy", None}


def test_gpu_check_names_the_holder_instead_of_blaming_comfy(world):
    from vorch import envcheck

    row = envcheck._check_gpu()
    assert row.status == "warn"
    assert "held by tts" in row.detail and "11268 MB" in row.detail
    assert "ComfyUI may still hold models" not in row.detail
    assert row.data["holders"][0]["service"] == "tts"
