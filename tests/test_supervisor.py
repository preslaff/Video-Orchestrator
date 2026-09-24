"""Supervisor, read-only half: is it running, is it ours, is it the RIGHT instance.

A fake /proc tree and a fake manifest, so the whole thing is checkable with no GPU, no network and
no process ever started. The cases are the ones that actually bit us: a narration server adopted
and never stopped, an orphan left by a SIGKILLed vorch, and a server quietly reading the wrong
cloned voice.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
import yaml

from vorch import config, services, supervisor, vram

TICKS = float(os.sysconf("SC_CLK_TCK"))
BOOT_OFFSET = 100_000.0


def dead_pid() -> int:
    """A pid that is certainly gone — a child we started and reaped."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def make_proc(root: Path, procs: dict[int, dict]) -> Path:
    """Build /proc/<pid>/{cmdline,cwd,stat,environ} plus /proc/uptime."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "uptime").write_text(f"{BOOT_OFFSET} 0.0\n", encoding="utf-8")
    for pid, p in procs.items():
        d = root / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "cmdline").write_bytes(p["cmdline"].replace(" ", "\0").encode() + b"\0")
        cwd = Path(p["cwd"])
        cwd.mkdir(parents=True, exist_ok=True)
        (d / "cwd").symlink_to(cwd)
        if (env := p.get("environ")) is not None:
            (d / "environ").write_bytes(
                b"".join(f"{k}={v}\0".encode() for k, v in env.items()))
        start = (BOOT_OFFSET - p.get("uptime", 60.0)) * TICKS
        (d / "stat").write_text(f"{pid} (fake) " + " ".join(["S"] + ["0"] * 18 + [str(int(start))]),
                                encoding="utf-8")
    return root


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A manifest with one env-verified service (tts) and one argv-verified one (gemma)."""
    tts_cwd, gemma_cwd = tmp_path / "omni", tmp_path / "llama"
    manifest = tmp_path / "services.yaml"
    doc = {"services": {
        "gemma": {"manage": "full", "argv": ["llama-server", "--port", "8080"], "cwd": str(gemma_cwd),
                  "match": {"cmdline_contains": ["llama-server"], "cwd": str(gemma_cwd)},
                  "health": {"probe": "http", "url": "http://localhost:8080/v1/models"},
                  "vram": {"class": "resident", "holds_mb": 1838},
                  "adopt": {"verify": "argv"}},
        "tts": {"manage": "full", "argv": ["bash", "serve.sh"], "cwd": str(tts_cwd),
                "match": {"cmdline_contains": ["omnivoice.cli.serve"], "cwd": str(tts_cwd)},
                "env": {"OMNIVOICE_BG_REF_AUDIO": "Viktoria/11s.wav"},
                "health": {"probe": "http", "url": "http://localhost:8002/health"},
                "vram": {"class": "exclusive", "needs_mb": 11500, "holds_mb": 11268},
                "lifetime": "warm", "idle_timeout": 600, "evict_adopted": True,
                "adopt": {"verify": "env", "on_unverified": "warn"}},
    }}
    manifest.write_text(yaml.safe_dump(doc), encoding="utf-8")
    # Replace the manifest rather than override it: load() normally merges the shipped built-ins
    # over a user file, and here we want exactly these two services and nothing else.
    monkeypatch.setattr(services, "load", lambda path=None, *, force=False: services.parse(
        yaml.safe_load(manifest.read_text(encoding="utf-8")), source=str(manifest)))
    monkeypatch.setattr(config, "SERVICES_FILE", manifest)
    monkeypatch.setattr(config, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    services.reset_cache()

    proc = make_proc(tmp_path / "proc", {
        251592: {"cmdline": "python -m omnivoice.cli.serve --model x", "cwd": tts_cwd,
                 "uptime": 41_000.0, "environ": {"OMNIVOICE_BG_REF_AUDIO": "Viktoria/11s.wav"}},
        8536: {"cmdline": "llama-server -m model.gguf --port 8080", "cwd": gemma_cwd,
               "uptime": 82_000.0, "environ": {}},
    })
    monkeypatch.setattr(supervisor, "PROC", proc)
    monkeypatch.setattr(vram, "PROC", proc)
    monkeypatch.setattr(vram, "holders", lambda: [
        vram.Holder(pid=251592, used_mb=11268, service="tts", cmdline="omnivoice.cli.serve",
                    cwd=str(tts_cwd), evictable=True, started_at=time.time() - 41_000),
    ])
    # Everything answers unless a test says otherwise.
    monkeypatch.setattr(supervisor.requests, "get",
                        lambda url, timeout=None: _Ok({"status": "ok"}))
    yield tmp_path
    services.reset_cache()


class _Ok:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def write_record(service: str, **kw) -> None:
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    kw.setdefault("name", service)
    kw.setdefault("started_at", time.time())
    (config.RUN_DIR / f"{service}.json").write_text(json.dumps(kw), encoding="utf-8")


# --- finding processes ----------------------------------------------------------------


def test_find_pids_uses_the_manifest_matcher(world):
    assert supervisor.find_pids(services.get("tts")) == [251592]
    assert supervisor.find_pids(services.get("gemma")) == [8536]


def test_a_matching_cmdline_in_the_wrong_cwd_is_not_a_match(world, tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "PROC", make_proc(tmp_path / "proc2", {
        111: {"cmdline": "python -m omnivoice.cli.serve", "cwd": tmp_path / "somewhere-else"},
    }))
    assert supervisor.find_pids(services.get("tts")) == []


# --- is it ours ------------------------------------------------------------------------


def test_a_server_we_started_is_ours(world):
    write_record("tts", pid=251592, owner_pid=os.getpid())
    s = supervisor.resolve("tts")
    assert s.running and s.identity == "ours" and not s.adopted and not s.orphan
    assert s.status == "ok"


def test_a_hand_started_server_is_adopted_not_a_fault(world):
    """Launching a service in a terminal is a supported way to work, not a degraded one."""
    s = supervisor.resolve("tts")
    assert s.running and s.adopted and s.record is None
    assert s.status == "ok", "an adopted, verified server is healthy"
    assert "started outside vorch" in s.detail


def test_an_orphan_is_a_failure_naming_what_it_holds(world):
    """SIGKILL cannot be caught, so the pidfile is the only record of what the crash left behind."""
    write_record("tts", pid=251592, owner_pid=dead_pid())
    s = supervisor.resolve("tts")
    assert s.orphan and s.adopted and s.status == "fail"
    assert s.vram_mb == 11268
    assert "ORPHAN" in s.detail


def test_a_stale_pidfile_is_discarded_rather_than_believed(world):
    """A pid may have been reused; trusting the file blindly would name the wrong process."""
    write_record("tts", pid=dead_pid(), owner_pid=os.getpid())
    s = supervisor.resolve("tts")
    assert s.record is None
    assert s.pid == 251592, "it falls through to the /proc scan"


def test_an_unreadable_pidfile_is_ignored(world):
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    (config.RUN_DIR / "tts.json").write_text("{not json", encoding="utf-8")
    assert supervisor.read_record("tts") is None
    assert supervisor.resolve("tts").running


# --- is it the RIGHT instance ----------------------------------------------------------


def test_the_cloned_voice_is_verified_from_proc_environ(world):
    """/health never reports the voice; /proc/<pid>/environ does. That is the whole check."""
    s = supervisor.resolve("tts")
    assert s.identity == "verified"
    assert "Viktoria/11s.wav" in s.identity_detail


def test_a_different_voice_is_a_named_failure(world, tmp_path, monkeypatch):
    """The standing bug: a server on serve_bg.sh's default reads every line in another voice."""
    monkeypatch.setattr(supervisor, "PROC", make_proc(tmp_path / "proc3", {
        251592: {"cmdline": "python -m omnivoice.cli.serve", "cwd": tmp_path / "omni",
                 "environ": {"OMNIVOICE_BG_REF_AUDIO": "examples/reklama_home_max.wav"}},
    }))
    s = supervisor.resolve("tts")
    assert s.identity == "mismatch" and s.status == "fail"
    # Both voices named, so the reader knows what to fix.
    assert "reklama_home_max.wav" in s.identity_detail and "Viktoria/11s.wav" in s.identity_detail


def test_a_missing_identity_variable_is_a_mismatch(world, tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "PROC", make_proc(tmp_path / "proc4", {
        251592: {"cmdline": "python -m omnivoice.cli.serve", "cwd": tmp_path / "omni",
                 "environ": {}},
    }))
    s = supervisor.resolve("tts")
    assert s.identity == "mismatch" and "is not set" in s.identity_detail


def test_unreadable_environ_only_warns(world, tmp_path, monkeypatch):
    """A hand-started server must never be blocked merely because we cannot inspect it."""
    monkeypatch.setattr(supervisor, "PROC", make_proc(tmp_path / "proc5", {
        251592: {"cmdline": "python -m omnivoice.cli.serve", "cwd": tmp_path / "omni"},
    }))  # no environ file at all
    s = supervisor.resolve("tts")
    assert s.identity == "unverified" and s.status == "warn"
    assert "hardened kernel" in s.identity_detail


def test_on_unverified_refuse_makes_it_a_failure(world, tmp_path, monkeypatch):
    manifest = yaml.safe_load((config.SERVICES_FILE).read_text(encoding="utf-8"))
    manifest["services"]["tts"]["adopt"]["on_unverified"] = "refuse"
    config.SERVICES_FILE.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    services.reset_cache()
    monkeypatch.setattr(supervisor, "PROC", make_proc(tmp_path / "proc6", {
        251592: {"cmdline": "python -m omnivoice.cli.serve", "cwd": tmp_path / "omni"},
    }))
    assert supervisor.resolve("tts").status == "fail"


def test_argv_verification_notices_a_different_command(world, tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "PROC", make_proc(tmp_path / "proc7", {
        8536: {"cmdline": "llama-server -m model.gguf --port 9999", "cwd": tmp_path / "llama"},
    }))
    s = supervisor.resolve("gemma")
    assert s.identity == "mismatch" and "8080" in s.identity_detail


def test_verify_port_claims_nothing_it_cannot_show(world, tmp_path, monkeypatch):
    manifest = yaml.safe_load((config.SERVICES_FILE).read_text(encoding="utf-8"))
    manifest["services"]["tts"]["adopt"]["verify"] = "port"
    config.SERVICES_FILE.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    services.reset_cache()
    s = supervisor.resolve("tts")
    assert s.identity == "n/a" and s.status == "ok"
    assert "port alone" in s.identity_detail


# --- not running ----------------------------------------------------------------------


def test_an_on_demand_service_that_is_down_is_correct_not_broken(world, tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "PROC", make_proc(tmp_path / "empty", {}))
    monkeypatch.setattr(supervisor.requests, "get",
                        lambda url, timeout=None: (_ for _ in ()).throw(requests.ConnectionError("no")))
    s = supervisor.resolve("tts")
    assert not s.running and s.status == "down"
    assert "which is correct" in s.detail


def test_a_health_probe_alone_can_show_something_is_listening(world, tmp_path, monkeypatch):
    """Last resort: no pidfile, no /proc match, but the port answers."""
    monkeypatch.setattr(supervisor, "PROC", make_proc(tmp_path / "empty2", {}))
    s = supervisor.resolve("tts")
    assert s.running and s.healthy is True and s.pid is None and s.adopted


def test_a_service_that_does_not_answer_is_a_failure(world, monkeypatch):
    monkeypatch.setattr(supervisor.requests, "get",
                        lambda url, timeout=None: (_ for _ in ()).throw(requests.ConnectionError("refused")))
    s = supervisor.resolve("tts")
    assert s.running, "the process is there"
    assert s.healthy is False and s.status == "fail"
    assert "not answering" in s.detail


# --- reporting -------------------------------------------------------------------------


def test_states_come_back_in_dependency_order(world):
    assert [s.name for s in supervisor.states()] == ["gemma", "tts"]


def test_orphans_lists_only_what_an_earlier_vorch_left(world):
    assert supervisor.orphans() == []
    write_record("tts", pid=251592, owner_pid=dead_pid())
    assert [s.name for s in supervisor.orphans()] == ["tts"]


def test_summary_carries_the_gpu_table_and_the_problems(world):
    write_record("tts", pid=251592, owner_pid=dead_pid())
    out = supervisor.summary()
    assert out["orphans"] == ["tts"]
    assert any("tts" in p for p in out["problems"])
    assert str(config.SERVICES_FILE) == out["services_file"]


def test_tail_explains_an_absent_log_rather_than_failing(world):
    text = supervisor.tail("tts")
    assert "no log at" in text and "vorch has not started tts" in text


def test_tail_returns_the_last_lines(world):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOG_DIR / "tts.log").write_text("\n".join(f"line {i}" for i in range(500)),
                                            encoding="utf-8")
    out = supervisor.tail("tts", 3).splitlines()
    assert out == ["line 497", "line 498", "line 499"]


def test_idle_seconds_come_from_the_record(world):
    write_record("tts", pid=251592, owner_pid=os.getpid(),
                 last_used_at=time.time() - 300)
    s = supervisor.resolve("tts")
    assert 290 < s.idle_sec < 310


def test_resolve_refuses_an_unknown_service(world):
    with pytest.raises(services.ServiceError, match="unknown service"):
        supervisor.resolve("ghost")
