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


# --- starting and stopping real processes ----------------------------------------------
#
# These use a genuine child (a python that sleeps) and the real /proc, because the point is the
# lifecycle: does the pidfile land, does the log capture output, does a stop actually stop it.

import uuid

import yaml as _yaml


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A manifest with one startable service: a python that sleeps, tagged so /proc can find it."""
    marker = f"--vorch-test-{uuid.uuid4().hex[:12]}"
    work = tmp_path / "work"
    work.mkdir()
    manifest = tmp_path / "services.yaml"
    doc = {"services": {
        "fake": {
            "manage": "full", "cwd": str(work),
            "argv": [sys.executable, "-c",
                     "import sys, time; print('hello from the child', flush=True); time.sleep(60)",
                     marker],
            "match": {"cmdline_contains": [marker], "cwd": str(work)},
            "health": {"probe": "http", "url": "http://localhost:59999/health", "timeout": 1},
            "ready_timeout": 20,
            "vram": {"class": "exclusive", "needs_mb": 500, "holds_mb": 400},
            "lifetime": "warm", "idle_timeout": 300, "evict_adopted": True,
            "adopt": {"verify": "port"},
        },
    }}
    manifest.write_text(_yaml.safe_dump(doc), encoding="utf-8")
    monkeypatch.setattr(services, "load", lambda path=None, *, force=False: services.parse(
        _yaml.safe_load(manifest.read_text(encoding="utf-8")), source=str(manifest)))
    monkeypatch.setattr(config, "SERVICES_FILE", manifest)
    monkeypatch.setattr(config, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "PROC", Path("/proc"))
    monkeypatch.setattr(vram, "holders", lambda: [])
    monkeypatch.setattr(vram, "admit", lambda need, **kw: vram.Admission(
        ok=True, need_mb=need, free_mb=99999, reason="test: plenty"))
    healthy = {"value": True}

    def fake_get(url, timeout=None):
        # Honest probe: it only answers once a matching process actually exists, so start() cannot
        # mistake an empty machine for a running service and adopt a phantom.
        if not healthy["value"]:
            raise requests.ConnectionError("refused")
        if not supervisor.find_pids(services.get("fake")):
            raise requests.ConnectionError("nothing listening yet")
        return _Ok({"status": "ok"})

    monkeypatch.setattr(supervisor.requests, "get", fake_get)
    services.reset_cache()
    yield {"marker": marker, "healthy": healthy, "manifest": manifest, "doc": doc, "work": work}
    # Never leave a child behind, whatever the test did.
    for pid in supervisor.find_pids(services.get("fake")):
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    supervisor._OURS.clear()
    services.reset_cache()


def test_start_launches_waits_and_records(live):
    state = supervisor.start("fake")
    assert state.running and state.pid and state.identity == "ours"
    rec = supervisor.read_record("fake")
    assert rec is not None and rec.pid == state.pid and rec.owner_pid == os.getpid()
    assert rec.owner_alive
    # Its own session, so a Ctrl-C in the terminal that ran `vorch up` cannot scatter into it.
    assert os.getpgid(state.pid) == state.pid
    supervisor.stop("fake")
    assert not supervisor.resolve("fake", do_probe=False).running
    assert supervisor.read_record("fake") is None


def test_the_log_captures_the_child_and_says_what_was_started(live):
    supervisor.start("fake")
    text = supervisor.tail("fake")
    # The header is what makes a past run answerable weeks later.
    assert "=== vorch start fake" in text
    assert live["marker"] in text and str(live["work"]) in text
    # And the child's own stdout — which used to go to DEVNULL and be lost entirely.
    assert "hello from the child" in text
    supervisor.stop("fake")


def test_a_child_that_exits_raises_with_the_tail_of_its_log(live, monkeypatch):
    doc = live["doc"]
    doc["services"]["fake"]["argv"] = [sys.executable, "-c",
                                       "import sys; print('boom: no model here'); sys.exit(3)",
                                       live["marker"]]
    live["manifest"].write_text(_yaml.safe_dump(doc), encoding="utf-8")
    services.reset_cache()
    live["healthy"]["value"] = False
    with pytest.raises(supervisor.SupervisorError) as e:
        supervisor.start("fake")
    assert "exited while starting" in str(e.value) and "boom: no model here" in str(e.value)
    assert supervisor.read_record("fake") is None, "a failed start leaves no pidfile"


def test_readiness_timeout_says_where_to_look(live):
    live["healthy"]["value"] = False
    with pytest.raises(supervisor.SupervisorError) as e:
        supervisor.start("fake", timeout=2)
    assert "did not answer within" in str(e.value) and "still running as pid" in str(e.value)
    supervisor.stop("fake", force=True)


def test_start_adopts_rather_than_restarting(live):
    first = supervisor.start("fake")
    again = supervisor.start("fake")
    assert again.pid == first.pid, "a running service is never restarted"
    supervisor.stop("fake")


def test_start_refuses_a_service_that_is_not_ours_to_start(live):
    doc = live["doc"]
    doc["services"]["fake"]["manage"] = "adopt"
    live["manifest"].write_text(_yaml.safe_dump(doc), encoding="utf-8")
    services.reset_cache()
    with pytest.raises(supervisor.SupervisorError, match="not vorch's to start"):
        supervisor.start("fake")


def test_stop_will_not_touch_what_vorch_did_not_start(live):
    """Launching a service by hand is supported, so vorch must not quietly kill it."""
    supervisor.start("fake")
    supervisor.clear_record("fake")          # as if a person had started it
    with pytest.raises(supervisor.SupervisorError, match="will not stop what it did not start"):
        supervisor.stop("fake")
    supervisor.stop("fake", force=True)      # explicit is fine
    assert not supervisor.resolve("fake", do_probe=False).running


def test_stop_refuses_a_pid_it_cannot_identify(live, tmp_path):
    """The rail that matters: vorch never signals a process the manifest cannot name."""
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    (config.RUN_DIR / "fake.json").write_text(json.dumps(
        {"name": "fake", "pid": os.getpid(), "owner_pid": 1, "started_at": time.time()}),
        encoding="utf-8")
    # The pidfile points at pytest itself, which does not match the manifest -> discarded, and the
    # /proc scan finds nothing, so there is nothing to stop rather than something wrong to kill.
    supervisor.stop("fake", force=True)
    assert os.getpid() and _alive(os.getpid())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_down_skips_what_it_did_not_start_but_force_takes_it(live):
    supervisor.start("fake")
    supervisor.clear_record("fake")
    assert supervisor.down() == [], "an adopted service is left alone"
    assert supervisor.resolve("fake", do_probe=False).running
    assert supervisor.down(force=True) == ["fake"]


def test_down_always_reclaims_an_orphan(live):
    """An orphan is ours by definition, so no --force is needed to clean it up."""
    supervisor.start("fake")
    rec = supervisor.read_record("fake")
    assert rec is not None
    supervisor.write_record(services.get("fake"), rec.pid)
    path = config.RUN_DIR / "fake.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["owner_pid"] = dead_pid()
    path.write_text(json.dumps(data), encoding="utf-8")
    assert supervisor.resolve("fake", do_probe=False).orphan
    assert supervisor.down() == ["fake"]


def test_up_is_additive_and_leaves_on_demand_services_alone(live):
    """`vorch up` must not start the narration server: it would block every render."""
    out = supervisor.up()
    assert [s.name for s in out] == ["fake"]
    assert not out[0].running, "a warm service is started on demand, not at bring-up"
    # Named explicitly, it does start — for auditioning a voice.
    out = supervisor.up(["fake"])
    assert out[0].running
    supervisor.stop("fake")


def test_up_reclaims_orphans_only_when_asked(live):
    supervisor.start("fake")
    path = config.RUN_DIR / "fake.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    pid, data["owner_pid"] = data["pid"], dead_pid()
    path.write_text(json.dumps(data), encoding="utf-8")
    supervisor.up()
    assert _alive(pid), "not touched without --reclaim-orphans"
    supervisor.up(reclaim_orphans=True)
    assert not _alive(pid)


# --- phases ---------------------------------------------------------------------------


def test_phase_leaves_a_warm_service_running_and_stamps_its_use(live):
    """Rewording a line and pressing Speak again must not pay another model load."""
    with supervisor.phase("fake") as state:
        assert state.running
        pid = state.pid
    assert _alive(pid), "warm: still up after the block"
    rec = supervisor.read_record("fake")
    assert rec is not None and time.time() - rec.last_used_at < 5
    # A second phase adopts the warm server instead of starting one.
    with supervisor.phase("fake") as state:
        assert state.pid == pid
    supervisor.stop("fake")


def test_phase_stops_a_phase_scoped_service(live):
    doc = live["doc"]
    doc["services"]["fake"]["lifetime"] = "phase"
    doc["services"]["fake"].pop("idle_timeout")
    live["manifest"].write_text(_yaml.safe_dump(doc), encoding="utf-8")
    services.reset_cache()
    with supervisor.phase("fake") as state:
        pid = state.pid
    assert not _alive(pid)


def test_phase_never_stops_a_server_someone_else_started(live):
    supervisor.start("fake")
    pid = supervisor.read_record("fake").pid
    supervisor.clear_record("fake")            # as if hand-started
    doc = live["doc"]
    doc["services"]["fake"]["lifetime"] = "phase"
    doc["services"]["fake"].pop("idle_timeout")
    live["manifest"].write_text(_yaml.safe_dump(doc), encoding="utf-8")
    services.reset_cache()
    with supervisor.phase("fake"):
        pass
    assert _alive(pid), "we did not start it, so it is not ours to close"
    supervisor.stop("fake", force=True)


def test_reap_idle_stops_only_ours_and_only_when_idle(live):
    supervisor.start("fake")
    assert supervisor.reap_idle() == [], "just used"
    path = config.RUN_DIR / "fake.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    pid = data["pid"]
    data["last_used_at"] = time.time() - 10_000
    path.write_text(json.dumps(data), encoding="utf-8")
    assert supervisor.reap_idle() == ["fake"]
    assert not _alive(pid)


def test_reap_idle_leaves_a_hand_started_server_alone(live):
    supervisor.start("fake")
    pid = supervisor.read_record("fake").pid
    supervisor.clear_record("fake")
    assert supervisor.reap_idle() == []
    assert _alive(pid)
    supervisor.stop("fake", force=True)


# --- headroom -------------------------------------------------------------------------


def test_ensure_headroom_frees_comfy_before_stopping_anything(world, monkeypatch):
    manifest = _yaml.safe_load(config.SERVICES_FILE.read_text(encoding="utf-8"))
    manifest["services"]["comfy"] = {
        "manage": "full", "argv": ["python", "main.py"],
        "vram": {"class": "elastic", "needs_mb": 12000, "holds_mb": 222, "free_endpoint": "/free"},
    }
    config.SERVICES_FILE.write_text(_yaml.safe_dump(manifest), encoding="utf-8")
    services.reset_cache()
    order: list[str] = []
    monkeypatch.setattr(vram, "admit", lambda need, **kw: vram.Admission(
        ok=True, need_mb=need, free_mb=2000, evict=("comfy", "tts"), reason="test")
        if not order else vram.Admission(ok=True, need_mb=need, free_mb=20000, reason="freed"))
    import vorch.clients.comfy as comfy_client
    monkeypatch.setattr(comfy_client, "free", lambda **kw: order.append("comfy-free"))
    monkeypatch.setattr(supervisor, "stop", lambda spec, **kw: order.append(f"stop:{getattr(spec, 'name', spec)}"))
    monkeypatch.setattr(supervisor, "resolve", lambda spec, **kw: supervisor.ServiceState(
        spec=spec if not isinstance(spec, str) else services.get(spec), running=True, pid=1, record=None))
    supervisor.ensure_headroom(12000, settle=0)
    assert order == ["comfy-free", "stop:tts"], "cheap first: /free before a 90 s reload"


def test_ensure_headroom_raises_when_nothing_can_be_freed(world, monkeypatch):
    monkeypatch.setattr(vram, "admit", lambda need, **kw: vram.Admission(
        ok=False, need_mb=need, free_mb=100, deficit_mb=need - 100, reason="nothing to free"))
    with pytest.raises(vram.VramUnavailable, match="nothing to free"):
        supervisor.ensure_headroom(12000)


def test_ensure_headroom_refuses_rather_than_hoping(world, monkeypatch):
    """If freeing did not actually help, that is a refusal — never a silent pass."""
    monkeypatch.setattr(vram, "admit", lambda need, **kw: vram.Admission(
        ok=True, need_mb=need, free_mb=100, evict=("comfy",), reason="try comfy"))
    import vorch.clients.comfy as comfy_client
    monkeypatch.setattr(comfy_client, "free", lambda **kw: None)
    with pytest.raises(vram.VramUnavailable):
        supervisor.ensure_headroom(12000, settle=0)


def test_evict_adopted_false_refuses_instead_of_killing(world, monkeypatch):
    """Per-service opt-out, for someone auditioning a voice who does not want it killed."""
    manifest = _yaml.safe_load(config.SERVICES_FILE.read_text(encoding="utf-8"))
    manifest["services"]["tts"]["evict_adopted"] = False
    config.SERVICES_FILE.write_text(_yaml.safe_dump(manifest), encoding="utf-8")
    services.reset_cache()
    monkeypatch.setattr(vram, "admit", lambda need, **kw: vram.Admission(
        ok=True, need_mb=need, free_mb=2000, evict=("tts",), reason="test"))
    with pytest.raises(vram.VramUnavailable, match="started outside vorch"):
        supervisor.ensure_headroom(12000, settle=0)


# --- log rotation ---------------------------------------------------------------------


def test_rotate_log_shifts_only_when_oversized(tmp_path):
    path = tmp_path / "svc.log"
    path.write_text("small", encoding="utf-8")
    supervisor.rotate_log(path, max_bytes=1000)
    assert path.read_text(encoding="utf-8") == "small" and not (tmp_path / "svc.log.1").exists()
    path.write_text("x" * 2000, encoding="utf-8")
    supervisor.rotate_log(path, max_bytes=1000, backups=3)
    assert not path.exists() and (tmp_path / "svc.log.1").stat().st_size == 2000


def test_stop_returns_promptly_and_does_not_wait_out_the_timeout(live):
    """A SIGTERMed child sits as a zombie until it is reaped, and os.kill(pid, 0) succeeds on a
    zombie — so polling liveness alone waited the full 30 s on a process that died at once. Every
    stop, and therefore every eviction, looked hung for half a minute."""
    supervisor.start("fake")
    t0 = time.monotonic()
    supervisor.stop("fake", timeout=30)
    assert time.monotonic() - t0 < 5, "it must reap the child, not wait out the timeout"


def test_a_child_that_ignores_sigterm_is_still_killed(live):
    doc = live["doc"]
    doc["services"]["fake"]["argv"] = [
        sys.executable, "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ignoring SIGTERM', flush=True); time.sleep(60)",
        live["marker"]]
    live["manifest"].write_text(_yaml.safe_dump(doc), encoding="utf-8")
    services.reset_cache()
    state = supervisor.start("fake")
    pid = state.pid
    supervisor.stop("fake", timeout=1)
    assert not _alive(pid) or _is_zombie_now(pid)


def _is_zombie_now(pid: int) -> bool:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return raw[raw.rindex(")") + 1:].split()[0] == "Z"
    except (OSError, ValueError, IndexError):
        return False
