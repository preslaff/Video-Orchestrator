"""ComfyUI client transport: per-backend sessions, retries, cold-start grace,
history error status, interrupt on deadline, and /view downloads — against a
scripted fake session and a fake clock (no network, no sleeping)."""

from __future__ import annotations

import requests
import pytest

from vorch import backends, config
from vorch.backends import ComfyBackend
from vorch.clients import comfy


class Resp:
    def __init__(self, status=200, json_data=None, content=b"", text=""):
        self.status_code = status
        self._json = json_data
        self.content = content
        self.text = text or (str(json_data) if json_data is not None else "")

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def close(self):
        pass


class FakeSession:
    """``script``: ordered handlers; each is called as f(method, path, kw) and returns
    a Resp or raises. Falls back to ``default`` when the script runs out."""

    def __init__(self, script=(), default=None):
        self.script = list(script)
        self.default = default
        self.calls: list[tuple[str, str, dict]] = []
        self.headers: dict = {}

    def request(self, method, url, **kw):
        path = url.split("://", 1)[1].split("/", 1)[1] if "://" in url else url
        path = "/" + path
        self.calls.append((method, path, kw))
        handler = self.script.pop(0) if self.script else self.default
        assert handler is not None, f"unexpected call {method} {path}"
        out = handler(method, path, kw)
        if isinstance(out, BaseException):
            raise out
        return out

    def paths(self):
        return [f"{m} {p.split('?')[0]}" for m, p, _ in self.calls]


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    slept: list[float] = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    monkeypatch.setattr(comfy, "_clock", lambda: now[0])
    monkeypatch.setattr(comfy, "_sleep", sleep)
    return now, slept


@pytest.fixture
def session(monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(comfy, "_session", lambda b: fake)
    return fake


REMOTE = ComfyBackend(name="modal-h100", url="https://ws--comfy.modal.run", remote=True,
                      headers={"Modal-Key": "k", "Modal-Secret": "s"},
                      request_timeout=5, poll_interval=1, cold_start_grace=8, gpu="H100")


def _prompt_ok(m, p, kw):
    assert m == "POST" and p == "/prompt" and kw["json"]["prompt"] == {"1": {}}
    return Resp(200, {"prompt_id": "pid"})


def _history(entry):
    return lambda m, p, kw: Resp(200, {"pid": entry} if entry is not None else {})


DONE = {"status": {"status_str": "success"}, "outputs": {"9": {"images": [{"filename": "a_00001_.png", "subfolder": "vorch/j", "type": "output"}]}}}


# --- run: local ---------------------------------------------------------------------


def test_local_run_submits_and_polls_without_readiness_probe(session, clock):
    session.script = [_prompt_ok, _history(None), _history(DONE)]
    out = comfy.run({"1": {}}, timeout=60)
    assert out == [{"filename": "a_00001_.png", "subfolder": "vorch/j", "type": "output", "node_id": "9"}]
    assert session.paths() == ["POST /prompt", "GET /history/pid", "GET /history/pid"]
    assert clock[1] == [config.COMFY_POLL_INTERVAL]  # one poll sleep, local interval
    # every call carries a timeout
    assert all(c[2]["timeout"] for c in session.calls)


def test_prompt_rejection_surfaces_body(session, clock):
    session.script = [lambda m, p, kw: Resp(400, text='{"error": "bad node"}')]
    with pytest.raises(comfy.ComfyError, match="bad node"):
        comfy.run({"1": {}})


def test_history_error_status_raises_with_messages(session, clock):
    entry = {"status": {"status_str": "error", "messages": [
        ["execution_start", {}],
        ["execution_error", {"node_type": "WanImageToVideo", "exception_message": "CUDA out of memory"}],
    ]}, "outputs": {}}
    session.script = [_prompt_ok, _history(entry)]
    with pytest.raises(comfy.ComfyError, match="WanImageToVideo: CUDA out of memory"):
        comfy.run({"1": {}}, timeout=60)


def test_poll_retries_transient_errors(session, clock):
    session.script = [
        _prompt_ok,
        lambda m, p, kw: Resp(503),
        lambda m, p, kw: requests.ConnectionError("reset"),
        _history(DONE),
    ]
    assert comfy.run({"1": {}}, timeout=60)[0]["filename"] == "a_00001_.png"
    assert session.paths().count("GET /history/pid") == 3


def test_deadline_interrupts_and_raises(session, clock):
    session.script = [_prompt_ok]
    session.default = lambda m, p, kw: Resp(200, {}) if p.startswith("/history") else Resp(200)
    with pytest.raises(comfy.ComfyError, match="did not finish within 5s"):
        comfy.run({"1": {}}, timeout=5)
    assert session.paths()[-1] == "POST /interrupt"
    assert clock[0][0] >= 5


def test_never_retries_the_submit(session, clock):
    session.script = [lambda m, p, kw: requests.ConnectionError("down")]
    with pytest.raises(comfy.ComfyError, match="POST /prompt failed"):
        comfy.run({"1": {}})
    assert session.paths() == ["POST /prompt"]


# --- run: remote ----------------------------------------------------------------------


def test_remote_run_waits_for_readiness_then_prompts(session, clock):
    session.script = [
        lambda m, p, kw: requests.ConnectionError("booting"),
        lambda m, p, kw: Resp(503),
        lambda m, p, kw: Resp(200, {"system": {"comfyui_version": "0.34.0"}}),
        _prompt_ok,
        _history(DONE),
    ]
    out = comfy.run({"1": {}}, timeout=60, backend=REMOTE)
    assert out[0]["node_id"] == "9"
    assert session.paths() == ["GET /system_stats", "GET /system_stats", "GET /system_stats", "POST /prompt", "GET /history/pid"]
    assert clock[1][:2] == [1.0, 2.0]  # backoff while booting


def test_remote_deadline_starts_after_readiness(session, clock):
    # 6 s of cold start must not eat a 5 s workflow timeout.
    session.script = [
        lambda m, p, kw: requests.ConnectionError("booting"),
        lambda m, p, kw: requests.ConnectionError("booting"),
        lambda m, p, kw: requests.ConnectionError("booting"),
        lambda m, p, kw: Resp(200, {}),
        _prompt_ok,
        _history(None),
        _history(DONE),
    ]
    assert comfy.run({"1": {}}, timeout=5, backend=REMOTE)
    assert clock[0][0] > 5


def test_grace_exhausted_raises(session, clock):
    session.default = lambda m, p, kw: requests.ConnectionError("booting")
    with pytest.raises(comfy.ComfyError, match="GET /system_stats failed"):
        comfy.run({"1": {}}, backend=REMOTE)
    assert clock[0][0] == pytest.approx(REMOTE.cold_start_grace)
    assert "POST /prompt" not in session.paths()


def test_session_carries_backend_headers():
    s = comfy._session(REMOTE)
    assert s.headers["Modal-Key"] == "k" and s.headers["Modal-Secret"] == "s"
    assert comfy._session(REMOTE) is s
    assert "Modal-Key" not in comfy._session(backends.local()).headers


# --- files ----------------------------------------------------------------------------


def test_local_output_path_is_path_arithmetic(session, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path)
    d = {"filename": "x.png", "subfolder": "vorch/j", "type": "output"}
    assert comfy.output_path(d) == tmp_path / "vorch/j/x.png"
    assert session.calls == []


def test_fetch_output_streams_to_part_then_renames_and_never_overwrites(session, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path)
    seen: list[str] = []

    def view(m, p, kw):
        assert kw["stream"] is True
        assert kw["params"] == {"filename": "clip_00001_.mp4", "subfolder": "vorch/j", "type": "output"}
        seen.append(str(sorted(tmp_path.rglob("*.part"))))
        return Resp(200, content=b"x" * (3 * 1024 * 1024 + 7))

    session.default = view
    d = {"filename": "clip_00001_.mp4", "subfolder": "vorch/j", "type": "output"}
    p1 = comfy.output_path(d, REMOTE)
    assert p1 == tmp_path / "vorch/j/clip_00001_.modal-h100.mp4"
    assert p1.stat().st_size == 3 * 1024 * 1024 + 7
    assert not list(tmp_path.rglob("*.part"))
    p2 = comfy.output_path(d, REMOTE)  # same remote name again (fresh container counter)
    assert p2 == tmp_path / "vorch/j/clip_00001_.modal-h100-1.mp4" and p1.exists()
    p3 = comfy.fetch_output(d, REMOTE)
    assert p3.name == "clip_00001_.modal-h100-2.mp4"


def test_fetch_output_http_error_leaves_no_part(session, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path / "out")
    session.default = lambda m, p, kw: Resp(404)
    with pytest.raises(comfy.ComfyError, match="returned 404"):
        comfy.fetch_output({"filename": "x.png", "subfolder": ""}, REMOTE)
    assert list((tmp_path / "out").iterdir()) == []


def test_upload_image_returns_name_and_retries_for_remote(session, clock, tmp_path):
    f = tmp_path / "face.png"
    f.write_bytes(b"PNG")
    session.script = [lambda m, p, kw: Resp(502), lambda m, p, kw: Resp(200, {"name": "face (1).png"})]
    assert comfy.upload_image(f, backend=REMOTE) == "face (1).png"
    assert session.paths() == ["POST /upload/image", "POST /upload/image"]
    assert session.calls[-1][2]["files"]["image"] == ("face.png", b"PNG")


def test_upload_image_local_fails_fast(session, clock, tmp_path):
    f = tmp_path / "face.png"
    f.write_bytes(b"PNG")
    session.default = lambda m, p, kw: requests.Timeout("slow")
    with pytest.raises(comfy.ComfyError):
        comfy.upload_image(f)
    assert len(session.calls) == 1


# --- introspection / memory ---------------------------------------------------------


def test_object_info_cached_per_backend(session, clock):
    session.default = lambda m, p, kw: Resp(200, {"backend": "local" if "modal" not in kw.get("headers", {}) else "r"})
    a = comfy.object_info()
    b = comfy.object_info(backend=REMOTE)
    assert comfy.object_info() is a and comfy.object_info(backend=REMOTE) is b
    assert session.paths().count("GET /object_info") == 2
    comfy.object_info(refresh=True)
    assert session.paths().count("GET /object_info") == 3


def test_free_is_best_effort(session, clock):
    session.default = lambda m, p, kw: requests.ConnectionError("down")
    comfy.free()  # no raise
    comfy.free(backend=REMOTE)


def test_string_backend_names_resolve_via_registry(session, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKENDS_FILE", tmp_path / "b.yaml")
    (tmp_path / "b.yaml").write_text("backends:\n  box:\n    url: http://box:8188\n    cold_start_grace: 0\n")
    backends.reset_cache()
    session.default = lambda m, p, kw: Resp(200, {"ok": 1})
    assert comfy.system_stats("box") == {"ok": 1}
    with pytest.raises(backends.BackendError):
        comfy.system_stats("nope")
