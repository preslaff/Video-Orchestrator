"""The ComfyUI API key for hosted api_node workflows.

The key authenticates to api.comfy.org (not to the provider), is passed per request in the
prompt's extra_data, and is a secret: it must not reach a log line, a workflow YAML or a
provenance manifest. The proxy's own error messages are opaque, so they get an actionable cause.
"""

from __future__ import annotations

import json
import logging

import pytest

from vorch import config, provenance
from vorch.clients import comfy


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.text = payload, status, json.dumps(payload)

    def json(self):
        return self._p


@pytest.fixture
def posted(monkeypatch, tmp_path):
    """Capture what would be POSTed to /prompt; finish the run immediately."""
    calls: list[dict] = []
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(config, "COMFY_OUTPUT", out)
    (out / "k.png").write_bytes(b"x")

    def fake_request(b, method, path, **kw):
        if path == "/prompt":
            calls.append(kw.get("json") or {})
            return _Resp({"prompt_id": "p1"})
        return _Resp({"p1": {"status": {"status_str": "success", "messages": []},
                             "outputs": {"9": {"images": [{"filename": "k.png", "subfolder": "", "type": "output"}]}}}})

    monkeypatch.setattr(comfy, "_request", fake_request)
    return calls


def test_key_is_sent_in_extra_data_when_set(posted, monkeypatch):
    monkeypatch.setenv(comfy.API_KEY_ENV, "comfyui-secret-abc123")
    comfy.run({"1": {"class_type": "OpenAIDalle3", "inputs": {}}})
    assert posted[-1]["extra_data"] == {"api_key_comfy_org": "comfyui-secret-abc123"}


def test_no_extra_data_when_unset_or_blank(posted, monkeypatch):
    monkeypatch.delenv(comfy.API_KEY_ENV, raising=False)
    comfy.run({"1": {"class_type": "SaveImage", "inputs": {}}})
    assert "extra_data" not in posted[-1]
    monkeypatch.setenv(comfy.API_KEY_ENV, "   ")  # whitespace is not a key
    comfy.run({"1": {"class_type": "SaveImage", "inputs": {}}})
    assert "extra_data" not in posted[-1]
    assert comfy.api_key() is None


def test_key_is_read_per_call_so_vorch_env_needs_no_restart(posted, monkeypatch):
    monkeypatch.delenv(comfy.API_KEY_ENV, raising=False)
    assert comfy.api_key() is None
    monkeypatch.setenv(comfy.API_KEY_ENV, "comfyui-later")
    assert comfy.api_key() == "comfyui-later"


def test_key_never_reaches_logs_or_a_manifest(posted, monkeypatch, caplog, tmp_path):
    monkeypatch.setenv(comfy.API_KEY_ENV, "comfyui-secret-abc123")
    with caplog.at_level(logging.DEBUG, logger="vorch"):
        comfy.run({"1": {"class_type": "OpenAIDalle3", "inputs": {}}})
    assert "comfyui-secret-abc123" not in caplog.text
    # provenance records the step's params; the key is transport-level, never a param
    media = tmp_path / "shot.png"
    media.write_bytes(b"x")
    provenance.write_manifest(media, "api_dalle3_t2i", {"positive": "x", "seed": 1})
    assert "comfyui-secret-abc123" not in json.dumps(provenance.read_manifest(media))


@pytest.mark.parametrize("raw,expect", [
    ("Unauthorized: Please login first to use this node.", "VORCH_COMFY_API_KEY"),
    ("Payment Required: Please add credits to your account to use this node.", "no credits left"),
    ("There is a problem with your account. Please contact support@comfy.org.", "rejected the account"),
    ("Rate Limit Exceeded: The server returned 429 after all retry attempts.", "rate-limited"),
])
def test_opaque_proxy_errors_get_an_actionable_cause(raw, expect):
    out = comfy._explain_api_node_error(raw)
    assert raw in out and expect in out


def test_unrelated_errors_are_left_alone():
    msg = "CUDA out of memory"
    assert comfy._explain_api_node_error(msg) == msg


def test_a_failed_run_surfaces_the_explained_message(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "COMFY_OUTPUT", tmp_path)

    def fake_request(b, method, path, **kw):
        if path == "/prompt":
            return _Resp({"prompt_id": "p1"})
        return _Resp({"p1": {"status": {"status_str": "error", "messages": [
            ["execution_error", {"node_type": "OpenAIDalle3", "exception_type": "Exception",
                                 "exception_message": "Unauthorized: Please login first to use this node."}]]}}})

    monkeypatch.setattr(comfy, "_request", fake_request)
    with pytest.raises(comfy.ComfyError, match="VORCH_COMFY_API_KEY"):
        comfy.run({"1": {"class_type": "OpenAIDalle3", "inputs": {}}})
