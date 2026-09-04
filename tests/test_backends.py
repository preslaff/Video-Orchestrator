"""ComfyUI backends: local is always present; remote ones come from backends.yaml
with auth headers resolved from environment variables (never stored)."""

from __future__ import annotations

import pytest

from vorch import backends, config


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BACKENDS_FILE", tmp_path / "backends.yaml")
    backends.reset_cache()
    yield
    backends.reset_cache()


def _write(tmp_path, text: str):
    (tmp_path / "backends.yaml").write_text(text)


def test_local_only_without_a_file(tmp_path):
    bs = backends.load()
    assert list(bs) == ["local"]
    b = bs["local"]
    assert b.url == config.COMFY_URL and b.is_local and not b.remote
    assert b.cold_start_grace == 0 and b.headers == {}
    assert backends.get(None) is b and backends.get("local") is b


def test_yaml_entry_with_modal_proxy_headers_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MK", "key-1")
    monkeypatch.setenv("MS", "secret-1")
    _write(tmp_path, """
backends:
  modal-h100:
    url: https://ws--vorch-comfy-comfy-h100.modal.run/
    auth: modal-proxy
    key_env: MK
    secret_env: MS
    request_timeout: 60
    poll_interval: 3
    cold_start_grace: 300
    gpu: H100
    cost_per_hour: 3.95
""")
    b = backends.get("modal-h100")
    assert b.remote and not b.is_local
    assert b.url == "https://ws--vorch-comfy-comfy-h100.modal.run"  # trailing slash dropped
    assert b.headers == {"Modal-Key": "key-1", "Modal-Secret": "secret-1"}
    assert (b.request_timeout, b.poll_interval, b.cold_start_grace) == (60, 3, 300)
    assert b.gpu == "H100" and b.cost_per_hour == 3.95
    assert backends.names() == ["local", "modal-h100"]


def test_remote_defaults_come_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "COMFY_COLD_START_GRACE", 111.0)
    monkeypatch.setattr(config, "COMFY_REQUEST_TIMEOUT", 7.0)
    _write(tmp_path, "backends:\n  gpu-box:\n    url: http://10.0.0.5:8188\n")
    b = backends.get("gpu-box")
    assert b.auth == "none" and b.headers == {}
    assert b.cold_start_grace == 111.0 and b.request_timeout == 7.0 and b.remote


def test_bearer_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("TOK", "abc")
    _write(tmp_path, "backends:\n  b:\n    url: https://x\n    auth: bearer\n    token_env: TOK\n")
    assert backends.get("b").headers == {"Authorization": "Bearer abc"}


def test_missing_env_var_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    monkeypatch.setenv("MS", "s")
    _write(tmp_path, "backends:\n  m:\n    url: https://x\n    auth: modal-proxy\n    key_env: NOPE_KEY\n    secret_env: MS\n")
    with pytest.raises(backends.BackendError, match="NOPE_KEY"):
        backends.load()


def test_cannot_redefine_local(tmp_path):
    _write(tmp_path, "backends:\n  local:\n    url: http://elsewhere:8188\n")
    with pytest.raises(backends.BackendError, match="cannot be redefined"):
        backends.load()


@pytest.mark.parametrize("name", ["Modal", "a b", "-x", "x" * 33, "ünï"])
def test_bad_names_rejected(tmp_path, name):
    _write(tmp_path, f"backends:\n  '{name}':\n    url: https://x\n")
    with pytest.raises(backends.BackendError, match="invalid"):
        backends.load()


def test_bad_url_and_auth_rejected(tmp_path):
    _write(tmp_path, "backends:\n  a:\n    url: ftp://x\n")
    with pytest.raises(backends.BackendError, match="url"):
        backends.load()
    backends.reset_cache()
    _write(tmp_path, "backends:\n  a:\n    url: https://x\n    auth: magic\n")
    with pytest.raises(backends.BackendError, match="unknown auth"):
        backends.load()


def test_to_dict_has_no_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("TOK", "abc")
    _write(tmp_path, "backends:\n  b:\n    url: https://x\n    auth: bearer\n    token_env: TOK\n")
    d = backends.get("b").to_dict()
    assert "headers" not in d and "abc" not in str(d)
    assert d["auth"] == "bearer" and d["remote"] is True
    assert "abc" not in repr(backends.get("b"))


def test_unknown_backend_is_an_error():
    with pytest.raises(backends.BackendError, match="unknown backend 'nope'"):
        backends.get("nope")


def test_for_workflow_defaults_to_local():
    assert backends.for_workflow("keyframe").name == "local"
    assert backends.for_spec(object()).name == "local"


def test_remote_false_entry_is_a_second_local(tmp_path):
    _write(tmp_path, "backends:\n  second:\n    url: http://localhost:8189\n    remote: false\n")
    b = backends.get("second")
    assert b.is_local and b.cold_start_grace == 0


def test_bad_yaml_is_an_error(tmp_path):
    _write(tmp_path, "backends: [not, a, mapping]\n")
    with pytest.raises(backends.BackendError, match="mapping"):
        backends.load()
