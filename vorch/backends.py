"""ComfyUI backends — the local ComfyUI plus optional remote ones (e.g. on Modal).

A backend is a ComfyUI HTTP endpoint the orchestrator can dispatch a workflow
to. ``local`` always exists (``config.COMFY_URL``, shares ``COMFY_ROOT`` with
this machine). Remote backends come from ``config.BACKENDS_FILE``
(``~/vorch/backends.yaml``)::

    backends:
      modal-h100:
        url: https://<workspace>--vorch-comfy-comfy-h100.modal.run
        auth: modal-proxy          # none | modal-proxy | bearer
        key_env: MODAL_PROXY_KEY   # env var names only — the file holds no secrets
        secret_env: MODAL_PROXY_SECRET
        request_timeout: 60        # per HTTP call (default config.COMFY_REQUEST_TIMEOUT)
        poll_interval: 3           # /history polling (default config.COMFY_POLL_INTERVAL)
        cold_start_grace: 300      # connection errors tolerated while the container boots
        gpu: H100                  # label only (provenance, UI)
        cost_per_hour: 3.95        # label only

Every workflow names its backend in its registry YAML (``backend: modal-h100``,
default ``local``); nothing else in the pipeline chooses. A remote backend's
outputs are downloaded into the local ``COMFY_OUTPUT`` layout (see
``clients.comfy.fetch_output``), so media, provenance, assembly and export do
not know the difference. Design: docs/modal-backend.md.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import config

log = logging.getLogger("vorch.backends")

LOCAL = "local"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
AUTH_MODES = ("none", "modal-proxy", "bearer")


class BackendError(ValueError):
    """The backends file is invalid or an auth env var is missing."""


@dataclass(frozen=True)
class ComfyBackend:
    name: str
    url: str
    remote: bool = False
    headers: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    request_timeout: float = 30.0
    poll_interval: float = 2.0
    cold_start_grace: float = 0.0
    gpu: str | None = None
    cost_per_hour: float | None = None
    auth: str = "none"

    @property
    def is_local(self) -> bool:
        return not self.remote

    def to_dict(self) -> dict:
        """Public view — never includes the auth headers."""
        return {
            "name": self.name,
            "url": self.url,
            "remote": self.remote,
            "auth": self.auth,
            "request_timeout": self.request_timeout,
            "poll_interval": self.poll_interval,
            "cold_start_grace": self.cold_start_grace,
            "gpu": self.gpu,
            "cost_per_hour": self.cost_per_hour,
        }


def local() -> ComfyBackend:
    """The always-present local ComfyUI (``VORCH_COMFY_URL``)."""
    return ComfyBackend(
        name=LOCAL,
        url=config.COMFY_URL.rstrip("/"),
        remote=False,
        request_timeout=config.COMFY_REQUEST_TIMEOUT,
        poll_interval=config.COMFY_POLL_INTERVAL,
        cold_start_grace=0.0,
    )


def _env(name: str, var: Any, *, what: str) -> str:
    if not var or not isinstance(var, str):
        raise BackendError(f"backend {name!r}: auth needs {what} (an environment variable name)")
    value = os.environ.get(var)
    if not value:
        raise BackendError(f"backend {name!r}: environment variable {var} ({what}) is not set")
    return value


def _headers(name: str, raw: dict) -> tuple[str, dict[str, str]]:
    auth = str(raw.get("auth") or "none")
    if auth not in AUTH_MODES:
        raise BackendError(f"backend {name!r}: unknown auth {auth!r} (want one of {AUTH_MODES})")
    if auth == "modal-proxy":
        return auth, {
            "Modal-Key": _env(name, raw.get("key_env"), what="key_env"),
            "Modal-Secret": _env(name, raw.get("secret_env"), what="secret_env"),
        }
    if auth == "bearer":
        return auth, {"Authorization": "Bearer " + _env(name, raw.get("token_env"), what="token_env")}
    return auth, {}


def _number(name: str, raw: dict, key: str, default: float | None) -> float | None:
    v = raw.get(key, default)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise BackendError(f"backend {name!r}: {key} must be a number, got {v!r}") from None
    if v < 0:
        raise BackendError(f"backend {name!r}: {key} must be >= 0")
    return v


def parse_backend(name: str, raw: Any) -> ComfyBackend:
    if not NAME_RE.match(name):
        raise BackendError(f"backend name {name!r} invalid (want {NAME_RE.pattern}; it appears in file names)")
    if name == LOCAL:
        raise BackendError(f"backend {LOCAL!r} is built in (VORCH_COMFY_URL) and cannot be redefined")
    if not isinstance(raw, dict):
        raise BackendError(f"backend {name!r}: expected a mapping")
    url = str(raw.get("url") or "").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise BackendError(f"backend {name!r}: url must start with http:// or https://")
    auth, headers = _headers(name, raw)
    remote = bool(raw.get("remote", True))
    return ComfyBackend(
        name=name,
        url=url,
        remote=remote,
        headers=headers,
        auth=auth,
        request_timeout=_number(name, raw, "request_timeout", config.COMFY_REQUEST_TIMEOUT) or 0.0,
        poll_interval=_number(name, raw, "poll_interval", config.COMFY_POLL_INTERVAL) or 0.0,
        cold_start_grace=_number(name, raw, "cold_start_grace",
                                 config.COMFY_COLD_START_GRACE if remote else 0.0) or 0.0,
        gpu=str(raw["gpu"]) if raw.get("gpu") is not None else None,
        cost_per_hour=_number(name, raw, "cost_per_hour", None),
    )


def parse(doc: Any) -> dict[str, ComfyBackend]:
    """Backends from a parsed YAML document (``local`` first). Raises BackendError."""
    out = {LOCAL: local()}
    if doc is None:
        return out
    if not isinstance(doc, dict):
        raise BackendError("backends file: YAML root must be a mapping")
    entries = doc.get("backends", doc)
    if entries is None:
        return out
    if not isinstance(entries, dict):
        raise BackendError("backends file: `backends` must be a mapping of name -> settings")
    for name, raw in entries.items():
        out[str(name)] = parse_backend(str(name), raw)
    return out


_cache: dict[str, tuple[tuple, dict[str, ComfyBackend]]] = {}  # path -> (file signature, backends)


def _signature(p: Path) -> tuple:
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return ("missing",)


def load(path: str | Path | None = None, *, force: bool = False) -> dict[str, ComfyBackend]:
    """All backends by name: ``local`` plus the entries of ``path`` (default
    ``config.BACKENDS_FILE``; a missing file means local only). Cached per path;
    the cache is invalidated when the file appears or changes (env vars are read
    at load time — a new token still needs a restart of the process)."""
    p = Path(path) if path is not None else Path(config.BACKENDS_FILE)
    key = str(p)
    sig = _signature(p)
    hit = _cache.get(key)
    if not force and hit is not None and hit[0] == sig:
        return hit[1]
    if not p.is_file():
        result = {LOCAL: local()}
    else:
        try:
            with open(p, encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            raise BackendError(f"backends file {p}: {e}") from None
        result = parse(doc)
        log.info("backends: %s (from %s)", ", ".join(result), p)
    _cache[key] = (sig, result)
    return result


def get(name: str | None) -> ComfyBackend:
    """Backend by name (``None`` -> local). Unknown names raise BackendError."""
    if not name or name == LOCAL:
        return load()[LOCAL]
    try:
        return load()[name]
    except KeyError:
        raise BackendError(
            f"unknown backend {name!r}; configured: {sorted(load())} (see {config.BACKENDS_FILE})"
        ) from None


def names() -> list[str]:
    return list(load())


def for_workflow(name: str) -> ComfyBackend:
    """The backend of registry workflow ``name`` (default local)."""
    from . import registry

    return for_spec(registry.load_registry().get(name))


def for_spec(spec: Any) -> ComfyBackend:
    """The backend a WorkflowSpec declares (``spec.backend``, default local)."""
    return get(getattr(spec, "backend", None))


def reset_cache() -> None:
    _cache.clear()
