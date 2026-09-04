"""ComfyUI client — dispatch a workflow, poll for completion, collect outputs.

Mirrors the flow documented in skills/comfyui-api.md:
POST /prompt -> poll /history/{id} -> read output files -> (optionally) /free.

Every call targets a *backend* (``vorch.backends``): ``backend=None`` is the
local ComfyUI, so existing callers are unchanged. Remote backends (e.g. Modal)
get auth headers, per-call timeouts, a retry budget for transient failures, a
cold-start grace period before a run, and their outputs are downloaded through
``GET /view`` into the local ``COMFY_OUTPUT`` layout (``fetch_output``) so the
rest of the pipeline never sees a remote path. Design: docs/modal-backend.md.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

import requests

from .. import backends, config
from ..backends import ComfyBackend

log = logging.getLogger("vorch.comfy")


class ComfyError(RuntimeError):
    """Raised when ComfyUI rejects a prompt, a run fails, or the backend is unreachable."""


# Patch points for tests (no real sleeping / wall clock).
_sleep = time.sleep
_clock = time.monotonic

_RETRY_STATUS = frozenset({502, 503, 504})
_LOCAL_POLL_RETRY = 30.0  # seconds a local /history hiccup is tolerated before failing


def _resolve(backend: ComfyBackend | str | None) -> ComfyBackend:
    return backend if isinstance(backend, ComfyBackend) else backends.get(backend)


# --- transport ---------------------------------------------------------------------

_sessions: dict[str, requests.Session] = {}


def _session(backend: ComfyBackend) -> requests.Session:
    """One session per backend, carrying its auth headers (never logged)."""
    s = _sessions.get(backend.name)
    if s is None:
        s = requests.Session()
        s.headers.update(backend.headers)
        _sessions[backend.name] = s
    return s


def _request(
    backend: ComfyBackend,
    method: str,
    path: str,
    *,
    timeout: float | tuple | None = None,
    retry_for: float = 0.0,
    **kw,
) -> requests.Response:
    """One HTTP call against ``backend``.

    Connection errors, timeouts and 502/503/504 are retried with backoff for up
    to ``retry_for`` seconds (0 = one attempt); anything else is returned as is.
    Raises ComfyError when the budget is exhausted.
    """
    url = backend.url + path
    timeout = backend.request_timeout if timeout is None else timeout
    deadline = _clock() + retry_for
    delay = 1.0
    while True:
        err: str
        try:
            resp = _session(backend).request(method, url, timeout=timeout, **kw)
            if resp.status_code not in _RETRY_STATUS:
                return resp
            err = f"HTTP {resp.status_code}"
        except (requests.ConnectionError, requests.Timeout) as e:
            err = f"{type(e).__name__}: {e}"
        remaining = deadline - _clock()
        if remaining <= 0:
            raise ComfyError(f"{backend.name}: {method} {path} failed ({err})")
        log.debug("comfy %s: %s %s -> %s; retrying (%.0fs left)", backend.name, method, path, err, remaining)
        _sleep(min(delay, remaining))
        delay = min(delay * 2, 10.0)


# --- status --------------------------------------------------------------------------


def system_stats(
    backend: ComfyBackend | str | None = None, *, retry_for: float = 0.0, timeout: float | None = None
) -> dict:
    """``GET /system_stats`` (ComfyUI version, devices)."""
    b = _resolve(backend)
    resp = _request(b, "GET", "/system_stats", retry_for=retry_for, timeout=timeout)
    if resp.status_code != 200:
        raise ComfyError(f"{b.name}: /system_stats returned {resp.status_code}")
    return resp.json()


def wait_ready(backend: ComfyBackend | str | None = None, *, grace: float | None = None) -> dict:
    """Block until the backend answers ``/system_stats`` (a remote container may
    still be booting). Tolerates failures for ``grace`` seconds (default: the
    backend's ``cold_start_grace``)."""
    b = _resolve(backend)
    grace = b.cold_start_grace if grace is None else grace
    t0 = _clock()
    stats = system_stats(b, retry_for=grace)
    waited = _clock() - t0
    if waited > 1:
        log.info("comfy %s: ready after %.0fs", b.name, waited)
    return stats


def interrupt(backend: ComfyBackend | str | None = None) -> None:
    """Best-effort ``POST /interrupt`` (stop the running prompt)."""
    b = _resolve(backend)
    try:
        _request(b, "POST", "/interrupt", timeout=10)
    except ComfyError as e:
        log.warning("comfy %s: interrupt failed: %s", b.name, e)


# --- run -----------------------------------------------------------------------------


def run(
    wf: dict, timeout: int | None = None, *, backend: ComfyBackend | str | None = None
) -> list[dict]:
    """Submit a workflow dict and block until it finishes.

    Returns the list of output file descriptors ({filename, subfolder, type,
    node_id}). For remote backends the container is awaited first, so a cold
    start never eats the workflow's ``timeout``. A history entry reporting an
    execution error raises ComfyError with ComfyUI's messages; on deadline the
    prompt is interrupted.
    """
    b = _resolve(backend)
    timeout = config.RUN_TIMEOUT if timeout is None else timeout
    if b.remote:
        wait_ready(b)
    client_id = str(uuid.uuid4())

    # Never retry the submit itself: a lost response after a successful POST
    # would run the workflow twice.
    resp = _request(b, "POST", "/prompt", json={"prompt": wf, "client_id": client_id})
    if resp.status_code != 200:
        # ComfyUI returns a detailed validation error body on 400 — surface it.
        raise ComfyError(f"{b.name}: /prompt rejected ({resp.status_code}): {resp.text}")
    prompt_id = resp.json()["prompt_id"]

    poll_retry = b.cold_start_grace if b.remote else _LOCAL_POLL_RETRY
    deadline = _clock() + timeout
    while True:
        remaining = deadline - _clock()
        if remaining <= 0:
            break
        resp = _request(b, "GET", f"/history/{prompt_id}", retry_for=min(poll_retry, remaining))
        if resp.status_code != 200:
            raise ComfyError(f"{b.name}: /history returned {resp.status_code}")
        history = resp.json()
        if prompt_id in history:
            entry = history[prompt_id]
            problems = _execution_errors(entry)
            if problems:
                raise ComfyError(f"{b.name}: workflow {prompt_id} failed: " + "; ".join(problems))
            return _extract_outputs(entry)
        _sleep(min(b.poll_interval, max(deadline - _clock(), 0)))
    interrupt(b)
    raise ComfyError(f"{b.name}: workflow {prompt_id} did not finish within {timeout}s (interrupted)")


def _execution_errors(history_entry: dict) -> list[str]:
    """ComfyUI's messages when a finished history entry reports an error."""
    status = history_entry.get("status") or {}
    if status.get("status_str") != "error":
        return []
    out: list[str] = []
    for msg in status.get("messages") or []:
        if not (isinstance(msg, (list, tuple)) and len(msg) == 2 and isinstance(msg[1], dict)):
            continue
        event, data = msg
        if event in ("execution_error", "execution_interrupted"):
            node = data.get("node_type") or data.get("node_id") or "?"
            text = data.get("exception_message") or data.get("exception_type") or event
            out.append(f"{node}: {text}")
    return out or ["execution error (no message)"]


def _extract_outputs(history_entry: dict) -> list[dict]:
    """Pull file descriptors from a finished history entry.

    Output files appear under outputs[node_id] keyed by media type. Filenames
    auto-increment, so always read the real name rather than assuming one.
    """
    files: list[dict] = []
    for node_id, out in history_entry.get("outputs", {}).items():
        for key in ("images", "gifs", "videos"):
            for d in out.get(key, []):
                files.append({**d, "node_id": str(node_id)})
    return files


# --- files ---------------------------------------------------------------------------


def output_path(descriptor: dict, backend: ComfyBackend | str | None = None) -> Path:
    """Resolve an output descriptor to an absolute local path under COMFY_OUTPUT.

    Local backends share the filesystem, so this is path arithmetic; a remote
    backend's file is downloaded first (``fetch_output``).
    """
    b = _resolve(backend)
    if b.remote:
        return fetch_output(descriptor, b)
    sub = descriptor.get("subfolder", "")
    return config.COMFY_OUTPUT / sub / descriptor["filename"]


def _download_target(dest_dir: Path, filename: str, tag: str) -> Path:
    """``<stem>.<backend>[-n]<ext>`` — a name that cannot collide with a local
    ComfyUI output nor with an earlier download of a same-named remote file
    (remote counters restart with every fresh container)."""
    stem, ext = os.path.splitext(filename)
    candidate = dest_dir / f"{stem}.{tag}{ext}"
    n = 1
    while candidate.exists() or candidate.with_name(candidate.name + ".part").exists():
        candidate = dest_dir / f"{stem}.{tag}-{n}{ext}"
        n += 1
    return candidate


def fetch_output(
    descriptor: dict, backend: ComfyBackend | str, dest_dir: str | Path | None = None
) -> Path:
    """Download one output file via ``GET /view`` into the local output layout.

    Streams to ``<name>.part`` and renames atomically; never overwrites.
    ``dest_dir`` defaults to ``COMFY_OUTPUT/<subfolder>``.
    """
    b = _resolve(backend)
    sub = descriptor.get("subfolder", "")
    dest_dir = Path(dest_dir) if dest_dir is not None else config.COMFY_OUTPUT / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = _download_target(dest_dir, descriptor["filename"], b.name)
    part = target.with_name(target.name + ".part")
    params = {"filename": descriptor["filename"], "subfolder": sub, "type": descriptor.get("type", "output")}
    resp = _request(b, "GET", "/view", params=params, stream=True, retry_for=b.cold_start_grace)
    if resp.status_code != 200:
        raise ComfyError(f"{b.name}: /view {descriptor['filename']!r} returned {resp.status_code}")
    try:
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        os.replace(part, target)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    finally:
        resp.close()
    log.info("comfy %s: downloaded %s/%s -> %s", b.name, sub, descriptor["filename"], target)
    return target


def upload_image(filepath: str | Path, *, backend: ComfyBackend | str | None = None) -> str:
    """Stage a local image into ComfyUI input/ so a LoadImage node can read it.

    Returns the name to assign to the LoadImage node's ``image`` field.
    """
    b = _resolve(backend)
    filepath = Path(filepath)
    data = filepath.read_bytes()  # bytes, so a retried request re-sends the whole file
    resp = _request(
        b, "POST", "/upload/image",
        files={"image": (filepath.name, data)},
        timeout=max(b.request_timeout, 120.0),
        retry_for=b.cold_start_grace if b.remote else 0.0,
    )
    if resp.status_code != 200:
        raise ComfyError(f"{b.name}: /upload/image {filepath.name!r} returned {resp.status_code}: {resp.text}")
    return resp.json()["name"]


# --- introspection / memory --------------------------------------------------------


_object_info_cache: dict[str, dict] = {}


def object_info(
    *, refresh: bool = False, timeout: float = 30, backend: ComfyBackend | str | None = None
) -> dict:
    """``GET /object_info``: every node class with its input specs (combo options =
    installed model files). Cached per process and backend; it is a multi-MB document."""
    b = _resolve(backend)
    if b.name not in _object_info_cache or refresh:
        resp = _request(b, "GET", "/object_info", timeout=max(timeout, b.request_timeout))
        if resp.status_code != 200:
            raise ComfyError(f"{b.name}: /object_info returned {resp.status_code}")
        _object_info_cache[b.name] = resp.json()
    return _object_info_cache[b.name]


def free(
    unload_models: bool = True, free_memory: bool = True, *, backend: ComfyBackend | str | None = None
) -> None:
    """Reclaim VRAM between heavy workflows (e.g. Klein -> Wan). Best effort."""
    try:
        _request(
            _resolve(backend), "POST", "/free",
            json={"unload_models": unload_models, "free_memory": free_memory}, timeout=30,
        )
    except ComfyError:
        pass  # /free is an optimization; never fail the run over it.


def reset_cache() -> None:
    """Drop cached sessions and object_info (tests, backend reloads)."""
    for s in _sessions.values():
        s.close()
    _sessions.clear()
    _object_info_cache.clear()
