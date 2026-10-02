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

from .. import backends, cancel, config
from ..backends import ComfyBackend

log = logging.getLogger("vorch.comfy")


class ComfyError(RuntimeError):
    """Raised when ComfyUI rejects a prompt, a run fails, or the backend is unreachable."""


# Patch points for tests (no real sleeping / wall clock).
_sleep = time.sleep
_clock = time.monotonic

_RETRY_STATUS = frozenset({502, 503, 504})
_LOCAL_POLL_RETRY = 30.0  # seconds a local /history hiccup is tolerated before failing
_POLL_HEARTBEAT = 60.0    # how often a wait says it is still waiting

# ComfyUI's built-in API nodes (OpenAIDalle3, Flux2ImageNode, …) do not talk to the provider
# directly: they proxy through api.comfy.org and authenticate with a ComfyUI-org credential,
# passed per request in the prompt's extra_data. It is a secret, so it is read from the
# environment at the point of use and never stored in a workflow YAML or a provenance manifest.
API_KEY_ENV = "VORCH_COMFY_API_KEY"

# Those nodes raise a bare Exception whose *message* is the only diagnostic — exception_type is
# the generic "Exception" for all of them. Worse, a missing key and an invalid key are
# indistinguishable (both 401), and "Please login first" means nothing to someone running
# headless. Rewrite the ones with an actionable cause.
_API_NODE_HINTS: tuple[tuple[str, str], ...] = (
    ("Unauthorized: Please login first",
     f"no usable ComfyUI API key — set {API_KEY_ENV} in ~/vorch/vorch.env (get one at "
     "platform.comfy.org); if it is set, it was rejected, so check it has not expired"),
    ("Payment Required",
     "the ComfyUI API key has no credits left — top up at platform.comfy.org"),
    ("There is a problem with your account",
     "api.comfy.org rejected the account (contact support@comfy.org)"),
    ("Rate Limit Exceeded",
     "api.comfy.org rate-limited this key after retrying — wait and run it again"),
)


def api_key() -> str | None:
    """The ComfyUI API key, or None. Read per call so vorch.env changes need no restart."""
    return (os.environ.get(API_KEY_ENV) or "").strip() or None


def _explain_api_node_error(text: str) -> str:
    """Append an actionable cause to one of the API proxy's opaque messages."""
    for needle, hint in _API_NODE_HINTS:
        if needle in text:
            return f"{text} [{hint}]"
    return text


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
    stoppable: bool = True,
    **kw,
) -> requests.Response:
    """One HTTP call against ``backend``.

    Connection errors, timeouts and 502/503/504 are retried with backoff for up
    to ``retry_for`` seconds (0 = one attempt); anything else is returned as is.
    Raises ComfyError when the budget is exhausted.

    A stop requested while waiting to retry raises ``cancel.JobStopped`` instead of
    sleeping out the budget (a remote cold start grace is minutes). Calls that must
    still go out *after* a stop — the queue delete and interrupt themselves — pass
    ``stoppable=False``.
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
        if stoppable:
            cancel.check()
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


def interrupt(backend: ComfyBackend | str | None = None, prompt_id: str | None = None) -> None:
    """Best-effort ``POST /interrupt``.

    With ``prompt_id`` newer ComfyUI interrupts only that prompt (so another client's
    work on a shared instance is left alone); older builds ignore the body and stop
    whatever is executing.
    """
    b = _resolve(backend)
    body = {"prompt_id": prompt_id} if prompt_id else None
    try:
        _request(b, "POST", "/interrupt", timeout=10, stoppable=False, json=body)
    except ComfyError as e:
        log.warning("comfy %s: interrupt failed: %s", b.name, e)


def delete_queued(prompt_ids: list[str], backend: ComfyBackend | str | None = None) -> None:
    """Best-effort ``POST /queue {"delete": [...]}``: drop prompts that have not started."""
    b = _resolve(backend)
    try:
        _request(b, "POST", "/queue", timeout=10, stoppable=False, json={"delete": list(prompt_ids)})
    except ComfyError as e:
        log.warning("comfy %s: queue delete failed: %s", b.name, e)


def _stop_prompt(b: ComfyBackend, prompt_id: str) -> None:
    """A stop was requested while ``prompt_id`` was queued or running: withdraw it."""
    log.info("comfy %s: stopping prompt %s", b.name, prompt_id)
    delete_queued([prompt_id], b)
    interrupt(b, prompt_id)


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
    cancel.check()  # nothing new is submitted once a stop was requested
    client_id = str(uuid.uuid4())

    # Never retry the submit itself: a lost response after a successful POST
    # would run the workflow twice.
    payload: dict = {"prompt": wf, "client_id": client_id}
    if (key := api_key()) is not None:
        # ComfyUI treats this as sensitive and keeps it out of /history (SENSITIVE_EXTRA_DATA_KEYS).
        payload["extra_data"] = {"api_key_comfy_org": key}
    resp = _request(b, "POST", "/prompt", json=payload)
    if resp.status_code != 200:
        # ComfyUI returns a detailed validation error body on 400 — surface it.
        raise ComfyError(f"{b.name}: /prompt rejected ({resp.status_code}): {resp.text}")
    prompt_id = resp.json()["prompt_id"]

    poll_retry = b.cold_start_grace if b.remote else _LOCAL_POLL_RETRY
    deadline = _clock() + timeout
    said = _clock()
    while True:
        remaining = deadline - _clock()
        if remaining <= 0:
            break
        try:
            # Cap the socket timeout by what is left of the budget as well. Without this a
            # single stuck request can outlive the whole deadline — the loop only checks the
            # clock between calls, so it never gets the chance to give up.
            resp = _request(b, "GET", f"/history/{prompt_id}",
                            timeout=max(1.0, min(b.request_timeout, remaining)),
                            retry_for=min(poll_retry, remaining))
        except cancel.JobStopped:  # stopped while riding out a network hiccup
            _stop_prompt(b, prompt_id)
            raise
        if resp.status_code != 200:
            raise ComfyError(f"{b.name}: /history returned {resp.status_code}")
        history = resp.json()
        if prompt_id in history:
            entry = history[prompt_id]
            problems = _execution_errors(entry)
            if problems:
                raise ComfyError(f"{b.name}: workflow {prompt_id} failed: " + "; ".join(problems))
            return _extract_outputs(entry)
        # A long render is silent by nature, and silence is indistinguishable from a hang: a Wan
        # prompt once sat wedged for thirteen hours with nothing in the log between submitting it
        # and the user noticing the fans. Say so periodically, with how much budget is left.
        if _clock() - said >= _POLL_HEARTBEAT:
            said = _clock()
            log.info("comfy %s: still waiting on %s (%.0fs elapsed, %.0fs of %ss left)",
                     b.name, prompt_id, timeout - remaining, max(deadline - _clock(), 0), timeout)
        # Checked only after a response showing the prompt unfinished, so a result
        # that already arrived is never thrown away.
        if cancel.requested():
            _stop_prompt(b, prompt_id)
            cancel.check()
        _sleep(min(b.poll_interval, max(deadline - _clock(), 0)))
    log.warning("comfy %s: %s did not finish within %ss; interrupting", b.name, prompt_id, timeout)
    interrupt(b, prompt_id)
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
            out.append(f"{node}: {_explain_api_node_error(str(text))}")
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


def workflow_templates(backend: ComfyBackend | str | None = None) -> list[dict]:
    """ComfyUI's shipped example workflows, flattened from ``/templates/index.json``.

    Fetched over HTTP rather than read from the ``comfyui_workflow_templates`` package, because
    that package lives in ComfyUI's own interpreter — which is not vorch's, and for a remote
    backend is not even on this machine. Each entry keeps its category, title and description,
    which is what lets a generated YAML say something useful about itself.
    """
    b = _resolve(backend)
    resp = _request(b, "GET", "/templates/index.json")
    if resp.status_code != 200:
        raise ComfyError(f"{b.name}: /templates/index.json returned {resp.status_code}")
    out: list[dict] = []
    for group in resp.json() or []:
        if not isinstance(group, dict):
            continue
        for tpl in group.get("templates") or []:
            if isinstance(tpl, dict) and tpl.get("name"):
                out.append({**tpl, "category": group.get("category"), "group": group.get("title")})
    return out


def workflow_template(name: str, backend: ComfyBackend | str | None = None) -> dict:
    """One shipped example workflow, as the UI-format document ComfyUI serves."""
    b = _resolve(backend)
    resp = _request(b, "GET", f"/templates/{name}.json")
    if resp.status_code != 200:
        raise ComfyError(f"{b.name}: template {name!r} returned {resp.status_code}")
    return resp.json()


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
            stoppable=False,
        )
    except ComfyError:
        pass  # /free is an optimization; never fail the run over it.


def reset_cache() -> None:
    """Drop cached sessions and object_info (tests, backend reloads)."""
    for s in _sessions.values():
        s.close()
    _sessions.clear()
    _object_info_cache.clear()
