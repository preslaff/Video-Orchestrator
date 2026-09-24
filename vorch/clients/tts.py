"""OmniVoice-BG client — Bulgarian narration in a cloned voice.

The third local service, after Gemma and ComfyUI, and spoken to the same way: plain HTTP.
``POST /synth`` takes text and returns 24 kHz mono WAV bytes.

Two things about it are not obvious and both are load-bearing:

**Send plain Cyrillic.** The server verbalises numbers and then marks stress itself, in that
order. Pre-processing here would either duplicate that or fight it — and a stress mark placed by
hand is *silently discarded*, because the served accentor strips marks from every word outside
its keep-list (measured: identical text with and without a combining acute produced byte-identical
audio). When a word comes out mis-stressed the fix is an override in OmniVoice, hot-reloaded::

    cd ~/WORK/OmniVoice && ./venv/bin/python bg_stress_add.py парИ

**It is started for a phase, not left running.** The model wants **11268 MiB** (measured
2026-09-24; this docstring long said ~5.3 GB and was wrong by over 2x). The card is 16303 MiB, so
it does not fit beside a loaded ComfyUI at all — never mind a Wan render, which wants ~22 GB with
RAM offload. ``serving()`` frees ComfyUI, starts the server, and stops it again, which is
invariant #1 applied to a third model.

**Known gap (see docs/services.md).** If a server is already answering on the port, ``serving()``
adopts it as-is and does *not* stop it afterwards — so one started by hand outlives the phase and
keeps holding 11 GB. It also cannot tell *which voice* that server was started with: the reference
arrives only as ``OMNIVOICE_BG_REF_AUDIO`` and ``/health`` does not report it, so an adopted server
running ``serve_bg.sh``'s default reads every line in a different announcer's voice, silently.
Until the service manager lands, check by hand:
``tr '\0' '\n' < /proc/<pid>/environ | grep OMNIVOICE_BG_REF_AUDIO``.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import requests

from .. import config

log = logging.getLogger("vorch.tts")


class TTSError(RuntimeError):
    """The narration server is unreachable, failed, or returned something that is not audio."""


def health(*, url: str | None = None, timeout: float = 5.0) -> dict:
    """What the server reports about itself. Raises TTSError when it is not there."""
    base = (url or config.TTS_URL).rstrip("/")
    try:
        r = requests.get(f"{base}/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        raise TTSError(f"narration server not reachable at {base}: {e}") from None


def synth(text: str, *, url: str | None = None, speed: float | None = None,
          timeout: float | None = None) -> bytes:
    """One line of Bulgarian as WAV bytes (24 kHz mono).

    ``text`` goes as written: the server owns verbalisation and stress.
    """
    line = text.strip()
    if not line:
        raise TTSError("nothing to say")
    base = (url or config.TTS_URL).rstrip("/")
    body: dict = {"text": line}
    if speed is not None:
        body["speed"] = speed
    try:
        r = requests.post(f"{base}/synth", json=body,
                          timeout=timeout if timeout is not None else config.TTS_REQUEST_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        raise TTSError(f"narration failed for {line[:60]!r}: {e}") from None
    if not r.content.startswith(b"RIFF"):
        raise TTSError(f"narration server returned {len(r.content)} bytes that are not a WAV")
    return r.content


# A sentence ends at . ! ? … — the same boundaries OmniVoice's own reader splits on.
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


def chunk(text: str, limit: int | None = None) -> list[str]:
    """Split a line into pieces the model reads well.

    OmniVoice's accentor is trained on sentences under 250 characters and its own long-form
    reader packs to 220, so anything longer is split at sentence boundaries — and, only when a
    single sentence is itself too long, at a space. Never mid-word.
    """
    cap = limit or config.TTS_CHUNK_CHARS
    out: list[str] = []
    for sentence in (s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()):
        while len(sentence) > cap:
            cut = sentence.rfind(" ", 0, cap)
            if cut <= 0:                       # one very long word: let it through whole
                break
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            if out and len(out[-1]) + 1 + len(sentence) <= cap:
                out[-1] = f"{out[-1]} {sentence}"      # pack short sentences together
            else:
                out.append(sentence)
    return out or ([text.strip()] if text.strip() else [])


def _free_comfy() -> None:
    """Ask ComfyUI to drop its models, so the narration model has somewhere to live."""
    try:
        requests.post(f"{config.COMFY_URL.rstrip('/')}/free",
                      json={"unload_models": True, "free_memory": True}, timeout=30)
    except requests.RequestException as e:      # not running, or busy: not fatal by itself
        log.info("could not free ComfyUI before narration (%s)", e)


@contextmanager
def serving(*, free_comfy: bool = True) -> Iterator[dict]:
    """Run the narration server for the duration of the block.

    Yields its ``/health``. If one is already running it is used as it is and left alone —
    which is what happens when a person started it by hand to audition a voice.
    """
    try:
        info = health(timeout=2.0)
        log.info("narration: using the server already running at %s", config.TTS_URL)
        yield info
        return
    except TTSError:
        pass

    script = config.OMNIVOICE_DIR / "serve_bg.sh"
    if not script.is_file():
        raise TTSError(f"OmniVoice not found at {config.OMNIVOICE_DIR} (no serve_bg.sh)")
    ref_text_file = config.OMNIVOICE_DIR / config.TTS_REF_TEXT_FILE
    if not ref_text_file.is_file():
        raise TTSError(f"the voice's transcript is missing: {ref_text_file}")

    if free_comfy:
        _free_comfy()
    env = {"OMNIVOICE_BG_REF_AUDIO": config.TTS_REF_AUDIO,
           "OMNIVOICE_BG_REF_TEXT": ref_text_file.read_text(encoding="utf-8").strip()}
    log.info("narration: starting OmniVoice with %s", config.TTS_REF_AUDIO)
    proc = subprocess.Popen(["bash", str(script)], cwd=str(config.OMNIVOICE_DIR),
                            env={**_environ(), **env},
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + config.TTS_START_TIMEOUT
        while True:
            if proc.poll() is not None:
                err = (proc.stderr.read() if proc.stderr else b"").decode(errors="replace")[-1500:]
                raise TTSError(f"OmniVoice exited while starting: {err.strip() or 'no output'}")
            try:
                info = health(timeout=2.0)
                break
            except TTSError:
                if time.monotonic() > deadline:
                    raise TTSError(f"OmniVoice did not answer within {config.TTS_START_TIMEOUT:g}s") from None
                time.sleep(2.0)
        log.info("narration: ready (%s)", info.get("accent_frontend", "?"))
        yield info
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.info("narration: server stopped, VRAM released")


def _environ() -> dict:
    import os

    return dict(os.environ)
