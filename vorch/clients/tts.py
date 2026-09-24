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

**It is not left running.** The model wants **11268 MiB** (measured 2026-09-24; this docstring
long said ~5.3 GB and was wrong by over 2x). The card is 16303 MiB, so it does not fit beside a
loaded ComfyUI at all — never mind a Wan render. Its lifetime is therefore managed rather than
assumed: :func:`serving` hands off to ``supervisor.phase("tts")``, which frees the card first,
verifies that a server already running is reading *our* cloned voice rather than ``serve_bg.sh``'s
default announcer (the voice arrives only as ``OMNIVOICE_BG_REF_AUDIO`` and ``/health`` never
reports it back), and leaves it **warm** afterwards so re-speaking a reworded line costs seconds.
An idle timeout and any step that needs the card both take it away again. See ``docs/services.md``.

"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager

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


@contextmanager
def serving(*, free_comfy: bool = True) -> Iterator[dict]:
    """Run the narration server for the duration of the block, and yield its ``/health``.

    A thin wrapper over :func:`vorch.supervisor.phase`, so there is exactly one lifecycle path in
    the codebase. Everything interesting happens there: adopt an instance that is already up,
    verify it is reading *our* cloned voice, make room on the card first, start it with its output
    captured and wait for readiness, and leave it warm afterwards so re-speaking a reworded line
    costs seconds instead of another 90 s load.
    """
    if not free_comfy:
        # This never really made sense as a caller's choice, and makes none now: how much has to be
        # free, and what should give it up, is decided centrally from the manifest.
        log.info("narration: free_comfy=False is ignored; VRAM headroom is arbitrated centrally")
    from .. import supervisor

    with supervisor.phase("tts") as svc:
        yield svc.health or health(timeout=2.0)
