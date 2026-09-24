"""The OmniVoice client: chunking, and the contract that the caller does not touch the text."""

from __future__ import annotations

import pytest

from vorch import config
from vorch.clients import tts


def test_a_long_line_is_split_at_sentence_boundaries():
    """The accentor is trained on sentences under 250 characters and OmniVoice's own reader
    packs to 220, so a paragraph is spoken in pieces — never cut mid-word."""
    line = ("Всяка капка носи спомен за лято. " * 8).strip()
    pieces = tts.chunk(line)
    assert len(pieces) > 1
    assert all(len(p) <= config.TTS_CHUNK_CHARS for p in pieces)
    assert all(p.endswith(".") for p in pieces), "split between sentences, not inside one"
    assert " ".join(pieces).replace("  ", " ") == line

    short = "Открий своя аромат."
    assert tts.chunk(short) == [short], "a short line is left whole"
    assert tts.chunk("") == []


def test_a_sentence_longer_than_the_window_is_split_at_a_space():
    one = "дума " * 80
    pieces = tts.chunk(one.strip(), limit=100)
    assert all(len(p) <= 100 for p in pieces)
    assert not any(p.startswith(" ") or p.endswith(" ") for p in pieces)
    assert "".join(p.replace(" ", "") for p in pieces) == one.replace(" ", ""), "no word lost"


def test_the_text_is_posted_exactly_as_written(monkeypatch):
    """The server verbalises and marks stress itself. Pre-processing here would fight it — and a
    hand-placed stress mark is discarded anyway, because the accentor strips what it did not add."""
    sent = {}

    class Reply:
        content = b"RIFF....WAVE"
        def raise_for_status(self): pass

    monkeypatch.setattr(tts.requests, "post",
                        lambda url, json, timeout: sent.update(url=url, body=json) or Reply())
    tts.synth("Само за 39 лева и 90 стотинки.")
    assert sent["body"] == {"text": "Само за 39 лева и 90 стотинки."}
    assert sent["url"].endswith("/synth")


def test_a_dead_server_and_a_non_wav_reply_are_both_clear(monkeypatch):
    monkeypatch.setattr(tts.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(tts.requests.ConnectionError("refused")))
    with pytest.raises(tts.TTSError, match="not reachable"):
        tts.health()

    class NotAudio:
        content = b'{"detail":"boom"}'
        def raise_for_status(self): pass

    monkeypatch.setattr(tts.requests, "post", lambda *a, **k: NotAudio())
    with pytest.raises(tts.TTSError, match="not a WAV"):
        tts.synth("Здравей.")
    with pytest.raises(tts.TTSError, match="nothing to say"):
        tts.synth("   ")
