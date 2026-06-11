"""GPU-free checks for the Gemma client — request payload shaping.

No live server: requests.post is monkeypatched to capture the payload. Verifies
that thinking is disabled by default (Gemma 4 is a reasoning model whose
reasoning_content otherwise eats the token budget before the answer — caught in
the first live run) and re-enabled on request.
"""

from __future__ import annotations

import pytest

from vorch.clients import gemma


class _FakeResp:
    status_code = 200

    def __init__(self, content="ok"):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture
def captured(monkeypatch):
    box = {}

    def fake_post(url, json=None, timeout=None, **kw):
        box["url"] = url
        box["payload"] = json
        return _FakeResp()

    monkeypatch.setattr(gemma.requests, "post", fake_post)
    return box


def test_complete_text_disables_thinking_by_default(captured):
    gemma.complete_text("sys", "user")
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_complete_text_think_true_omits_kwargs(captured):
    gemma.complete_text("sys", "user", think=True)
    assert "chat_template_kwargs" not in captured["payload"]


def test_complete_vision_disables_thinking_and_keeps_response_format(captured, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n")
    rf = {"type": "json_schema", "json_schema": {"name": "v", "schema": {}}}
    gemma.complete_vision("sys", "judge", [img], response_format=rf)
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["payload"]["response_format"] == rf
