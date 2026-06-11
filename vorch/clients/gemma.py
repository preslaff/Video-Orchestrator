"""Gemma 4 client — text completions via the OpenAI-compatible llama-server.

Phase 1 only needs text (storyboard / prompt generation). Vision evaluation and
strict structured output (llama.cpp grammar + Instructor/Pydantic) arrive in
Phase 2 — see DESIGN.md §11. For now JSON is parsed leniently.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests

from .. import config


class GemmaError(RuntimeError):
    """Raised when llama-server returns an error or unparseable content."""


def complete_text(
    system: str,
    user: str,
    *,
    max_tokens: int = 800,
    temperature: float = 0.7,
    timeout: int = 300,
) -> str:
    """Return the assistant message text for a system+user prompt."""
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(
        f"{config.GEMMA_URL}/v1/chat/completions", json=payload, timeout=timeout
    )
    if resp.status_code != 200:
        raise GemmaError(f"Gemma error ({resp.status_code}): {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


def encode_image(path: str | Path) -> str:
    """Read an image file and return a base64 data-URL payload string."""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()
    suffix = Path(path).suffix.lstrip(".").lower() or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{b64}"


def complete_vision(
    system: str,
    user_text: str,
    image_paths: list[str | Path],
    *,
    response_format: dict | None = None,
    max_tokens: int = 400,
    temperature: float = 0.2,
    timeout: int = 300,
) -> str:
    """Return the assistant text for a vision prompt (images + text).

    Requires llama-server started with ``--mmproj`` (else Gemma is blind —
    SKILL.md invariant 4). Pass ``response_format`` (a json_schema spec) to
    constrain output at the source; always validate the result externally,
    since llama.cpp grammar can 'fail open'.
    """
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": encode_image(p)}}
        for p in image_paths
    ]
    content.append({"type": "text", "text": user_text})

    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    resp = requests.post(
        f"{config.GEMMA_URL}/v1/chat/completions", json=payload, timeout=timeout
    )
    if resp.status_code != 200:
        raise GemmaError(f"Gemma vision error ({resp.status_code}): {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_lenient(raw: str) -> Any:
    """Parse JSON that Gemma may have wrapped in a ```json fence.

    Phase-2 note: replace this with grammar-constrained output + Pydantic
    validation; llama.cpp grammar can 'fail open' so always validate the schema.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        # Drop the opening fence (``` or ```json) and any trailing fence.
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())
