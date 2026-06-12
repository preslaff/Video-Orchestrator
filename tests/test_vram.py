"""GPU-free checks for vram.py — nvidia-smi parsing and the reclaim guard.

subprocess and comfy.free are monkeypatched so there is no real GPU/sleep.
"""

from __future__ import annotations

import subprocess

import pytest

from vorch import vram


def _fake_smi(mapping):
    """Return a fake check_output that maps a query field to a CSV line."""
    def check_output(cmd, **kw):
        query = next(a.split("=", 1)[1] for a in cmd if a.startswith("--query-gpu="))
        if query not in mapping:
            raise subprocess.CalledProcessError(1, cmd)
        return (mapping[query] + "\n").encode()
    return check_output


def test_gpu_free_mb_parses(monkeypatch):
    monkeypatch.setattr(vram.subprocess, "check_output", _fake_smi({"memory.free": "13744"}))
    assert vram.gpu_free_mb() == 13744


def test_gpu_free_mb_none_without_nvidia_smi(monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(vram.subprocess, "check_output", boom)
    assert vram.gpu_free_mb() is None
    assert vram.snapshot() is None


def test_snapshot_parses_triplet(monkeypatch):
    monkeypatch.setattr(
        vram.subprocess, "check_output",
        _fake_smi({"memory.used,memory.free,memory.total": "2091, 13744, 16303"}),
    )
    assert vram.snapshot() == {"used": 2091, "free": 13744, "total": 16303}


def test_reclaim_unconditional_frees(monkeypatch):
    freed = []
    monkeypatch.setattr(vram.comfy, "free", lambda *a, **k: freed.append(True))
    monkeypatch.setattr(vram, "gpu_free_mb", lambda: 5000)
    vram.reclaim(settle=0)
    assert freed == [True]


def test_reclaim_skips_when_above_guard(monkeypatch):
    freed = []
    monkeypatch.setattr(vram.comfy, "free", lambda *a, **k: freed.append(True))
    monkeypatch.setattr(vram, "gpu_free_mb", lambda: 14000)
    out = vram.reclaim(min_mb=10000, settle=0)
    assert freed == [] and out == 14000  # enough headroom, no reclaim


def test_reclaim_frees_when_below_guard(monkeypatch):
    freed = []
    monkeypatch.setattr(vram.comfy, "free", lambda *a, **k: freed.append(True))
    monkeypatch.setattr(vram, "gpu_free_mb", lambda: 3000)
    vram.reclaim(min_mb=10000, settle=0)
    assert freed == [True]


def test_log_state_handles_missing_smi(monkeypatch):
    monkeypatch.setattr(vram, "snapshot", lambda: None)
    vram.log_state("test")  # must not raise
