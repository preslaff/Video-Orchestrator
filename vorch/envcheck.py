"""Environment check — is this machine ready to run a job with a given profile?

A structured report (ok / warn / fail per check) the UI shows on the Setup
page and the job submit consults: **fail** blocks a submit (unless forced),
**warn** does not. Checks: database + migrations, ComfyUI (+ its input/output
dirs), Gemma (+ an optional vision probe for the ``--mmproj`` mistake), ffmpeg,
GPU headroom, every registered workflow (YAML vs template, and — when ComfyUI
answers — node classes and model files actually installed), the profile
itself, and the export / projects roots.
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import psycopg
import requests

from . import config, db, registry, vram
from .clients import comfy, gemma

log = logging.getLogger("vorch.envcheck")

Status = Literal["ok", "warn", "fail"]
_ORDER = {"ok": 0, "warn": 1, "fail": 2}



def probe_png(width: int = 64, height: int = 64, rgb: tuple[int, int, int] = (200, 40, 40)) -> bytes:
    """A plain 8-bit RGB PNG for the vision probe.

    llama-server's image loader rejects tiny palette PNGs ("Failed to load image
    or audio file"), so the probe is generated as truecolor at a sane size.
    """
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


_PROBE_PNG = probe_png()


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail, "data": self.data}


@dataclass
class EnvReport:
    checks: list[CheckResult]
    checked_at: str
    duration_ms: int
    profile_id: str | None = None
    deep: bool = False

    @property
    def ok(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def status(self) -> Status:
        return max((c.status for c in self.checks), key=_ORDER.get, default="ok")

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status,
            "checked_at": self.checked_at,
            "duration_ms": self.duration_ms,
            "profile_id": self.profile_id,
            "deep": self.deep,
            "checks": [c.to_dict() for c in self.checks],
        }


class EnvironmentNotReady(RuntimeError):
    def __init__(self, report: EnvReport):
        names = ", ".join(f"{c.name}: {c.detail}" for c in report.failures)
        super().__init__(f"environment not ready — {names}")
        self.report = report


# --- individual checks -------------------------------------------------------------


def _check_database(store: Any | None) -> CheckResult:
    try:
        if store is not None:
            version = store.schema_version()
        else:
            with psycopg.connect(config.DATABASE_URL, connect_timeout=3) as conn:
                conn.execute("SELECT 1")
                version = max(db.applied_versions(conn), default=0)
    except (psycopg.Error, db.DatabaseUnavailable) as e:
        return CheckResult("database", "fail", f"PostgreSQL unreachable: {e}".splitlines()[0])
    latest = db.latest_version()
    if version < latest:
        return CheckResult("database", "fail", f"schema version {version} < {latest}; run `vorch db init`",
                           {"version": version, "latest": latest})
    return CheckResult("database", "ok", f"schema version {version}", {"version": version})


def _check_comfy() -> CheckResult:
    try:
        r = requests.get(f"{config.COMFY_URL}/system_stats", timeout=3)
        r.raise_for_status()
        stats = r.json()
    except (requests.RequestException, ValueError) as e:
        return CheckResult("comfy", "fail", f"ComfyUI unreachable at {config.COMFY_URL}: {e}")
    sysinfo = stats.get("system", {})
    devices = stats.get("devices", [])
    data = {
        "url": config.COMFY_URL,
        "version": sysinfo.get("comfyui_version"),
        "devices": [{"name": d.get("name"), "vram_total_mb": int(d.get("vram_total", 0) / 2**20)} for d in devices],
    }
    return CheckResult("comfy", "ok", f"ComfyUI {data['version'] or '?'}", data)


def _check_comfy_dirs() -> CheckResult:
    problems = []
    for label, p in (("input", config.COMFY_INPUT), ("output", config.COMFY_OUTPUT)):
        if not p.is_dir():
            problems.append(f"{label} dir missing: {p}")
        elif not os.access(p, os.W_OK):
            problems.append(f"{label} dir not writable: {p}")
    if problems:
        return CheckResult("comfy_dirs", "fail", "; ".join(problems))
    return CheckResult("comfy_dirs", "ok", f"{config.COMFY_INPUT} / {config.COMFY_OUTPUT}")


def _check_gemma() -> CheckResult:
    try:
        r = requests.get(f"{config.GEMMA_URL}/v1/models", timeout=3)
        r.raise_for_status()
        models = [m.get("id") for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError) as e:
        return CheckResult("gemma", "fail", f"Gemma (llama-server) unreachable at {config.GEMMA_URL}: {e}")
    return CheckResult("gemma", "ok", ", ".join(str(m) for m in models) or "llama-server up",
                       {"url": config.GEMMA_URL, "models": models})


def _check_gemma_vision() -> CheckResult:
    """Deep: send a tiny image; a text-only server (no --mmproj) cannot see it."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_PROBE_PNG)
        probe = f.name
    try:
        raw = gemma.complete_vision(
            "You answer with one word.",
            "Is an image attached to this message? Answer YES or NO.",
            [probe], max_tokens=5, timeout=120,
        )
    except Exception as e:  # noqa: BLE001 - any transport/parse failure is a warn here
        return CheckResult("gemma_vision", "warn", f"vision probe failed: {e}")
    finally:
        try:
            os.unlink(probe)
        except OSError:
            pass
    text = (raw or "").strip().upper()
    if "YES" in text:
        return CheckResult("gemma_vision", "ok", "vision works (mmproj loaded)")
    return CheckResult("gemma_vision", "warn",
                       f"model does not see images (reply: {raw!r}); start llama-server with --mmproj")


def _check_tool(name: str) -> CheckResult:
    path = shutil.which(name)
    if path:
        return CheckResult(name, "ok", path)
    return CheckResult(name, "warn", f"{name} not on PATH (continuity, export probing and video eval need it)")


def _check_gpu() -> CheckResult:
    snap = vram.snapshot()
    if snap is None:
        return CheckResult("gpu", "warn", "nvidia-smi unavailable; VRAM cannot be monitored")
    if snap["free"] < config.VRAM_GUARD_MB:
        return CheckResult("gpu", "warn", f"only {snap['free']} MB free (< {config.VRAM_GUARD_MB} guard); "
                           "ComfyUI may still hold models", snap)
    return CheckResult("gpu", "ok", f"{snap['free']} / {snap['total']} MB free", snap)


def _object_info(comfy_ok: bool, force: bool) -> dict | None:
    if not comfy_ok:
        return None
    try:
        return comfy.object_info(refresh=force)
    except Exception as e:  # noqa: BLE001
        log.warning("envcheck: /object_info failed: %s", e)
        return None


def _check_workflows(reg: registry.Registry, used: set[str], object_info: dict | None) -> list[CheckResult]:
    out: list[CheckResult] = []
    for spec in reg.list():
        problems = reg.validate(spec, object_info=object_info)
        in_use = spec.name in used
        if not problems:
            detail = spec.template.name + ("" if object_info is not None else " (offline check)")
            out.append(CheckResult(f"workflow:{spec.name}", "ok", detail, {"used": in_use}))
            continue
        out.append(CheckResult(
            f"workflow:{spec.name}", "fail" if in_use else "warn", "; ".join(problems),
            {"used": in_use, "problems": problems},
        ))
    for name, err in reg.errors.items():
        out.append(CheckResult(f"workflow:{name}", "warn", f"invalid user workflow: {err.splitlines()[0]}"))
    return out


def _check_profile(profile: Any | None, reg: registry.Registry) -> CheckResult:
    if profile is None:
        return CheckResult("profile", "ok", "no profile selected")
    missing = [w for w in profile.workflows_used() if w not in reg]
    if missing:
        return CheckResult("profile", "fail", f"profile {profile.id!r} uses unregistered workflow(s): {missing}")
    try:
        from .genparams import GenParams

        warnings = GenParams.from_profile(profile).validate()
    except ValueError as e:
        return CheckResult("profile", "fail", f"profile {profile.id!r} invalid: {e}")
    if warnings:
        return CheckResult("profile", "warn", "; ".join(warnings), {"id": profile.id})
    return CheckResult("profile", "ok", f"{profile.id} ({profile.kind}, {profile.aspect})", {"id": profile.id})


def _check_dir(name: str, path: Path) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise PermissionError(path)
    except OSError as e:
        return CheckResult(name, "warn", f"{path} not writable: {e}")
    return CheckResult(name, "ok", str(path))


# --- report ----------------------------------------------------------------------------

_cache: dict[tuple, tuple[float, EnvReport]] = {}


def run_checks(
    *,
    profile: Any | None = None,
    project: Any | None = None,
    store: Any | None = None,
    deep: bool = False,
    force: bool = False,
    ttl: float = 30.0,
) -> EnvReport:
    """Run every check; cached per (profile, deep) for ``ttl`` seconds unless ``force``."""
    from . import projects as projects_mod

    key = (getattr(profile, "id", None), deep)
    now = time.monotonic()
    if not force and key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]

    t0 = time.monotonic()
    checks: list[CheckResult] = [_check_database(store)]
    comfy_res = _check_comfy()
    checks += [comfy_res, _check_comfy_dirs()]
    gemma_res = _check_gemma()
    checks.append(gemma_res)
    if deep and gemma_res.status == "ok":
        checks.append(_check_gemma_vision())
    checks += [_check_tool("ffmpeg"), _check_tool("ffprobe"), _check_gpu()]

    reg = registry.load_registry(force=force)
    used = set(profile.workflows_used()) if profile is not None else set()
    info = _object_info(comfy_res.status == "ok", force)
    checks += _check_workflows(reg, used, info)
    checks.append(_check_profile(profile, reg))
    checks.append(_check_dir("export_root", projects_mod.export_root(project)))
    checks.append(_check_dir("projects_root", config.PROJECTS_ROOT))

    report = EnvReport(
        checks=checks,
        checked_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=int((time.monotonic() - t0) * 1000),
        profile_id=getattr(profile, "id", None),
        deep=deep,
    )
    _cache[key] = (now, report)
    return report


def gate(profile: Any | None = None, project: Any | None = None, *, store: Any | None = None) -> EnvReport:
    """Raise :class:`EnvironmentNotReady` on any failing check; return the report otherwise."""
    report = run_checks(profile=profile, project=project, store=store)
    if not report.ok:
        raise EnvironmentNotReady(report)
    return report


def clear_cache() -> None:
    _cache.clear()
