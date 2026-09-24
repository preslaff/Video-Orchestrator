"""Handoff export — a clean per-job folder Shotcut (or any NLE) can import.

Layout::

    <export_root>/<project slug | _unassigned>/<job8>_<idea slug>/
        README.txt                 fps, resolution, frames, clip length, order, Shotcut notes
        manifest.json              profile snapshot, prompts, seeds, verdicts, sources
        storyboard.md              per scene: description, prompts, verdicts
        s01_keyframe.png           s01_keyframe_v2.png ... (stills variants)
        s01_swapped.png            (when the master face was imposed)
        s01_narration.wav          (when the scene has a spoken line)
        s01_clip.mp4               (video jobs; the final post-processed clip)
        s01_clip_raw.mp4           (only when post-steps changed the clip)
        assembly.mp4               (optional: selected clips joined with xfade transitions)
        provenance/<file>.json     copied sidecar manifests

Idempotent: existing identical files are skipped; hardlinks when possible,
copies across filesystems; manifest/README/storyboard are rewritten every
time. Missing sources are reported, never raised — the job stays DONE.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import provenance
from . import scene_assets
from .projects import export_root, slugify
from .state import Job, Store

log = logging.getLogger("vorch.export")

UNASSIGNED = "_unassigned"


@dataclass
class ExportRecord:
    path: str
    exported_at: str
    files: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {"path": self.path, "exported_at": self.exported_at, "files": list(self.files),
                "missing": list(self.missing), "ok": self.ok, "error": self.error}

    @classmethod
    def from_dict(cls, d: dict) -> "ExportRecord":
        return cls(path=d.get("path", ""), exported_at=d.get("exported_at", ""),
                   files=list(d.get("files", []) or []), missing=list(d.get("missing", []) or []),
                   ok=bool(d.get("ok", True)), error=d.get("error", ""))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def export_dir(job: Job, project: Any | None) -> Path:
    root = export_root(project)
    bucket = project.slug if project is not None else UNASSIGNED
    return root / bucket / f"{job.id[:8]}_{slugify(job.idea or 'job', max_len=40)}"


def plan(job: Job) -> list[tuple[str, str | None]]:
    """``[(deliverable name, source path or None)]`` in delivery order."""
    items: list[tuple[str, str | None]] = []
    for s in job.scenes:
        n = s.index + 1
        if s.variants:
            for i, v in enumerate(s.variants):
                suffix = "" if i == 0 else f"_v{i + 1}"
                items.append((f"s{n:02d}_keyframe{suffix}{_ext(v.path, '.png')}", v.path))
                if v.swapped_path:
                    items.append((f"s{n:02d}_swapped{suffix}{_ext(v.swapped_path, '.png')}", v.swapped_path))
                for k, p in enumerate(v.post_paths):
                    items.append((f"s{n:02d}_post{suffix}_{k + 1}{_ext(p, '.png')}", p))
        else:
            if s.keyframe_path:
                items.append((f"s{n:02d}_keyframe{_ext(s.keyframe_path, '.png')}", s.keyframe_path))
            if s.swapped_path:
                items.append((f"s{n:02d}_swapped{_ext(s.swapped_path, '.png')}", s.swapped_path))
        if s.narration_path:
            items.append((f"s{n:02d}_narration{_ext(s.narration_path, '.wav')}", s.narration_path))
        if s.video_path:
            items.append((f"s{n:02d}_clip{_ext(s.video_path, '.mp4')}", s.video_path))
            if s.raw_video_path and s.raw_video_path != s.video_path:
                items.append((f"s{n:02d}_clip_raw{_ext(s.raw_video_path, '.mp4')}", s.raw_video_path))
    asm = job.assembly or {}
    if asm.get("ok") and asm.get("path"):
        items.append(("assembly.mp4", asm["path"]))
    return items


def _ext(path: str, default: str) -> str:
    return Path(path).suffix or default


def _place(src: Path, dst: Path) -> bool:
    """Hardlink (or copy) ``src`` to ``dst``; skip when an identical file is already there."""
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return False
            a, b = src.stat(), dst.stat()
            if a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime):
                return False
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.EOPNOTSUPP):
            raise
        shutil.copy2(src, dst)
    return True


def _verdict_dict(v) -> dict | None:
    if v is None:
        return None
    d = v.model_dump()
    d["decision"] = v.decision
    d["needs_review"] = v.needs_review
    return d


def build_manifest(job: Job, project: Any | None, store: Store | None, files: dict[str, str]) -> dict:
    scenes = []
    for s in job.scenes:
        n = s.index + 1
        seeds = {}
        for label, p in (("keyframe", s.keyframe_path), ("swapped", s.swapped_path), ("clip", s.video_path)):
            m = provenance.read_manifest(p) if p else None
            if m:
                seeds[label] = m.get("params", {}).get("seed", m.get("params", {}).get("seed_high"))
        scenes.append({
            "index": s.index,
            "description": s.description,
            "character": s.character,
            "setting": s.setting,
            "mood": s.mood,
            "keyframe_prompt": s.keyframe_prompt,
            "keyframe_negative": s.keyframe_negative,
            "animation_prompt": s.animation_prompt,
            "animation_negative": s.animation_negative,
            "mode": s.mode,
            "identity": ({"path": s.identity, "asset_id": s.identity_asset_id,
                          "label": scene_assets.identity_label(project, s)} if s.identity else None),
            "reference_asset_ids": list(s.reference_asset_ids),
            "seeds": seeds,
            "verdicts": {"keyframe": _verdict_dict(s.keyframe_verdict),
                         "video": _verdict_dict(s.video_verdict),
                         "swap": _verdict_dict(s.swap_verdict)},
            "variants": [{"path": v.path, "seed": v.seed, "verdict": _verdict_dict(v.verdict),
                          "swapped_path": v.swapped_path, "post_paths": list(v.post_paths)} for v in s.variants],
            "files": {name: src for name, src in files.items() if name.startswith(f"s{n:02d}_")},
            "sources": {"keyframe": s.keyframe_path, "swapped": s.swapped_path,
                        "clip": s.video_path, "raw_clip": s.raw_video_path},
        })
    return {
        "job_id": job.id,
        "idea": job.idea,
        "style": job.style,
        "kind": job.kind,
        "project": ({"id": project.id, "slug": project.slug, "name": project.name} if project else None),
        "profile_id": job.profile_id,
        "profile": job.profile,
        "created_at": job.created_at,
        "exported_at": _now(),
        "scenes": scenes,
        "assembly": ({k: v for k, v in job.assembly.items() if k != "versions"}
                     if (job.assembly or {}).get("ok") else None),
        "assembly_versions": list((job.assembly or {}).get("versions") or []),
        "decisions": len(store.decisions(job.id)) if store is not None else None,
    }


def build_readme(job: Job, manifest: dict, files: list[str]) -> str:
    prof = job.profile or {}
    video = prof.get("video") or {}
    kf = prof.get("keyframe") or {}
    lines = [
        f"vorch export — job {job.id}",
        f"idea: {job.idea}",
        f"kind: {job.kind}    profile: {job.profile_id or 'default'}    aspect: {prof.get('aspect', '16:9')}",
    ]
    if job.kind == "video" and video:
        secs = round(video.get("length", 0) / max(1, video.get("fps", 1)), 2)
        lines += [
            f"clips: {video.get('width')}x{video.get('height')} @ {video.get('fps')} fps, "
            f"{video.get('length')} frames (~{secs}s each), h264 mp4",
        ]
    lines += [f"stills: {kf.get('width')}x{kf.get('height')} png"]
    asm = manifest.get("assembly") or {}
    if asm.get("ok"):
        cuts = ", ".join(
            c["transition"] if c["transition"] == "none" else f"{c['transition']} {c['duration']:g}s"
            for c in (asm.get("spec") or {}).get("cuts", [])
        )
        spec = asm.get("spec") or {}
        speeds = ", ".join(
            f"#{i + 1} s{int(it.get('scene_index', 0)) + 1:02d} {it.get('speed', 1):g}×"
            for i, it in enumerate(spec.get("items", []))
            if it.get("target_seconds") is not None
        )
        mastered = (f", {asm.get('width')}x{asm.get('height')} @ {asm.get('fps'):g} fps"
                    if asm.get("width") and asm.get("fps") else "")
        lines.append(f"assembly.mp4: {asm.get('n_clips')} clip(s), {asm.get('duration')}s{mastered}"
                     + (f", cuts: {cuts}" if cuts else "") + (f", speed: {speeds}" if speeds else "")
                     + (" (smooth slow motion)" if spec.get("smooth") else "")
                     + " (xfade transitions, re-assemble from the job page)")
    lines.append("")
    lines.append("scene order:")
    for s in manifest["scenes"]:
        ident = s.get("identity")
        lines.append(f"  {s['index'] + 1:02d}. {s['description'] or s['keyframe_prompt'][:70]}"
                     + (f"  — identity: {ident['label']}" if ident else ""))
    lines += ["", "files:"] + [f"  {f}" for f in files]
    # The raw clips keep their rendered size; assembly.mp4 may have been mastered to a bigger one.
    mode_w = video.get("width") or kf.get("width")
    mode_h = video.get("height") or kf.get("height")
    mode_fps = video.get("fps") or asm.get("fps") or 24
    lines += [
        "",
        "Shotcut: File > Open (or drag the folder) — clips are named in scene order;",
        f"set the project Video Mode to match the clips ({mode_w}x{mode_h}, {mode_fps:g} fps) "
        "before adding to the timeline."
        + (f" assembly.mp4 is mastered at {asm.get('width')}x{asm.get('height')} @ {asm.get('fps'):g} fps."
           if asm.get("ok") and asm.get("width") else ""),
        "Titles and colour are done in the NLE; assembly.mp4 (when present) is a ready cut with",
        "the chosen transitions. Regenerate a scene or re-assemble from the job page if needed.",
        "manifest.json / provenance/ hold the prompts + seeds that made every file.",
    ]
    return "\n".join(lines) + "\n"


def build_storyboard(job: Job, manifest: dict) -> str:
    out = [f"# {job.idea}", ""]
    for s in manifest["scenes"]:
        out += [f"## Scene {s['index'] + 1}", ""]
        if s["description"]:
            out += [s["description"], ""]
        out += [f"- **keyframe prompt:** {s['keyframe_prompt']}"]
        if s["keyframe_negative"]:
            out += [f"- **keyframe negative:** {s['keyframe_negative']}"]
        if job.kind == "video":
            out += [f"- **animation prompt:** {s['animation_prompt']}"]
        v = s["verdicts"].get("keyframe")
        if v:
            out += [f"- **verdict:** {v['decision']} (score {v['score']})" + (f" — {v['fix_suggestion']}" if v.get("fix_suggestion") else "")]
        if s["seeds"]:
            out += ["- **seeds:** " + ", ".join(f"{k}={val}" for k, val in s["seeds"].items())]
        out.append("")
    return "\n".join(out)


def export_job(store: Store | None, job: Job, project: Any | None) -> ExportRecord:
    """Materialise the handoff folder for ``job``; never raises for missing sources."""
    dest = export_dir(job, project)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "provenance").mkdir(exist_ok=True)
    except OSError as e:
        return ExportRecord(path=str(dest), exported_at=_now(), ok=False, error=f"cannot create {dest}: {e}")
    files: dict[str, str] = {}
    missing: list[str] = []
    placed = 0
    for name, src in plan(job):
        if not src or not Path(src).is_file():
            missing.append(name if not src else f"{name} <- {src}")
            continue
        if _place(Path(src), dest / name):
            placed += 1
        files[name] = src
        side = provenance.manifest_path(src)
        if side.is_file():
            shutil.copy2(side, dest / "provenance" / f"{name}.json")
    manifest = build_manifest(job, project, store, files)
    manifest["missing"] = missing
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "README.txt").write_text(build_readme(job, manifest, sorted(files)), encoding="utf-8")
    (dest / "storyboard.md").write_text(build_storyboard(job, manifest), encoding="utf-8")
    ok = bool(files) or not plan(job)
    log.info("export %s: %d files (%d new, %d missing) -> %s", job.id[:8], len(files), placed, len(missing), dest)
    return ExportRecord(path=str(dest), exported_at=_now(), files=sorted(files), missing=missing, ok=ok,
                        error="" if ok else "no deliverables were found on disk")


def status(job: Job) -> dict:
    """What the API/UI shows: exported?, where, stale (job changed since)?"""
    if not job.export:
        return {"exported": False, "path": None, "files": [], "missing": [], "ok": None, "error": "", "stale": False}
    rec = ExportRecord.from_dict(job.export)
    stale = bool(job.updated_at and rec.exported_at and job.updated_at > rec.exported_at)
    return {"exported": True, **rec.to_dict(), "stale": stale, "exists": Path(rec.path).is_dir()}
