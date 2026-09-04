"""Assemble a job's clips into one video with per-cut transitions (ffmpeg xfade).

Sits between ``media.py`` (pure ffmpeg: probe / build_filtergraph / assemble) and
``queue.py`` (background task): knows about jobs, scenes and where the result
lives. The assembled file is written next to the clips under
``COMFY_OUTPUT/vorch/<job8>/assembly.mp4`` (served by ``/media``) with a
provenance sidecar, and ``export.py`` copies it into the handoff folder.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, media, provenance
from .media import Cut
from .state import Job, Scene

log = logging.getLogger("vorch.assembly")

DEFAULT_TRANSITION = "fade"
DEFAULT_CUT_SECONDS = 0.5
MAX_CUT_SECONDS = 5.0
SOURCES = ("final", "raw")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AssemblyItem:
    scene_index: int
    source: str = "final"      # final = scene.video_path (post-processed), raw = scene.raw_video_path


@dataclass
class AssemblySpec:
    items: list[AssemblyItem]
    cuts: list[Cut]
    fps: int | None = None          # None -> profile / first clip
    width: int | None = None
    height: int | None = None
    crf: int = 18
    encoder: str = "libx264"

    def to_dict(self) -> dict:
        return {
            "items": [asdict(i) for i in self.items],
            "cuts": [asdict(c) for c in self.cuts],
            "fps": self.fps, "width": self.width, "height": self.height,
            "crf": self.crf, "encoder": self.encoder,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssemblySpec":
        return cls(
            items=[AssemblyItem(int(i["scene_index"]), str(i.get("source", "final"))) for i in d.get("items", [])],
            cuts=[Cut(str(c.get("transition", DEFAULT_TRANSITION)), float(c.get("duration", DEFAULT_CUT_SECONDS)))
                  for c in d.get("cuts", [])],
            fps=d.get("fps"), width=d.get("width"), height=d.get("height"),
            crf=int(d.get("crf", 18)), encoder=str(d.get("encoder", "libx264")),
        )


@dataclass
class AssemblyRecord:
    path: str
    created_at: str
    duration: float = 0.0
    n_clips: int = 0
    spec: dict = field(default_factory=dict)
    ok: bool = True
    error: str = ""
    size_bytes: int = 0
    fps: float | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AssemblyRecord":
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        known.setdefault("path", d.get("path", ""))
        known.setdefault("created_at", d.get("created_at", ""))
        return cls(**known)


# --- job helpers --------------------------------------------------------------------


def clips_of(job: Job) -> list[Scene]:
    """Scenes that have a final clip, in scene order."""
    return [s for s in sorted(job.scenes, key=lambda x: x.index) if s.video_path]


def default_spec(job: Job) -> AssemblySpec:
    scenes = clips_of(job)
    return AssemblySpec(
        items=[AssemblyItem(s.index) for s in scenes],
        cuts=[Cut(DEFAULT_TRANSITION, DEFAULT_CUT_SECONDS) for _ in scenes[1:]],
    )


def output_path(job: Job) -> Path:
    return config.COMFY_OUTPUT / "vorch" / job.id[:8] / "assembly.mp4"


def _clip_path(scene: Scene, source: str) -> str | None:
    if source == "raw":
        return scene.raw_video_path or scene.video_path
    return scene.video_path


def profile_video(job: Job) -> dict:
    prof = job.profile or {}
    return dict(prof.get("video") or {})


def resolve(job: Job, spec: dict | None) -> tuple[AssemblySpec, list[Path]]:
    """Merge a (partial) spec over the defaults and validate it against the job.

    Raises ValueError with a message the API can return as 400.
    """
    if job.kind != "video":
        raise ValueError("only video jobs can be assembled")
    base = default_spec(job)
    if not base.items:
        raise ValueError("the job has no clips yet")
    d = dict(spec or {})
    items = [AssemblyItem(int(i["scene_index"]), str(i.get("source", "final"))) for i in d["items"]] \
        if d.get("items") is not None else base.items
    if not items:
        raise ValueError("select at least one clip")
    by_index = {s.index: s for s in job.scenes}
    seen: set[int] = set()
    paths: list[Path] = []
    for it in items:
        if it.scene_index in seen:
            raise ValueError(f"scene {it.scene_index + 1} is listed twice")
        seen.add(it.scene_index)
        scene = by_index.get(it.scene_index)
        if scene is None:
            raise ValueError(f"no scene {it.scene_index + 1}")
        if it.source not in SOURCES:
            raise ValueError(f"scene {it.scene_index + 1}: source must be one of {SOURCES}")
        p = _clip_path(scene, it.source)
        if not p:
            raise ValueError(f"scene {it.scene_index + 1} has no clip")
        if not Path(p).is_file():
            raise ValueError(f"scene {it.scene_index + 1}: clip file is missing on disk ({Path(p).name})")
        paths.append(Path(p))
    if d.get("cuts") is not None:
        cuts = [Cut(str(c.get("transition", DEFAULT_TRANSITION)), float(c.get("duration", DEFAULT_CUT_SECONDS)))
                for c in d["cuts"]]
    else:
        cuts = [Cut(DEFAULT_TRANSITION, DEFAULT_CUT_SECONDS) for _ in items[1:]]
    if len(cuts) != len(items) - 1:
        raise ValueError(f"{len(items)} clip(s) need {len(items) - 1} cut(s), got {len(cuts)}")
    known = media.transitions()
    for i, c in enumerate(cuts):
        if c.transition not in known:
            raise ValueError(f"cut {i + 1}: unknown transition {c.transition!r}")
        if c.transition != "none" and not 0 < c.duration <= MAX_CUT_SECONDS:
            raise ValueError(f"cut {i + 1}: duration must be within 0 and {MAX_CUT_SECONDS} s")
    crf = int(d.get("crf", 18))
    if not 0 <= crf <= 51:
        raise ValueError("crf must be within 0..51")
    encoder = str(d.get("encoder", "libx264"))
    if encoder not in media.ENCODERS:
        raise ValueError(f"encoder must be one of {media.ENCODERS}")
    for key in ("fps", "width", "height"):
        v = d.get(key)
        if v is not None and (not isinstance(v, int) or v <= 0):
            raise ValueError(f"{key} must be a positive integer")
    resolved = AssemblySpec(items=items, cuts=cuts, fps=d.get("fps"), width=d.get("width"),
                            height=d.get("height"), crf=crf, encoder=encoder)
    # Transition lengths vs clip lengths (needs the real durations).
    infos = [media.probe(p) for p in paths]
    fps = float(resolved.fps or profile_video(job).get("fps") or infos[0].fps or config.WAN_FPS)
    media.validate_cuts([i.frame_duration for i in infos], cuts, fps=fps)
    return resolved, paths


def clip_infos(job: Job) -> list[dict]:
    """Per-clip facts for the UI (duration/fps/size via ffprobe; None when unreadable)."""
    out = []
    for s in clips_of(job):
        row: dict[str, Any] = {"scene_index": s.index, "source": "final", "path": s.video_path,
                               "has_raw": bool(s.raw_video_path and s.raw_video_path != s.video_path),
                               "duration": None, "fps": None, "width": None, "height": None}
        try:
            info = media.probe(s.video_path)
            row.update(duration=round(info.frame_duration, 3), fps=info.fps, width=info.width, height=info.height)
        except (media.MediaError, FileNotFoundError) as e:
            row["error"] = str(e)
        out.append(row)
    return out


def target_format(job: Job, spec: AssemblySpec, first: media.MediaInfo) -> tuple[float, int, int]:
    pv = profile_video(job)
    fps = float(spec.fps or pv.get("fps") or first.fps or config.WAN_FPS)
    width = int(spec.width or pv.get("width") or first.width)
    height = int(spec.height or pv.get("height") or first.height)
    return fps, width, height


def run(job: Job, spec: AssemblySpec, paths: list[Path], *, emit: Callable[[dict], None] | None = None) -> AssemblyRecord:
    """Render the assembly; never raises (failures are recorded in the returned record)."""
    dest = output_path(job)
    created = _now()
    try:
        first = media.probe(paths[0])
        fps, width, height = target_format(job, spec, first)
        durations = [media.probe(p).frame_duration for p in paths]
        total = media.total_duration(durations, spec.cuts)
        if emit:
            emit({"job_id": job.id, "type": "step", "step": "assemble", "n_clips": len(paths),
                  "seconds": round(total, 2), "fps": fps, "width": width, "height": height})
        media.assemble(paths, spec.cuts, dest, fps=fps, size=(width, height), crf=spec.crf, encoder=spec.encoder)
        info = media.probe(dest)
        rec = AssemblyRecord(
            path=str(dest), created_at=created, duration=round(info.frame_duration, 3), n_clips=len(paths),
            spec=spec.to_dict(), ok=True, size_bytes=dest.stat().st_size, fps=info.fps,
            width=info.width, height=info.height,
        )
        provenance.write_manifest(dest, "assemble", {
            "spec": spec.to_dict(), "clips": [str(p) for p in paths], "fps": fps, "width": width,
            "height": height, "crf": spec.crf, "encoder": spec.encoder,
        }, {"kind": "assembly", "job_id": job.id})
        return rec
    except (media.MediaError, ValueError, FileNotFoundError, OSError) as e:
        log.warning("job %s: assembly failed: %s", job.id, e)
        return AssemblyRecord(path=str(dest), created_at=created, n_clips=len(paths), spec=spec.to_dict(),
                              ok=False, error=str(e))


def status(job: Job, *, active: bool = False) -> dict:
    """Everything the Assemble panel needs in one payload."""
    scenes = clips_of(job)
    record = AssemblyRecord.from_dict(job.assembly) if job.assembly else None
    exported = "assembly.mp4" in ((job.export or {}).get("files") or [])
    rec_out = None
    if record is not None:
        rec_out = {**record.to_dict(), "exists": Path(record.path).is_file(), "exported": exported}
    pv = profile_video(job)
    return {
        "available": job.kind == "video" and bool(scenes),
        "active": active,
        "transitions": list(media.transitions()),
        "default_spec": default_spec(job).to_dict(),
        "clips": clip_infos(job),
        "profile_fps": pv.get("fps"),
        "record": rec_out,
    }
