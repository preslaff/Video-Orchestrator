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
from . import versions
from .state import Clip, Job, Scene

log = logging.getLogger("vorch.assembly")

DEFAULT_TRANSITION = "fade"
DEFAULT_CUT_SECONDS = 0.5
MAX_CUT_SECONDS = 5.0
SOURCES = ("final", "raw")

# --- delivery ---------------------------------------------------------------------------
#
# What the clips are *rendered* at is the profile's business (832x480 on a 16 GB card).
# What you *deliver* is a different question: a 1080p master, or a vertical file a platform
# will accept. The assembly is the one place that re-encodes, so mastering happens here —
# ffmpeg scales every clip to the target anyway (`media._norm`), it just needed a sensible
# target and a way to pick it.
DELIVERY_SOURCE = "source"
DELIVERY_DEFAULT_FPS = {"master_1080": 24, "social_1080": 30, "youtube_1080": 30}
_SHORT_SIDE_1080 = 1080


@dataclass(frozen=True)
class Delivery:
    """A delivery target resolved for one job: what the assembly is rendered as."""

    id: str
    label: str
    width: int | None       # None = keep the clips' own size
    height: int | None
    fps: int | None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _aspect_of(job: Job, clip_w: int, clip_h: int) -> tuple[float, float]:
    """The job's intended aspect. The profile's declared `aspect` ("16:9") wins over the clip's
    pixel ratio, because the render sizes are rounded to the model's multiple — 832×480 is a
    16:9 job that would otherwise master to 1920×1108."""
    declared = str((job.profile or {}).get("aspect") or "")
    if ":" in declared:
        try:
            w, h = (float(x) for x in declared.split(":", 1))
            if w > 0 and h > 0:
                return w, h
        except ValueError:
            pass
    return float(clip_w), float(clip_h)


def _fit_1080(aspect_w: float, aspect_h: float) -> tuple[int, int]:
    """That aspect at the 1080 class, both sides even (h264 wants 4:2:0-friendly sizes)."""
    if aspect_w > aspect_h:
        width = 1920
        height = round(width * aspect_h / aspect_w)
    elif aspect_h > aspect_w:
        height = 1920
        width = round(height * aspect_w / aspect_h)
    else:
        width = height = _SHORT_SIDE_1080
    return max(2, width - width % 2), max(2, height - height % 2)


def deliveries(job: Job) -> list[Delivery]:
    """The delivery targets offered for this job, resolved against its aspect."""
    pv = profile_video(job)
    w, h = int(pv.get("width") or 0), int(pv.get("height") or 0)
    src_fps = pv.get("fps")
    out = [Delivery(DELIVERY_SOURCE, "Source (as rendered)", None, None, None,
                    f"{w}×{h} @ {src_fps} fps" if w and h else "the clips' own size and rate")]
    if not (w and h):
        return out
    mw, mh = _fit_1080(*_aspect_of(job, w, h))
    vertical = h > w
    out.append(Delivery("master_1080", "Master 1080p", mw, mh, DELIVERY_DEFAULT_FPS["master_1080"],
                        "for an edit, a screening or a festival screener"))
    out.append(Delivery("social_1080", "Social 1080", mw, mh, DELIVERY_DEFAULT_FPS["social_1080"],
                        "Reels / TikTok / Shorts accept 23–60 fps" + ("" if vertical else "; they prefer 9:16")))
    if not vertical:
        out.append(Delivery("youtube_1080", "YouTube 1080p", 1920, 1080, DELIVERY_DEFAULT_FPS["youtube_1080"],
                            "landscape upload"))
    return out


def delivery_default(job: Job) -> str:
    """Which target is preselected: social purposes deliver for a feed, everything else a master."""
    purpose = str((job.profile or {}).get("purpose") or "")
    ids = {d.id for d in deliveries(job)}
    if purpose.startswith("social") and "social_1080" in ids:
        return "social_1080"
    return "master_1080" if "master_1080" in ids else DELIVERY_SOURCE


def delivery_by_id(job: Job, delivery_id: str | None) -> Delivery | None:
    if not delivery_id:
        return None
    found = next((d for d in deliveries(job) if d.id == delivery_id), None)
    if found is None:
        raise ValueError(f"unknown delivery {delivery_id!r}")
    return found


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AssemblyItem:
    scene_index: int
    source: str = "final"      # final = scene.video_path (post-processed), raw = scene.raw_video_path
    target_seconds: float | None = None   # play the kept part in N seconds; None = natural length
    speed: float = 1.0                    # derived by resolve(): kept duration / target_seconds
    # Which take. None = whatever the scene currently points at, which is how every spec written
    # before this field behaved, and still does.
    clip_id: str | None = None
    # The part of the clip to keep, in seconds; None = from the start / to the end. Snapped to the
    # clip's own frame grid by ``media.trim_span``. Trim first, then retime.
    start: float | None = None
    end: float | None = None


@dataclass
class AssemblySpec:
    items: list[AssemblyItem]
    cuts: list[Cut]
    fps: int | None = None          # None -> profile / first clip
    width: int | None = None
    height: int | None = None
    crf: int = 18
    encoder: str = "libx264"
    smooth: bool = False            # motion-interpolated slow motion (slow render)
    audio: bool = False             # keep the clips' audio (LTX clips carry a track; Wan clips do not)
    narration: bool = True          # lay each scene's spoken line over its clip, when it has one
    duck: float = 6.0               # how hard the clips duck under the voice (compression ratio)
    delivery: str = DELIVERY_SOURCE  # a target from `deliveries()`; width/height/fps override it
    max_mbps: float | None = None   # cap the bitrate (a platform ceiling); None = CRF alone
    grain: float = 0.0              # 0-3: light film grain, which upscaled 480p needs to not read plastic

    @property
    def speeds(self) -> list[float]:
        return [i.speed for i in self.items]

    def to_dict(self) -> dict:
        return {
            "items": [asdict(i) for i in self.items],
            "cuts": [asdict(c) for c in self.cuts],
            "fps": self.fps, "width": self.width, "height": self.height,
            "crf": self.crf, "encoder": self.encoder, "smooth": self.smooth, "audio": self.audio,
            "narration": self.narration, "duck": self.duck,
            "delivery": self.delivery, "max_mbps": self.max_mbps, "grain": self.grain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssemblySpec":
        return cls(
            items=[_item_from_dict(i) for i in d.get("items", [])],
            cuts=[Cut(str(c.get("transition", DEFAULT_TRANSITION)), float(c.get("duration", DEFAULT_CUT_SECONDS)))
                  for c in d.get("cuts", [])],
            fps=d.get("fps"), width=d.get("width"), height=d.get("height"),
            crf=int(d.get("crf", 18)), encoder=str(d.get("encoder", "libx264")),
            smooth=bool(d.get("smooth", False)), audio=bool(d.get("audio", False)),
            narration=bool(d.get("narration", True)), duck=float(d.get("duck", 6.0) or 6.0),
            delivery=str(d.get("delivery") or DELIVERY_SOURCE),
            max_mbps=float(d["max_mbps"]) if d.get("max_mbps") else None,
            grain=float(d.get("grain", 0) or 0),
        )


def _item_from_dict(i: dict) -> AssemblyItem:
    """Parse one stored item. Every field after ``source`` is optional, so a spec saved before
    takes and trims existed loads unchanged."""
    def num(key: str) -> float | None:
        v = i.get(key)
        return float(v) if v is not None else None

    return AssemblyItem(
        int(i["scene_index"]), str(i.get("source", "final")),
        target_seconds=num("target_seconds"),
        speed=float(i.get("speed", 1.0) or 1.0),
        clip_id=str(i["clip_id"]) if i.get("clip_id") else None,
        start=num("start"), end=num("end"),
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
    version: int = 0                 # 1-based; versions keep their own file assembly_v<N>.mp4
    clips: list[str] = field(default_factory=list)  # the clip files joined, in item order (stale check)

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


def any_clip_has_audio(job: Job) -> bool:
    """Does any clip in this job carry an audio track?

    A Wan clip is silent, so keeping audio used to be pointless and the default was off. An
    imported clip brings its own sound, and with narration on by default the result was that the
    voice *replaced* it — the mix ducks the clip under the voice, but only when there is a clip
    track to duck. Unreadable clips count as silent: a guess must not turn audio on by accident.
    """
    for s in clips_of(job):
        try:
            if media.probe(s.video_path).has_audio:
                return True
        except (media.MediaError, FileNotFoundError, OSError):
            continue
    return False


def default_spec(job: Job) -> AssemblySpec:
    scenes = clips_of(job)
    return AssemblySpec(
        items=[AssemblyItem(s.index) for s in scenes],
        cuts=[Cut(DEFAULT_TRANSITION, DEFAULT_CUT_SECONDS) for _ in scenes[1:]],
        # On when there is something to keep, so an imported clip's own sound survives narration;
        # off for an all-Wan timeline, where it would only add a silent track.
        audio=any_clip_has_audio(job),
    )


def output_dir(job: Job) -> Path:
    return config.COMFY_OUTPUT / "vorch" / job.id[:8]


def output_path(job: Job) -> Path:
    """The *current* assembly (what the export folder ships as assembly.mp4)."""
    return output_dir(job) / "assembly.mp4"


def version_path(job: Job, version: int) -> Path:
    return output_dir(job) / f"assembly_v{version}.mp4"


# --- versions ---------------------------------------------------------------------------
#
# ``job.assembly`` is the current record (consumers: export, API, UI). It carries
# ``versions``: every successful assembly, each with its own file, so different
# transition choices can be compared and any of them made current again.


def versions_of(job: Job) -> list[dict]:
    return list((job.assembly or {}).get("versions") or [])


def next_version(job: Job) -> int:
    return max((int(v.get("version", 0)) for v in versions_of(job)), default=0) + 1


def _strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "versions"}


def register_version(job: Job, rec: AssemblyRecord) -> dict:
    """Store ``rec`` as the current assembly and (when ok) append it to the versions list.

    Returns the new ``job.assembly`` dict (not saved).
    """
    versions = versions_of(job)
    current = rec.to_dict()
    if rec.ok:
        versions = [v for v in versions if int(v.get("version", 0)) != rec.version] + [_strip(current)]
        versions.sort(key=lambda v: int(v.get("version", 0)))
    current["versions"] = versions
    job.assembly = current
    return current


def select_version(job: Job, version: int) -> dict:
    """Make an earlier version current: copy its file over assembly.mp4 (hardlink) and its record."""
    match = next((v for v in versions_of(job) if int(v.get("version", 0)) == version), None)
    if match is None:
        raise KeyError(f"no assembly version {version}")
    src = Path(match["path"])
    if not src.is_file():
        raise FileNotFoundError(f"assembly version {version} is no longer on disk ({src.name})")
    dest = output_path(job)
    _link_replace(src, dest)
    side = provenance.manifest_path(src)
    if side.is_file():
        _link_replace(side, provenance.manifest_path(dest))
    current = dict(match)
    current["versions"] = versions_of(job)
    job.assembly = current
    return current


def delete_version(job: Job, version: int) -> dict:
    """Remove a version's file + sidecar; if it was current, the newest remaining becomes current."""
    versions = versions_of(job)
    match = next((v for v in versions if int(v.get("version", 0)) == version), None)
    if match is None:
        raise KeyError(f"no assembly version {version}")
    for f in (Path(match["path"]), provenance.manifest_path(match["path"])):
        try:
            f.unlink()
        except OSError:
            pass
    remaining = [v for v in versions if int(v.get("version", 0)) != version]
    current = dict(job.assembly or {})
    if int(current.get("version", 0)) == version:
        if remaining:
            job.assembly = {**remaining[-1], "versions": remaining}
            select_version(job, int(remaining[-1]["version"]))
            return job.assembly
        for f in (output_path(job), provenance.manifest_path(output_path(job))):
            try:
                f.unlink()
            except OSError:
                pass
        job.assembly = None
        return {}
    current["versions"] = remaining
    job.assembly = current
    return current


def _link_replace(src: Path, dest: Path) -> None:
    import os
    import shutil

    tmp = dest.with_name(dest.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    os.replace(tmp, dest)


def _where(position: int, item: "AssemblyItem") -> str:
    """How to name an item in an error. Position first, because a scene can now appear twice and
    "scene 4" alone would not say which of them."""
    return f"clip {position + 1} (scene {item.scene_index + 1})"


def _clip_path(scene: Scene, source: str, clip_id: str | None = None) -> str | None:
    if clip_id:
        take = next((c for c in scene.clips if c.id == clip_id), None)
        if take is None:
            return None
        if source == "raw":
            return take.raw_video_path or take.video_path
        return take.video_path
    if source == "raw":
        return scene.raw_video_path or scene.video_path
    return scene.video_path


def save_cut(job: Job, index: int, *, clip_id: str | None = None, start: float | None = None,
             end: float | None = None, make_active: bool = False) -> Clip:
    """Render a trimmed span to a real file and keep it as one of the scene's takes.

    A trim in a timeline is an instruction: it lives in the spec and reaches nothing but
    ``assembly.mp4``. Saved, the same cut becomes a clip like any other — offered in the take
    selector, listed in the version strip, and (with ``make_active``) the scene's deliverable, so
    the export folder carries the cut rather than the whole render.

    One ffmpeg pass over an existing file; no GPU, no model.
    """
    scene = next((s for s in job.scenes if s.index == index), None)
    if scene is None:
        raise KeyError(f"scene {index}")
    if clip_id and not any(c.id == clip_id for c in scene.clips):
        raise ValueError(f"scene {index + 1}: no take {clip_id!r} in its history")
    src = _clip_path(scene, "final", clip_id)
    if not src or not Path(src).is_file():
        raise ValueError(f"scene {index + 1}: no clip to cut")
    if start is None and end is None:
        raise ValueError(f"scene {index + 1}: nothing is trimmed, so there is no cut to save")
    info = media.probe(src)
    lo, hi = media.trim_span(info.frame_duration, start, end, info.fps or 0.0)

    folder = config.COMFY_OUTPUT / "vorch" / job.id[:8]
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"s{index:02d}_cut"
    n = 1 + sum(1 for p in folder.glob(f"{stem}_*.mp4"))
    dest = folder / f"{stem}_{n}.mp4"
    media.assemble([Path(src)], [], dest, trims=[(lo, hi)], audio=True)

    source_take = next((c for c in scene.clips if c.video_path == src), None)
    clip = Clip(
        video_path=str(dest), raw_video_path=None, post_paths=[],
        seed=source_take.seed if source_take else None,
        # The same footage, so whatever was judged about it still holds; only its length changed.
        verdict=source_take.verdict if source_take else scene.video_verdict,
        keyframe_path=source_take.keyframe_path if source_take else scene.keyframe_path,
        prompt=scene.animation_prompt, negative=scene.animation_negative,
        created_at=_now(), batch=scene.regenerations,
    )
    scene.clips.append(clip)
    if make_active:
        versions.select_clip(scene, clip)
        job.export = None          # the handoff folder holds the uncut clip
    provenance.write_manifest(dest, "cut", {
        "source": src, "start": lo, "end": hi, "seconds": round(hi - lo, 3),
    }, {"kind": "clip_cut", "job_id": job.id, "scene": index})
    log.info("job %s: scene %d cut %.3f-%.3f s -> %s", job.id, index, lo, hi, dest.name)
    return clip


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
    items = [_item_from_dict(i) for i in d["items"]] if d.get("items") is not None else base.items
    for n, it in enumerate(items):
        it.speed = 1.0  # derived below from the probed source length
        if it.target_seconds is not None and it.target_seconds <= 0:
            raise ValueError(f"{_where(n, it)}: target_seconds must be positive")
    if not items:
        raise ValueError("select at least one clip")
    # A scene may appear more than once: a repeat, or two different takes joined into one longer
    # shot. Everything downstream is positional, so the only thing that ever forbade it was a
    # guard here.
    by_index = {s.index: s for s in job.scenes}
    paths: list[Path] = []
    for n, it in enumerate(items):
        scene = by_index.get(it.scene_index)
        if scene is None:
            raise ValueError(f"no scene {it.scene_index + 1}")
        if it.source not in SOURCES:
            raise ValueError(f"{_where(n, it)}: source must be one of {SOURCES}")
        if it.clip_id and not any(c.id == it.clip_id for c in scene.clips):
            raise ValueError(f"{_where(n, it)}: no take {it.clip_id!r} in its history")
        p = _clip_path(scene, it.source, it.clip_id)
        if not p:
            raise ValueError(f"{_where(n, it)} has no clip")
        if not Path(p).is_file():
            raise ValueError(f"{_where(n, it)}: clip file is missing on disk ({Path(p).name})")
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
    grain = float(d.get("grain", 0) or 0)
    if not 0 <= grain <= 3:
        raise ValueError("grain must be within 0..3")
    max_mbps = float(d["max_mbps"]) if d.get("max_mbps") else None
    if max_mbps is not None and not 0 < max_mbps <= 200:
        raise ValueError("max_mbps must be within 0 and 200")
    delivery = str(d.get("delivery") or DELIVERY_SOURCE)
    delivery_by_id(job, delivery if delivery != DELIVERY_SOURCE else None)   # ValueError -> 400
    resolved = AssemblySpec(items=items, cuts=cuts, fps=d.get("fps"), width=d.get("width"),
                            height=d.get("height"), crf=crf, encoder=encoder, smooth=bool(d.get("smooth", False)),
                            audio=bool(d.get("audio", False)), delivery=delivery, max_mbps=max_mbps, grain=grain,
                            narration=bool(d.get("narration", True)), duck=float(d.get("duck", 6.0) or 6.0))
    # Speeds from target durations, then transition lengths vs the RETIMED clip lengths.
    infos = [media.probe(p) for p in paths]
    # Trim first: the kept length is what a target duration is a target *for*, and what the
    # transitions must fit inside.
    kept: list[float] = []
    for n, (it, info) in enumerate(zip(items, infos)):
        if it.start is None and it.end is None:
            kept.append(info.frame_duration)
            continue
        try:
            lo, hi = media.trim_span(info.frame_duration, it.start, it.end, info.fps or 0.0)
        except ValueError as e:
            raise ValueError(f"{_where(n, it)}: {e}") from None
        it.start, it.end = lo, hi          # store what was actually used, snapped
        kept.append(hi - lo)
    for n, (it, src) in enumerate(zip(items, kept)):
        if it.target_seconds is None:
            continue
        speed = src / it.target_seconds
        if not media.MIN_SPEED <= speed <= media.MAX_SPEED:
            raise ValueError(
                f"{_where(n, it)}: target {it.target_seconds:g} s is outside "
                f"{src / media.MAX_SPEED:.2f}–{src * media.MAX_SPEED:.2f} s "
                f"({media.MIN_SPEED:g}×–{media.MAX_SPEED:g}× of its {src:.2f} s clip)"
            )
        it.speed = speed
    fps = float(resolved.fps or profile_video(job).get("fps") or infos[0].fps or config.WAN_FPS)
    media.validate_cuts(media.retimed(kept, resolved.speeds), cuts, fps=fps)
    return resolved, paths


def clip_infos(job: Job) -> list[dict]:
    """Per-clip facts for the UI (duration/fps/size via ffprobe; None when unreadable)."""
    out = []
    for s in clips_of(job):
        row: dict[str, Any] = {"scene_index": s.index, "source": "final", "path": s.video_path,
                               "has_raw": bool(s.raw_video_path and s.raw_video_path != s.video_path),
                               "duration": None, "fps": None, "width": None, "height": None,
                               "has_audio": False}
        try:
            info = media.probe(s.video_path)
            row.update(duration=round(info.frame_duration, 3), fps=info.fps, width=info.width,
                       height=info.height, has_audio=info.has_audio)
        except (media.MediaError, FileNotFoundError) as e:
            row["error"] = str(e)
        # Every take this scene has rendered, so the timeline can place an older one beside the
        # current take and make one longer shot without paying for a new render.
        row["takes"] = [
            {"id": c.id, "path": c.video_path, "created_at": c.created_at, "batch": c.batch,
             "active": c.video_path == s.video_path}
            for c in s.clips if c.video_path
        ]
        out.append(row)
    return out


def target_format(job: Job, spec: AssemblySpec, first: media.MediaInfo) -> tuple[float, int, int]:
    """fps and size of the rendered assembly: explicit spec > delivery target > profile > first clip."""
    pv = profile_video(job)
    d = delivery_by_id(job, spec.delivery if spec.delivery != DELIVERY_SOURCE else None)
    fps = float(spec.fps or (d.fps if d else None) or pv.get("fps") or first.fps or config.WAN_FPS)
    width = int(spec.width or (d.width if d else None) or pv.get("width") or first.width)
    height = int(spec.height or (d.height if d else None) or pv.get("height") or first.height)
    return fps, width, height


def _narration_for(job: Job, spec: AssemblySpec, durations: list[float]) -> list[tuple[Path, float]]:
    """Each scene's spoken line, paired with the second its clip starts at.

    The offsets come from ``media.item_offsets`` — the same arithmetic the picture uses for its
    transitions — so a voice cannot drift from the shot it belongs to. A scene placed twice is
    spoken twice; that is what placing it twice means.
    """
    if not spec.narration:
        return []
    by_index = {s.index: s for s in job.scenes}
    offsets = media.item_offsets(durations, spec.cuts, spec.speeds)
    out: list[tuple[Path, float]] = []
    for item, at in zip(spec.items, offsets):
        scene = by_index.get(item.scene_index)
        said = (scene.narration_path or "") if scene else ""
        if said and Path(said).is_file():
            out.append((Path(said), at))
    return out


def run(job: Job, spec: AssemblySpec, paths: list[Path], *, emit: Callable[[dict], None] | None = None) -> AssemblyRecord:
    """Render a new assembly version; never raises (failures are recorded in the returned record).

    The file is ``assembly_v<N>.mp4``; ``assembly.mp4`` (the current one) is
    updated to point at it. Register the result with :func:`register_version`.
    """
    version = next_version(job)
    dest = version_path(job, version)
    created = _now()
    try:
        first = media.probe(paths[0])
        fps, width, height = target_format(job, spec, first)
        # The KEPT length of each clip: `resolve` has already snapped any trim onto the item, so
        # a trimmed clip contributes only its span. Using the full length here would misreport
        # the total and — worse — put the narration offsets in the wrong place.
        durations = []
        for p, it in zip(paths, spec.items):
            if it.start is not None and it.end is not None:
                durations.append(it.end - it.start)
            else:
                durations.append(media.probe(p).frame_duration)
        speeds = spec.speeds
        total = media.total_duration(durations, spec.cuts, speeds)
        voices = _narration_for(job, spec, durations)
        if emit:
            emit({"job_id": job.id, "type": "step", "step": "assemble", "n_clips": len(paths),
                  "seconds": round(total, 2), "fps": fps, "width": width, "height": height,
                  "smooth": spec.smooth, "audio": spec.audio, "retimed": sum(1 for sp in speeds if sp != 1.0)})
        media.assemble(paths, spec.cuts, dest, fps=fps, size=(width, height), crf=spec.crf,
                       encoder=spec.encoder, speeds=speeds, smooth=spec.smooth, audio=spec.audio,
                       max_mbps=spec.max_mbps, grain=spec.grain,
                       trims=[(i.start, i.end) for i in spec.items],
                       narration=voices, duck=spec.duck)
        info = media.probe(dest)
        rec = AssemblyRecord(
            path=str(dest), created_at=created, duration=round(info.frame_duration, 3), n_clips=len(paths),
            spec=spec.to_dict(), ok=True, size_bytes=dest.stat().st_size, fps=info.fps,
            width=info.width, height=info.height, version=version, clips=[str(p) for p in paths],
        )
        provenance.write_manifest(dest, "assemble", {
            "spec": spec.to_dict(), "clips": [str(p) for p in paths], "fps": fps, "width": width,
            "height": height, "crf": spec.crf, "encoder": spec.encoder, "version": version,
            "speeds": speeds, "smooth": spec.smooth, "audio": spec.audio,
            "trims": [[i.start, i.end] for i in spec.items],
        }, {"kind": "assembly", "job_id": job.id})
        _link_replace(dest, output_path(job))
        side = provenance.manifest_path(dest)
        if side.is_file():
            _link_replace(side, provenance.manifest_path(output_path(job)))
        return rec
    except (media.MediaError, ValueError, FileNotFoundError, OSError) as e:
        log.warning("job %s: assembly failed: %s", job.id, e)
        return AssemblyRecord(path=str(dest), created_at=created, n_clips=len(paths), spec=spec.to_dict(),
                              ok=False, error=str(e), version=version)


def is_stale(job: Job, record: AssemblyRecord) -> bool:
    """True when a clip the assembly joined is no longer the scene's clip (regenerated,
    replaced keyframe, removed). New clips always get new file names, so comparing paths
    is enough. Records made before ``clips`` was stored fall back to the sidecar manifest;
    when the clips cannot be known the assembly is not flagged."""
    if not record.ok:
        return False
    clips = record.clips or ((provenance.read_manifest(record.path) or {}).get("params") or {}).get("clips") or []
    items = (record.spec or {}).get("items") or []
    if not clips or len(clips) != len(items):
        return False
    by_index = {s.index: s for s in job.scenes}
    for item, clip in zip(items, clips):
        scene = by_index.get(int(item["scene_index"]))
        if scene is None:
            return True
        if item.get("clip_id"):
            # A named take is not "the scene's current clip" and was never meant to be: it was
            # pinned on purpose, so re-rendering that scene does not make this assembly stale.
            continue
        if _clip_path(scene, str(item.get("source", "final"))) != clip:
            return True
    return False


def status(job: Job, *, active: bool = False) -> dict:
    """Everything the Assemble panel needs in one payload."""
    scenes = clips_of(job)
    record = AssemblyRecord.from_dict(job.assembly) if job.assembly else None
    exported = "assembly.mp4" in ((job.export or {}).get("files") or [])
    rec_out = None
    if record is not None:
        rec_out = {**record.to_dict(), "exists": Path(record.path).is_file(), "exported": exported,
                   "stale": is_stale(job, record)}
    current_version = int((job.assembly or {}).get("version", 0) or 0)
    versions = [
        {**v, "exists": Path(v.get("path", "")).is_file(), "current": int(v.get("version", 0)) == current_version}
        for v in versions_of(job)
    ]
    pv = profile_video(job)
    return {
        "available": job.kind == "video" and bool(scenes),
        "active": active,
        "transitions": list(media.transitions()),
        "default_spec": default_spec(job).to_dict(),
        "clips": clip_infos(job),
        "profile_fps": pv.get("fps"),
        "deliveries": [d.to_dict() for d in deliveries(job)],
        "delivery": delivery_default(job),
        "record": rec_out,
        "versions": versions,
    }
