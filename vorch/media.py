"""Media helpers — extract a still frame from a generated video clip.

Used by sequential continuity (DESIGN.md / docs/sequential-continuity-plan.md):
the last frame of scene N's clip becomes the start image for scene N+1. Pure I/O
via ffprobe + ffmpeg (no GPU); mirrors the frame-extraction pattern in
skills/gemma-evaluate.md.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from . import config

log = logging.getLogger("vorch.media")


class MediaError(RuntimeError):
    """Raised when ffprobe/ffmpeg fail or produce no frame."""


def _count_frames(video_path: Path) -> int | None:
    """Return the exact frame count via ffprobe, or None if probing fails."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", str(video_path),
            ],
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning("ffprobe failed for %s: %s", video_path, e)
        return None
    text = out.decode().strip()
    return int(text) if text.isdigit() else None


def last_frame(video_path: str | Path, dest: str | Path | None = None) -> Path:
    """Extract the final frame of ``video_path`` to a PNG and return its path.

    Probes the exact frame count and selects frame ``TOTAL-1``; if probing fails
    (e.g. a container without frame counts), falls back to seeking near the end
    (``-sseof``). ``dest`` defaults to ``<video>_last.png`` beside the clip.
    """
    video_path = Path(video_path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    dest = Path(dest) if dest is not None else video_path.with_name(f"{video_path.stem}_last.png")

    total = _count_frames(video_path)
    if total is not None and total > 0:
        # Exact: pick the last frame by index. Clips are short (~49 frames), trivial.
        select = f"select=eq(n\\,{total - 1})"
        cmd = ["ffmpeg", "-y", "-i", str(video_path),
               "-vf", select, "-frames:v", "1", str(dest)]
    else:
        # Fallback: seek 0.2s before the end and grab one frame.
        log.warning("frame count unavailable for %s; using -sseof fallback", video_path)
        cmd = ["ffmpeg", "-y", "-sseof", "-0.2", "-i", str(video_path),
               "-frames:v", "1", str(dest)]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise MediaError(f"ffmpeg failed to extract last frame of {video_path}: {e}") from e
    if not dest.is_file():
        raise MediaError(f"ffmpeg produced no frame for {video_path}")
    log.info("last frame: %s -> %s", video_path, dest)
    return dest


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def sniff_image(path: str | Path) -> str:
    """'png' | 'jpeg' | 'other' from the file's magic bytes (extensions lie: uploads
    named .png are often WebP or AVIF)."""
    with open(path, "rb") as f:
        head = f.read(12)
    if head.startswith(PNG_MAGIC):
        return "png"
    if head.startswith(JPEG_MAGIC):
        return "jpeg"
    return "other"


def ensure_png(src: str | Path, *, max_side: int = 1024) -> Path:
    """A real PNG for image models that only decode PNG/JPEG (llama.cpp's mmproj
    loader): PNG/JPEG sources are returned as they are; anything else (WebP, AVIF,
    GIF, BMP…) is transcoded once with ffmpeg to ``<stem>.vision.png`` beside the
    source (or in the temp dir when that is not writable), bounded to ``max_side``."""
    src = Path(src).expanduser()
    kind = sniff_image(src)
    if kind in ("png", "jpeg"):
        return src
    dest = src.with_name(f"{src.stem}.vision.png")
    try:
        if dest.is_file() and dest.stat().st_mtime_ns >= src.stat().st_mtime_ns:
            return dest
        writable = os.access(dest.parent, os.W_OK)
    except OSError:
        writable = False
    if not writable:
        dest = Path(tempfile.gettempdir()) / f"vorch-vision-{abs(hash(str(src)))}.png"
        if dest.is_file():
            return dest
    vf = f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vf", vf, "-frames:v", "1", str(dest)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise MediaError(f"cannot convert {src.name} to PNG for the vision model: {e}") from e
    return dest


def downscale(src: str | Path, *, max_side: int = 768, dest: str | Path | None = None) -> Path:
    """Bound an image's long side to ``max_side``; return ``src`` unchanged when it already fits.

    ``ensure_png`` only resizes on the transcode path, so a PNG/JPEG straight from a
    camera reaches the vision model at full resolution. On CPU Gemma the mmproj
    prefill dominates and scales with image tokens, so captioning a bounded copy is
    several times faster for no useful loss of detail.

    Aspect ratio is preserved; ``dest`` defaults to ``<stem>.small.png`` beside the
    source, falling back to the temp dir when that directory is not writable.
    """
    src = Path(src).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"image not found: {src}")
    try:
        info = probe(src)
    except MediaError:
        return src  # not probeable (not an image?): leave it to the caller/server
    if info.width and info.height and max(info.width, info.height) <= max_side:
        return src
    if dest is not None:
        dest = Path(dest)
    else:
        dest = src.with_name(f"{src.stem}.small.png")
        try:
            if dest.is_file() and dest.stat().st_mtime_ns >= src.stat().st_mtime_ns:
                return dest
            writable = os.access(dest.parent, os.W_OK)
        except OSError:
            writable = False
        if not writable:
            dest = Path(tempfile.gettempdir()) / f"vorch-small-{abs(hash(str(src)))}.png"
            if dest.is_file():
                return dest
    vf = f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vf", vf, "-frames:v", "1", str(dest)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise MediaError(f"cannot downscale {src.name}: {e}") from e
    if not dest.is_file():
        raise MediaError(f"ffmpeg produced no image for {src}")
    return dest


def fit_image(src: str | Path, width: int, height: int, dest: str | Path | None = None) -> Path:
    """Cover-fit ``src`` to exactly ``width``x``height`` (scale up/down, centre crop).

    Used for edit references so a reference-driven workflow (KV-edit derives its
    output size from the first reference) produces the profile's format.
    ``dest`` defaults to ``<src>_<W>x<H>.png`` beside the source.
    """
    src = Path(src).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"image not found: {src}")
    dest = Path(dest) if dest is not None else src.with_name(f"{src.stem}_{width}x{height}.png")
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-frames:v", "1", str(dest)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise MediaError(f"ffmpeg failed to fit {src} to {width}x{height}: {e}") from e
    if not dest.is_file():
        raise MediaError(f"ffmpeg produced no image for {src}")
    return dest


def sample_frames(video_path: str | Path, n: int = 5, dest_dir: str | Path | None = None) -> list[Path]:
    """``n`` evenly spaced frames (first .. last) as PNGs, for vision evaluation."""
    video_path = Path(video_path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    n = max(1, n)
    dest_dir = Path(dest_dir) if dest_dir is not None else video_path.parent
    total = _count_frames(video_path)
    if not total:
        raise MediaError(f"cannot count frames of {video_path}")
    if n == 1:
        indices = [total - 1]
    else:
        indices = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})
    out: list[Path] = []
    for idx in indices:
        dest = dest_dir / f"{video_path.stem}_f{idx:04d}.png"
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"select=eq(n\\,{idx})",
               "-frames:v", "1", str(dest)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise MediaError(f"ffmpeg failed to extract frame {idx} of {video_path}: {e}") from e
        if not dest.is_file():
            raise MediaError(f"ffmpeg produced no frame {idx} for {video_path}")
        out.append(dest)
    return out


# --- clip assembly (ffmpeg xfade) -----------------------------------------------

# Fallback list (ffmpeg 6.1) used when `ffmpeg -h filter=xfade` cannot be parsed.
_XFADE_FALLBACK = (
    "fade", "wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright", "slideup",
    "slidedown", "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite", "radial",
    "smoothleft", "smoothright", "smoothup", "smoothdown", "circleopen", "circleclose",
    "vertopen", "vertclose", "horzopen", "horzclose", "dissolve", "pixelize", "diagtl", "diagtr",
    "diagbl", "diagbr", "hlslice", "hrslice", "vuslice", "vdslice", "hblur", "fadegrays",
    "wipetl", "wipetr", "wipebl", "wipebr", "squeezeh", "squeezev", "zoomin", "fadefast",
    "fadeslow", "hlwind", "hrwind", "vuwind", "vdwind", "coverleft", "coverright", "coverup",
    "coverdown", "revealleft", "revealright", "revealup", "revealdown",
)
ENCODERS = ("libx264", "h264_nvenc")
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")
MIN_SPEED, MAX_SPEED = 0.25, 4.0
# Cloned tail fed to minterpolate (it drops ~3 source frames of lookahead); trimmed away.
TPAD_SECONDS = 2.0
_MINTERPOLATE = "minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"


@lru_cache(maxsize=1)
def transitions() -> tuple[str, ...]:
    """``("none", <xfade transitions of the installed ffmpeg>...)``; ``custom`` excluded."""
    names: list[str] = []
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-h", "filter=xfade"], capture_output=True,
                             text=True, timeout=10).stdout
        block = out.split("transition", 1)[1] if "transition" in out else ""
        for line in block.splitlines():
            parts = line.split()
            if not parts or not line.startswith(" "):
                continue
            name = parts[0]
            if name in ("duration", "offset", "expr"):
                break
            if name.isidentifier() and name != "custom" and name not in names:
                names.append(name)
    except (OSError, subprocess.SubprocessError, IndexError):
        names = []
    if "fade" not in names:
        names = list(_XFADE_FALLBACK)
    return ("none", *names)


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int | None
    codec: str
    has_audio: bool

    @property
    def frame_duration(self) -> float:
        """Exact duration from the frame count when known (container durations are rounded)."""
        if self.nb_frames and self.fps:
            return self.nb_frames / self.fps
        return self.duration


@dataclass(frozen=True)
class Cut:
    """The transition between two consecutive clips; ``none`` is a hard cut."""

    transition: str = "fade"
    duration: float = 0.5


def probe(path: str | Path) -> MediaInfo:
    """One ffprobe call: size, fps, duration, frame count, codec, audio presence."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"media not found: {path}")
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration",
             "-show_entries", "format=duration", "-of", "json", str(path)],
            stderr=subprocess.STDOUT, timeout=60,
        )
        data = json.loads(out.decode())
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        raise MediaError(f"ffprobe failed for {path}: {e}") from e
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream in {path}")
    try:
        fps = float(Fraction(video.get("r_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb = video.get("nb_frames")
    return MediaInfo(
        width=int(video.get("width", 0)), height=int(video.get("height", 0)), fps=fps,
        duration=duration, nb_frames=int(nb) if nb and str(nb).isdigit() else None,
        codec=str(video.get("codec_name", "")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def retimed(durations: list[float], speeds: list[float] | None = None) -> list[float]:
    """Output length of each clip: source duration / speed (speed 1 = natural)."""
    if not speeds:
        return list(durations)
    return [d / s for d, s in zip(durations, speeds)]


def validate_speeds(speeds: list[float] | None, n: int) -> None:
    if speeds is None:
        return
    if len(speeds) != n:
        raise ValueError(f"{n} clip(s) need {n} speed(s), got {len(speeds)}")
    for i, sp in enumerate(speeds):
        if not (isinstance(sp, (int, float)) and MIN_SPEED <= sp <= MAX_SPEED):
            raise ValueError(f"clip {i + 1}: speed {sp!r}× is outside {MIN_SPEED:g}–{MAX_SPEED:g}×")


def total_duration(durations: list[float], cuts: list[Cut], speeds: list[float] | None = None) -> float:
    return sum(retimed(durations, speeds)) - sum(c.duration for c in cuts if c.transition != "none")


def validate_cuts(durations: list[float], cuts: list[Cut], *, fps: float) -> None:
    """Raise ValueError (naming the cut) when a cut cannot be rendered as asked."""
    if not durations:
        raise ValueError("at least one clip is required")
    if len(cuts) != len(durations) - 1:
        raise ValueError(f"{len(durations)} clip(s) need {len(durations) - 1} cut(s), got {len(cuts)}")
    known = transitions()
    frame = 1.0 / fps if fps else 0.0
    for i, c in enumerate(cuts):
        if c.transition not in known:
            raise ValueError(f"cut {i + 1}: unknown transition {c.transition!r}")
        if c.transition == "none":
            continue
        if c.duration < frame or c.duration <= 0:
            raise ValueError(f"cut {i + 1}: duration {c.duration} s is shorter than one frame ({frame:.3f} s)")
        limit = min(durations[i], durations[i + 1])
        if c.duration >= limit:
            raise ValueError(f"cut {i + 1}: transition {c.duration} s must be shorter than both clips (shortest {limit:.2f} s)")
    # A clip must keep at least one frame of its own between its in- and out-transition.
    for i, d in enumerate(durations):
        t_in = cuts[i - 1].duration if i > 0 and cuts[i - 1].transition != "none" else 0.0
        t_out = cuts[i].duration if i < len(cuts) and cuts[i].transition != "none" else 0.0
        if t_in + t_out > d - frame:
            raise ValueError(f"clip {i + 1}: transitions in ({t_in} s) + out ({t_out} s) exceed its {d:.2f} s")


def _norm(width: int, height: int, fps: float, *, speed: float = 1.0, smooth: bool = False,
          target: float | None = None, grain: float = 0.0) -> str:
    """Per-input normalisation: size, (retimed) rate, format, timebase.

    speed 1 keeps the historical chain byte for byte. Other speeds retime with
    ``setpts``; slow-downs use frame duplication (``fps``) or, with ``smooth``,
    motion-compensated interpolation. The source tail is padded with clones
    before the interpolator (it needs lookahead) and the result is trimmed to
    the exact retimed length so xfade offsets stay frame-exact.
    """
    fps_s = f"{fps:g}"
    # lanczos, not ffmpeg's default: delivering a 480p clip as 1080p is an upscale, where the
    # scaler is visible. `decrease` never stretches; a mismatched clip is padded, not cropped.
    head = (f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    # Light grain last: an upscale smooths AI footage into plastic, and a little noise reads as film.
    noise = f",noise=alls={max(0.0, min(grain, 3.0)):g}:allf=t+u" if grain else ""
    tail = f"format=yuv420p{noise},settb=AVTB,setpts=PTS-STARTPTS"
    if speed == 1.0:
        return f"{head},fps={fps_s},{tail}"
    pad = f"tpad=stop_mode=clone:stop_duration={TPAD_SECONDS:g}"
    trim = f"trim=duration={target:.4f}" if target is not None else "trim=start=0"
    if speed < 1.0 and smooth:
        rate = f"{pad},{_MINTERPOLATE.format(fps=fps_s)}"
    else:
        rate = f"fps={fps_s},{pad}"
    return f"{head},setpts=PTS/{speed:g},{rate},{trim},{tail}"


def _tempo(speed: float) -> str:
    """``atempo`` chain for ``speed`` (each stage stays within 0.5–2)."""
    parts: list[str] = []
    s = speed
    while s > 2.0:
        parts.append("atempo=2")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s *= 2.0
    if abs(s - 1.0) > 1e-9:
        parts.append(f"atempo={s:g}")
    return "".join(p + "," for p in parts)


def _audio_norm(i: int, has_audio: bool, *, speed: float, target: float) -> str:
    """Per-input audio: retimed with ``atempo``, padded/trimmed to exactly ``target``
    seconds (so ``acrossfade`` durations line up with ``xfade``); silence for
    clips without a track."""
    if not has_audio:
        return f"anullsrc=r=48000:cl=stereo,atrim=duration={target:.4f},asetpts=PTS-STARTPTS[a{i}]"
    return (f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"{_tempo(speed)}apad=whole_dur={target:.4f},atrim=duration={target:.4f},asetpts=PTS-STARTPTS[a{i}]")


def build_audio_filtergraph(
    durations: list[float], cuts: list[Cut], has_audio: list[bool], *, speeds: list[float] | None = None,
) -> tuple[str, str]:
    """``(filter_complex_part, output_label)`` for the audio track, mirroring
    :func:`build_filtergraph`: ``acrossfade`` where the video ``xfade``s, ``concat``
    at hard cuts, silence for clips without audio (LTX clips carry audio; Wan clips do not)."""
    n = len(durations)
    speeds = list(speeds) if speeds else [1.0] * n
    durations = retimed(durations, speeds)
    parts = [_audio_norm(i, has_audio[i], speed=speeds[i], target=durations[i]) for i in range(n)]
    groups: list[list[int]] = [[0]]
    for i, c in enumerate(cuts):
        if c.transition == "none":
            groups.append([i + 1])
        else:
            groups[-1].append(i + 1)
    labels: list[str] = []
    for g, members in enumerate(groups):
        current = f"[a{members[0]}]"
        for k, idx in enumerate(members[1:]):
            c = cuts[idx - 1]
            last = k == len(members) - 2
            label = f"[ag{g}]" if last else f"[ax{g}_{k}]"
            parts.append(f"{current}[a{idx}]acrossfade=d={c.duration:g}:c1=tri:c2=tri{label}")
            current = label
        labels.append(current)
    if len(labels) == 1:
        return ";".join(parts), labels[0]
    parts.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[aout]")
    return ";".join(parts), "[aout]"


def build_filtergraph(
    durations: list[float], cuts: list[Cut], *, width: int, height: int, fps: float,
    speeds: list[float] | None = None, smooth: bool = False, grain: float = 0.0,
) -> tuple[str, str]:
    """``(filter_complex, output_label)`` for ``xfade`` chains joined by ``concat`` at hard cuts.

    ``durations`` are the *source* lengths; ``speeds`` (default 1.0 each) retime
    them. Inside a chain the xfade offset is the retimed duration of the chain so
    far minus the transition; groups (split at every ``none``) are concatenated.
    """
    n = len(durations)
    speeds = list(speeds) if speeds else [1.0] * n
    durations = retimed(durations, speeds)
    parts = [
        f"[{i}:v]{_norm(width, height, fps, speed=speeds[i], smooth=smooth, target=durations[i], grain=grain)}[v{i}]"
        for i in range(n)
    ]
    groups: list[list[int]] = [[0]]
    for i, c in enumerate(cuts):
        if c.transition == "none":
            groups.append([i + 1])
        else:
            groups[-1].append(i + 1)
    group_labels: list[str] = []
    for g, members in enumerate(groups):
        if len(members) == 1:
            group_labels.append(f"[v{members[0]}]")
            continue
        current = f"[v{members[0]}]"
        elapsed = durations[members[0]]
        for k, idx in enumerate(members[1:]):
            c = cuts[idx - 1]
            offset = elapsed - c.duration
            last = k == len(members) - 2
            label = f"[g{g}]" if last else f"[x{g}_{k}]"
            parts.append(f"{current}[v{idx}]xfade=transition={c.transition}:duration={c.duration:g}:offset={offset:.4f}{label}")
            current = label
            elapsed = offset + durations[idx]
        group_labels.append(current)
    if len(group_labels) == 1:
        out = group_labels[0]
        if out == "[v0]":
            return ";".join(parts), "[v0]"
        return ";".join(parts), out
    parts.append("".join(group_labels) + f"concat=n={len(group_labels)}:v=1:a=0,format=yuv420p[out]")
    return ";".join(parts), "[out]"


def assemble(
    clips: list[str | Path],
    cuts: list[Cut],
    dest: str | Path,
    *,
    fps: float | None = None,
    size: tuple[int, int] | None = None,
    crf: int = 18,
    preset: str = "medium",
    encoder: str = "libx264",
    timeout: float | None = None,
    speeds: list[float] | None = None,
    smooth: bool = False,
    audio: bool = False,
    max_mbps: float | None = None,
    grain: float = 0.0,
) -> Path:
    """Join ``clips`` into one h264 mp4 with per-cut xfade transitions.

    ``speeds`` (one per clip, 0.25–4) retime clips; ``smooth`` renders slow-downs
    with motion interpolation (much slower). ``audio=True`` keeps the clips'
    audio (AAC; crossfaded/concatenated like the video, silence where a clip has
    none) when at least one clip has a track — otherwise the result is silent,
    as before. Size and fps default to the first clip. Written atomically
    (``.part`` then rename). Raises ValueError for a bad spec, MediaError when
    ffmpeg fails.
    """
    paths = [Path(c).expanduser() for c in clips]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"clip not found: {p}")
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
    infos = [probe(p) for p in paths]
    fps = float(fps or infos[0].fps or config.WAN_FPS)
    width, height = size or (infos[0].width, infos[0].height)
    durations = [i.frame_duration for i in infos]
    validate_speeds(speeds, len(paths))
    rt = retimed(durations, speeds)
    validate_cuts(rt, cuts, fps=fps)
    graph, out = build_filtergraph(durations, cuts, width=width, height=height, fps=fps,
                                   speeds=speeds, smooth=smooth, grain=grain)
    with_audio = bool(audio) and any(i.has_audio for i in infos)
    if with_audio:
        agraph, aout = build_audio_filtergraph(durations, cuts, [i.has_audio for i in infos], speeds=speeds)
        graph = graph + ";" + agraph
    if timeout is None:
        smooth_secs = sum(t for t, sp in zip(rt, speeds or []) if smooth and sp < 1.0)
        timeout = config.ASSEMBLE_TIMEOUT + 30 * smooth_secs

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", graph, "-map", out]
    cmd += ["-map", aout, "-c:a", "aac", "-b:a", "192k"] if with_audio else ["-an"]
    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
    # Delivery hygiene: High profile, a closed GOP of 2 s (what platforms ask for) and an explicit
    # Rec.709 tag, so a master is not left for the player to guess at.
    gop = max(1, round(fps * 2))
    cmd += ["-profile:v", "high", "-level", "4.2", "-bf", "2", "-g", str(gop), "-keyint_min", str(round(fps)),
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    if max_mbps:   # a platform ceiling (Instagram documents 25 Mbps); CRF alone is usually under it
        cmd += ["-maxrate", f"{max_mbps:g}M", "-bufsize", f"{max_mbps * 2:g}M"]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(part)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.CalledProcessError as e:
        part.unlink(missing_ok=True)
        tail = (e.stderr or b"").decode(errors="replace")[-2000:].strip()
        raise MediaError(f"ffmpeg assembly failed: {tail or e}") from e
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        part.unlink(missing_ok=True)
        raise MediaError(f"ffmpeg assembly failed: {e}") from e
    if not part.is_file() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise MediaError("ffmpeg produced no output")
    os.replace(part, dest)
    log.info("assembled %d clip(s)%s%s -> %s (%.2fs)", len(paths),
             f", {sum(1 for sp in (speeds or []) if sp != 1)} retimed{' smooth' if smooth else ''}" if speeds else "",
             ", with audio" if with_audio else "", dest, total_duration(durations, cuts, speeds))
    return dest


# --- locked regions ----------------------------------------------------------
#
# A video model redraws every frame from a latent compressed 8x in each direction, so printed
# type below roughly 32 px of cap height has nowhere to live: it comes back as a plausible
# smear, wrong in frame 0 and worse as the shot moves. No prompt, seed or sampler changes
# that. The only reliable answer is not to ask the model for it — render the shot, then put
# the real pixels back from the source frame, which is where the type is still correct.
#
# This is exact when the marked object does not move, which is why the motion prompt keeps
# printed text still (``storyboard.TEXT_SAFE_RULE``); a push-in is added afterwards, zooming
# a sharp composite instead of asking the model to redraw a brand name.


@dataclass(frozen=True)
class Region:
    """A rectangle of the frame to restore from the source image, in fractions of the
    frame (0–1), so it survives any change of render size."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if not (self.w > 0 and self.h > 0):
            raise ValueError(f"region must have a positive size, got {self.w}x{self.h}")
        if not (0 <= self.x and 0 <= self.y and self.x + self.w <= 1.0001 and self.y + self.h <= 1.0001):
            raise ValueError(f"region {self} falls outside the frame")

    def box(self, width: int, height: int, *, grow: int = 0) -> tuple[int, int, int, int]:
        """Pixel ``(x, y, w, h)`` in a ``width``x``height`` frame, optionally grown by
        ``grow`` px on every side and clamped to the frame."""
        x0 = max(0, round(self.x * width) - grow)
        y0 = max(0, round(self.y * height) - grow)
        x1 = min(width, round((self.x + self.w) * width) + grow)
        y1 = min(height, round((self.y + self.h) * height) + grow)
        return x0, y0, x1 - x0, y1 - y0


def lock_regions(
    video: str | Path,
    source: str | Path,
    regions: Sequence[Region],
    dest: str | Path,
    *,
    feather: int = 4,
    crf: int = 16,
    preset: str = "medium",
    timeout: float | None = None,
) -> Path:
    """Paste ``regions`` of ``source`` over every frame of ``video``.

    ``source`` is the clip's own keyframe — the image the model was given, where the label
    is still the real one. It is scaled to the clip's size, masked to the regions with a
    ``feather``-px falloff so the seam does not show, and overlaid on all frames in a single
    pass (one decode, a few hundred milliseconds for a 5 s clip). Audio is copied through,
    which matters for the models that generate their own.
    """
    video, source, dest = Path(video).expanduser(), Path(source).expanduser(), Path(dest)
    for p in (video, source):
        if not p.is_file():
            raise FileNotFoundError(f"not found: {p}")
    if not regions:
        raise ValueError("no regions to lock")
    info = probe(video)
    w, h = info.width, info.height
    if not (w and h):
        raise MediaError(f"cannot read the size of {video}")
    # The mask is built from the scaled source itself rather than a `color` source, so there is
    # no second stream whose duration has to be matched to the clip's.
    boxes = "".join(
        f",drawbox=x={x}:y={y}:w={bw}:h={bh}:color=white@1:t=fill"
        for x, y, bw, bh in (r.box(w, h, grow=feather) for r in regions)
    )
    graph = (
        f"[1:v]scale={w}:{h}:flags=lanczos,format=rgb24,split[patch][mask];"
        f"[mask]drawbox=x=0:y=0:w={w}:h={h}:color=black@1:t=fill{boxes},format=gray"
        + (f",gblur=sigma={feather / 2:g}" if feather else "")
        + "[m];"
        "[patch][m]alphamerge[ov];"
        "[0:v]format=rgb24[base];"
        "[base][ov]overlay=0:0:eof_action=repeat:format=auto,format=yuv420p[out]"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-i", str(source),
           "-filter_complex", graph, "-map", "[out]"]
    cmd += ["-map", "0:a?", "-c:a", "copy"] if info.has_audio else ["-an"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(part)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       timeout=timeout if timeout is not None else config.ASSEMBLE_TIMEOUT)
    except subprocess.CalledProcessError as e:
        part.unlink(missing_ok=True)
        tail = (e.stderr or b"").decode(errors="replace")[-2000:].strip()
        raise MediaError(f"ffmpeg could not lock the regions of {video.name}: {tail or e}") from e
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        part.unlink(missing_ok=True)
        raise MediaError(f"ffmpeg could not lock the regions of {video.name}: {e}") from e
    if not part.is_file() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise MediaError("ffmpeg produced no output")
    os.replace(part, dest)
    log.info("locked %d region(s) of %s onto %s -> %s", len(regions), source.name, video.name, dest)
    return dest
