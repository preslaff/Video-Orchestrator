"""Job persistence + decision log (DESIGN.md §6).

A run is a **Job** (idea/style/master) holding a list of **Scene** records, each
tracking its generated prompts, artifact paths, and Gemma **Verdict**s through the
keyframe -> video -> swap chain. Two things are persisted to one SQLite file:

- the whole Job, as a JSON blob in ``jobs`` — the **resumable state** (reload and
  continue from each scene's status / which artifacts already exist);
- an append-only ``decisions`` table — the **decision log**: one immutable row per
  accept/retry/approval, so "why was this frame retried?" is answerable after the
  fact (Georgi's debugging requirement).

Stdlib ``sqlite3``, synchronous. The later asyncio ``queue.py`` wraps these calls
in an executor; nothing here assumes an event loop. GPU-free and unit-testable.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .evaluator import Verdict


def _now() -> str:
    """UTC timestamp, ISO 8601 — used for created_at/updated_at and the log."""
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    NEEDS_APPROVAL = "needs_approval"  # paused at a semi-auto human gate
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SceneStatus(str, Enum):
    PENDING = "pending"
    KEYFRAME = "keyframe"  # keyframe generated/accepted, awaiting animation
    VIDEO = "video"  # animated, awaiting swap
    SWAP = "swap"  # face swapped, awaiting final eval
    COMPLETE = "complete"
    FAILED = "failed"


# --- Data model (DESIGN.md §6) -----------------------------------------------


@dataclass
class Scene:
    """One scene's full record — creative brief + generated artifacts + verdicts.

    Distinct from ``pipeline.Scene`` (the execution *input*: prompts + continuous
    flag). This is the *persisted* scene the orchestrator advances and the UI reads.
    """

    index: int
    description: str = ""
    character: str = ""
    setting: str = ""
    mood: str = ""
    continuous: bool = False
    keyframe_prompt: str = ""
    keyframe_path: str | None = None
    keyframe_verdict: Verdict | None = None
    animation_prompt: str = ""
    video_path: str | None = None
    video_verdict: Verdict | None = None
    swapped_path: str | None = None
    swap_verdict: Verdict | None = None
    status: SceneStatus = SceneStatus.PENDING


@dataclass
class Job:
    id: str
    idea: str = ""
    style: str = ""
    master_face: str = ""
    autonomy: str = "semi"  # semi | full
    status: JobStatus = JobStatus.PENDING
    scenes: list[Scene] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Decision:
    """One immutable entry in the decision log."""

    job_id: str
    scene_index: int | None  # None for job-level decisions (e.g. storyboard approval)
    phase: str  # "storyboard" | "keyframe" | "video" | "swap"
    attempt: int
    action: str  # "accept" | "retry" | "approved" | "rejected" | "failed" | ...
    score: int | None = None
    reason: str = ""
    created_at: str = ""
    id: int | None = None  # set by the store on insert


def new_job(
    *,
    idea: str = "",
    style: str = "",
    master_face: str = "",
    autonomy: str = "semi",
    scenes: list[Scene] | None = None,
) -> Job:
    """Create a fresh Job with a generated id and creation timestamp."""
    ts = _now()
    return Job(
        id=uuid.uuid4().hex,
        idea=idea,
        style=style,
        master_face=master_face,
        autonomy=autonomy,
        scenes=scenes or [],
        created_at=ts,
        updated_at=ts,
    )


# --- (De)serialization -------------------------------------------------------


def _verdict_to_dict(v: Verdict | None) -> dict | None:
    return v.model_dump() if v is not None else None


def _verdict_from_dict(d: dict | None) -> Verdict | None:
    return Verdict.model_validate(d) if d is not None else None


def _scene_to_dict(s: Scene) -> dict:
    return {
        "index": s.index,
        "description": s.description,
        "character": s.character,
        "setting": s.setting,
        "mood": s.mood,
        "continuous": s.continuous,
        "keyframe_prompt": s.keyframe_prompt,
        "keyframe_path": s.keyframe_path,
        "keyframe_verdict": _verdict_to_dict(s.keyframe_verdict),
        "animation_prompt": s.animation_prompt,
        "video_path": s.video_path,
        "video_verdict": _verdict_to_dict(s.video_verdict),
        "swapped_path": s.swapped_path,
        "swap_verdict": _verdict_to_dict(s.swap_verdict),
        "status": s.status.value,
    }


def _scene_from_dict(d: dict) -> Scene:
    return Scene(
        index=d["index"],
        description=d.get("description", ""),
        character=d.get("character", ""),
        setting=d.get("setting", ""),
        mood=d.get("mood", ""),
        continuous=d.get("continuous", False),
        keyframe_prompt=d.get("keyframe_prompt", ""),
        keyframe_path=d.get("keyframe_path"),
        keyframe_verdict=_verdict_from_dict(d.get("keyframe_verdict")),
        animation_prompt=d.get("animation_prompt", ""),
        video_path=d.get("video_path"),
        video_verdict=_verdict_from_dict(d.get("video_verdict")),
        swapped_path=d.get("swapped_path"),
        swap_verdict=_verdict_from_dict(d.get("swap_verdict")),
        status=SceneStatus(d.get("status", "pending")),
    )


def job_to_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "idea": job.idea,
        "style": job.style,
        "master_face": job.master_face,
        "autonomy": job.autonomy,
        "status": job.status.value,
        "scenes": [_scene_to_dict(s) for s in job.scenes],
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def job_from_dict(d: dict) -> Job:
    return Job(
        id=d["id"],
        idea=d.get("idea", ""),
        style=d.get("style", ""),
        master_face=d.get("master_face", ""),
        autonomy=d.get("autonomy", "semi"),
        status=JobStatus(d.get("status", "pending")),
        scenes=[_scene_from_dict(s) for s in d.get("scenes", [])],
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
    )


# --- Store -------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    idea        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    data        TEXT NOT NULL            -- full Job as JSON (resumable state)
);
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    scene_index INTEGER,
    phase       TEXT NOT NULL,
    attempt     INTEGER NOT NULL DEFAULT 1,
    action      TEXT NOT NULL,
    score       INTEGER,
    reason      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_decisions_job ON decisions(job_id, id);
"""


@dataclass
class JobSummary:
    """Lightweight row for listing jobs without loading the full blob."""

    id: str
    status: JobStatus
    idea: str
    updated_at: str


class Store:
    """SQLite-backed persistence for jobs and the decision log.

    Use as a context manager (closes the connection on exit) or call ``close()``.
    ``path`` may be a file path or ``":memory:"`` (the latter lives only as long
    as this Store instance).
    """

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        # check_same_thread=False so the future asyncio queue can call from an
        # executor thread; access is otherwise serialized by the caller.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- jobs --

    def save_job(self, job: Job) -> Job:
        """Insert or update a job, stamping ``updated_at`` (and ``created_at``)."""
        now = _now()
        if not job.created_at:
            job.created_at = now
        job.updated_at = now
        self._conn.execute(
            """
            INSERT INTO jobs (id, status, idea, created_at, updated_at, data)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                idea=excluded.idea,
                updated_at=excluded.updated_at,
                data=excluded.data
            """,
            (
                job.id,
                job.status.value,
                job.idea,
                job.created_at,
                job.updated_at,
                json.dumps(job_to_dict(job), ensure_ascii=False),
            ),
        )
        self._conn.commit()
        return job

    def load_job(self, job_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT data FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return job_from_dict(json.loads(row["data"])) if row else None

    def list_jobs(self) -> list[JobSummary]:
        rows = self._conn.execute(
            "SELECT id, status, idea, updated_at FROM jobs ORDER BY updated_at DESC"
        ).fetchall()
        return [
            JobSummary(
                id=r["id"],
                status=JobStatus(r["status"]),
                idea=r["idea"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def delete_job(self, job_id: str) -> None:
        """Delete a job and (via ON DELETE CASCADE) its decision log."""
        self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._conn.commit()

    # -- decision log --

    def log_decision(self, decision: Decision) -> Decision:
        """Append one entry to the decision log; returns it with ``id`` set."""
        if not decision.created_at:
            decision.created_at = _now()
        cur = self._conn.execute(
            """
            INSERT INTO decisions
                (job_id, scene_index, phase, attempt, action, score, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.job_id,
                decision.scene_index,
                decision.phase,
                decision.attempt,
                decision.action,
                decision.score,
                decision.reason,
                decision.created_at,
            ),
        )
        self._conn.commit()
        decision.id = cur.lastrowid
        return decision

    def record_verdict(
        self,
        job_id: str,
        scene_index: int | None,
        phase: str,
        verdict: Verdict,
        *,
        attempt: int = 1,
    ) -> Decision:
        """Log a verdict as a decision — the authoritative ``decision``, not the
        raw model ``verdict`` (which can disagree; see evaluator.Verdict)."""
        return self.log_decision(
            Decision(
                job_id=job_id,
                scene_index=scene_index,
                phase=phase,
                attempt=attempt,
                action=verdict.decision,
                score=verdict.score,
                reason=verdict.fix_suggestion,
            )
        )

    def decisions(self, job_id: str) -> list[Decision]:
        """The decision log for a job, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM decisions WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
        return [
            Decision(
                id=r["id"],
                job_id=r["job_id"],
                scene_index=r["scene_index"],
                phase=r["phase"],
                attempt=r["attempt"],
                action=r["action"],
                score=r["score"],
                reason=r["reason"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # -- lifecycle --

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
