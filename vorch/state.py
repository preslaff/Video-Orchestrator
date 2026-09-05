"""Job persistence + decision log (DESIGN.md §6).

A run is a **Job** (idea/style/master) holding a list of **Scene** records, each
tracking its generated prompts, artifact paths, and Gemma **Verdict**s through the
keyframe -> video -> swap chain. Two things are persisted to PostgreSQL:

- the whole Job, as a JSON blob in ``jobs`` — the **resumable state** (reload and
  continue from each scene's status / which artifacts already exist);
- an append-only ``decisions`` table — the **decision log**: one immutable row per
  accept/retry/approval, so "why was this frame retried?" is answerable after the
  fact (Georgi's debugging requirement).

psycopg 3, synchronous, behind a small thread-safe connection pool (see ``db.py``
for DSN/migrations). The asyncio ``queue.py`` calls these from an executor
thread and from the loop; nothing here assumes an event loop. GPU-free and
unit-testable against a throwaway schema (tests/conftest.py).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from . import config, db
from .evaluator import Verdict

log = logging.getLogger(__name__)


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
class Variant:
    """One generated candidate for a scene's keyframe (stills jobs make several)."""

    path: str
    seed: int | None = None
    verdict: Verdict | None = None
    swapped_path: str | None = None
    post_paths: list[str] = field(default_factory=list)


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
    keyframe_negative: str = ""       # per-scene negative (profile negative when empty)
    mode: str = "t2i"                 # t2i | edit (reference-locked keyframe)
    references: list[str] = field(default_factory=list)  # local paths for mode="edit"
    keyframe_path: str | None = None
    keyframe_verdict: Verdict | None = None
    animation_prompt: str = ""
    animation_negative: str = ""
    video_path: str | None = None
    video_verdict: Verdict | None = None
    swapped_path: str | None = None
    swap_verdict: Verdict | None = None
    status: SceneStatus = SceneStatus.PENDING
    variants: list[Variant] = field(default_factory=list)   # keyframe candidates; keyframe_path = chosen
    raw_video_path: str | None = None    # Wan output before post-steps (video_path = final)
    post_paths: list[str] = field(default_factory=list)      # post-step outputs in order
    regenerations: int = 0               # manual keyframe/clip redo count (seed offset)


@dataclass
class Job:
    id: str
    idea: str = ""
    style: str = ""
    master_face: str = ""
    autonomy: str = "semi"  # semi | full
    # Run-config — persisted so a resumed run (approve/reload) reconstructs the
    # orchestrator faithfully instead of falling back to defaults.
    n_scenes: int | None = None
    seed: int = 0
    max_attempts: int = 2
    evaluate: bool = True
    face_swap: bool = True
    mode: str = "t2i"                        # default keyframe mode for new scenes
    reference_images: list[str] = field(default_factory=list)
    project_id: str | None = None
    profile_id: str | None = None
    profile: dict | None = None            # frozen OutputProfile snapshot at submit
    kind: str = "video"                      # video | stills (copy of profile.kind)
    export: dict | None = None             # ExportRecord (handoff folder) once exported
    assembly: dict | None = None           # AssemblyRecord (clips joined with transitions)
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
    n_scenes: int | None = None,
    seed: int = 0,
    max_attempts: int = 2,
    evaluate: bool = True,
    face_swap: bool = True,
    mode: str = "t2i",
    reference_images: list[str] | None = None,
    project_id: str | None = None,
    profile_id: str | None = None,
    profile: dict | None = None,
    kind: str = "video",
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
        n_scenes=n_scenes,
        seed=seed,
        max_attempts=max_attempts,
        evaluate=evaluate,
        face_swap=face_swap,
        mode=mode,
        reference_images=list(reference_images or []),
        project_id=project_id,
        profile_id=profile_id,
        profile=profile,
        kind=kind,
        scenes=scenes or [],
        created_at=ts,
        updated_at=ts,
    )


# --- (De)serialization -------------------------------------------------------


def _verdict_to_dict(v: Verdict | None) -> dict | None:
    return v.model_dump() if v is not None else None


def _verdict_from_dict(d: dict | None) -> Verdict | None:
    return Verdict.model_validate(d) if d is not None else None


def _variant_to_dict(v: Variant) -> dict:
    return {
        "path": v.path,
        "seed": v.seed,
        "verdict": _verdict_to_dict(v.verdict),
        "swapped_path": v.swapped_path,
        "post_paths": list(v.post_paths),
    }


def _variant_from_dict(d: dict) -> Variant:
    return Variant(
        path=d["path"],
        seed=d.get("seed"),
        verdict=_verdict_from_dict(d.get("verdict")),
        swapped_path=d.get("swapped_path"),
        post_paths=list(d.get("post_paths", []) or []),
    )


def _scene_to_dict(s: Scene) -> dict:
    return {
        "index": s.index,
        "description": s.description,
        "character": s.character,
        "setting": s.setting,
        "mood": s.mood,
        "continuous": s.continuous,
        "keyframe_prompt": s.keyframe_prompt,
        "keyframe_negative": s.keyframe_negative,
        "mode": s.mode,
        "references": list(s.references),
        "keyframe_path": s.keyframe_path,
        "keyframe_verdict": _verdict_to_dict(s.keyframe_verdict),
        "animation_prompt": s.animation_prompt,
        "animation_negative": s.animation_negative,
        "video_path": s.video_path,
        "video_verdict": _verdict_to_dict(s.video_verdict),
        "swapped_path": s.swapped_path,
        "swap_verdict": _verdict_to_dict(s.swap_verdict),
        "status": s.status.value,
        "variants": [_variant_to_dict(v) for v in s.variants],
        "raw_video_path": s.raw_video_path,
        "post_paths": list(s.post_paths),
        "regenerations": s.regenerations,
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
        keyframe_negative=d.get("keyframe_negative", ""),
        mode=d.get("mode", "t2i"),
        references=list(d.get("references", []) or []),
        keyframe_path=d.get("keyframe_path"),
        keyframe_verdict=_verdict_from_dict(d.get("keyframe_verdict")),
        animation_prompt=d.get("animation_prompt", ""),
        animation_negative=d.get("animation_negative", ""),
        video_path=d.get("video_path"),
        video_verdict=_verdict_from_dict(d.get("video_verdict")),
        swapped_path=d.get("swapped_path"),
        swap_verdict=_verdict_from_dict(d.get("swap_verdict")),
        status=SceneStatus(d.get("status", "pending")),
        variants=[_variant_from_dict(v) for v in d.get("variants", []) or []],
        raw_video_path=d.get("raw_video_path"),
        post_paths=list(d.get("post_paths", []) or []),
        regenerations=int(d.get("regenerations", 0) or 0),
    )


def job_to_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "idea": job.idea,
        "style": job.style,
        "master_face": job.master_face,
        "autonomy": job.autonomy,
        "n_scenes": job.n_scenes,
        "seed": job.seed,
        "max_attempts": job.max_attempts,
        "evaluate": job.evaluate,
        "face_swap": job.face_swap,
        "mode": job.mode,
        "reference_images": list(job.reference_images),
        "project_id": job.project_id,
        "profile_id": job.profile_id,
        "profile": job.profile,
        "kind": job.kind,
        "export": job.export,
        "assembly": job.assembly,
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
        n_scenes=d.get("n_scenes"),
        seed=d.get("seed", 0),
        max_attempts=d.get("max_attempts", 2),
        evaluate=d.get("evaluate", True),
        face_swap=d.get("face_swap", True),
        mode=d.get("mode", "t2i"),
        reference_images=list(d.get("reference_images", []) or []),
        project_id=d.get("project_id"),
        profile_id=d.get("profile_id"),
        profile=d.get("profile"),
        kind=d.get("kind", "video"),
        export=d.get("export"),
        assembly=d.get("assembly"),
        status=JobStatus(d.get("status", "pending")),
        scenes=[_scene_from_dict(s) for s in d.get("scenes", [])],
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
    )


# --- Store -------------------------------------------------------------------

def _to_ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _from_ts(ts: datetime) -> str:
    # timestamptz comes back in the session time zone; normalise to UTC so the
    # strings round-trip exactly with what _now() produced.
    return ts.astimezone(timezone.utc).isoformat()


@dataclass
class JobSummary:
    """Lightweight row for listing jobs without loading the full blob."""

    id: str
    status: JobStatus
    idea: str
    updated_at: str
    project_id: str | None = None
    kind: str = "video"
    profile_id: str | None = None
    profile_name: str | None = None  # from the snapshot (user profiles have opaque ids)


class Store:
    """PostgreSQL-backed persistence for jobs and the decision log.

    ``dsn`` defaults to ``config.DATABASE_URL``. ``schema`` (optional) is put at
    the head of ``search_path`` on every pooled connection — production leaves it
    unset (``public``); tests pass a throwaway schema. Pending migrations are
    applied on construction. Use as a context manager or call ``close()``.

    The Store does **not** create the database itself; use :meth:`open` (or
    ``vorch db init``) for that.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        schema: str | None = None,
        pool_size: int = 4,
    ):
        self.dsn = dsn or config.DATABASE_URL
        self.schema = schema
        self._closed = False
        try:
            self._pool = ConnectionPool(
                self.dsn,
                min_size=1,
                max_size=pool_size,
                open=True,
                configure=self._configure,
                kwargs={"row_factory": dict_row},
            )
            self._pool.wait(timeout=10)
            with self._pool.connection() as conn:
                self.version = db.migrate(conn)
        except psycopg.OperationalError as e:
            raise db.DatabaseUnavailable(db._unavailable_msg(self.dsn, e)) from e

    @classmethod
    def open(cls, dsn: str | None = None) -> Store:
        """Create the database if missing, then return a migrated Store."""
        dsn = dsn or config.DATABASE_URL
        db.ensure_database(dsn)
        return cls(dsn)

    def _configure(self, conn: psycopg.Connection) -> None:
        if self.schema:
            conn.execute(
                sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema))
            )
            conn.commit()

    # -- jobs --

    def save_job(self, job: Job) -> Job:
        """Insert or update a job, stamping ``updated_at`` (and ``created_at``)."""
        now = _now()
        if not job.created_at:
            job.created_at = now
        job.updated_at = now
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, status, idea, created_at, updated_at, data, project_id, kind)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    idea = EXCLUDED.idea,
                    updated_at = EXCLUDED.updated_at,
                    data = EXCLUDED.data,
                    project_id = EXCLUDED.project_id,
                    kind = EXCLUDED.kind
                """,
                (
                    job.id,
                    job.status.value,
                    job.idea,
                    _to_ts(job.created_at),
                    _to_ts(job.updated_at),
                    Jsonb(job_to_dict(job)),
                    job.project_id,
                    job.kind,
                ),
            )
        return job

    def load_job(self, job_id: str) -> Job | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT data, project_id FROM jobs WHERE id = %s", (job_id,)).fetchone()
        if not row:
            return None
        job = job_from_dict(row["data"])
        job.project_id = row["project_id"]  # the column is authoritative (nulled on project delete)
        return job

    def list_jobs(
        self,
        *,
        limit: int | None = None,
        offset: int = 0,
        status: JobStatus | str | None = None,
        project_id: str | None = None,
    ) -> list[JobSummary]:
        """Newest first; optional status / project filters and limit/offset paging."""
        where, params = self._job_filter(status, project_id)
        q = (
            "SELECT id, status, idea, updated_at, project_id, kind, data->>'profile_id' AS profile_id, "
            "data->'profile'->>'name' AS profile_name "
            f"FROM jobs{where} ORDER BY updated_at DESC"
        )
        if limit is not None:
            q += " LIMIT %s OFFSET %s"
            params += [limit, offset]
        with self._pool.connection() as conn:
            rows = conn.execute(q, params).fetchall()
        return [
            JobSummary(
                id=r["id"],
                status=JobStatus(r["status"]),
                idea=r["idea"],
                updated_at=_from_ts(r["updated_at"]),
                project_id=r["project_id"],
                kind=r["kind"],
                profile_id=r["profile_id"],
                profile_name=r["profile_name"],
            )
            for r in rows
        ]

    def count_jobs(self, *, status: JobStatus | str | None = None, project_id: str | None = None) -> int:
        where, params = self._job_filter(status, project_id)
        with self._pool.connection() as conn:
            return conn.execute(f"SELECT count(*) AS n FROM jobs{where}", params).fetchone()["n"]

    @staticmethod
    def _job_filter(status: JobStatus | str | None, project_id: str | None = None) -> tuple[str, list]:
        clauses, params = [], []
        if status is not None:
            clauses.append("status = %s")
            params.append(status.value if isinstance(status, JobStatus) else str(status))
        if project_id is not None:
            clauses.append("project_id = %s")
            params.append(project_id)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def delete_job(self, job_id: str) -> None:
        """Delete a job and (via ON DELETE CASCADE) its decision log."""
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM jobs WHERE id = %s", (job_id,))

    # -- decision log --

    def log_decision(self, decision: Decision) -> Decision:
        """Append one entry to the decision log; returns it with ``id`` set."""
        if not decision.created_at:
            decision.created_at = _now()
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO decisions
                    (job_id, scene_index, phase, attempt, action, score, reason, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    decision.job_id,
                    decision.scene_index,
                    decision.phase,
                    decision.attempt,
                    decision.action,
                    decision.score,
                    decision.reason,
                    _to_ts(decision.created_at),
                ),
            ).fetchone()
        decision.id = row["id"]
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
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE job_id = %s ORDER BY id", (job_id,)
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
                created_at=_from_ts(r["created_at"]),
            )
            for r in rows
        ]

    # -- projects (dict rows; vorch.projects owns the model) --

    def save_project(self, d: dict) -> dict:
        from .projects import SlugTaken  # local: avoid import cycle

        with self._pool.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO projects (id, slug, name, created_at, updated_at, data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        slug = EXCLUDED.slug, name = EXCLUDED.name,
                        updated_at = EXCLUDED.updated_at, data = EXCLUDED.data
                    """,
                    (d["id"], d["slug"], d["name"], _to_ts(d["created_at"]), _to_ts(d["updated_at"]), Jsonb(d)),
                )
            except psycopg.errors.UniqueViolation as e:
                raise SlugTaken(f"project slug {d['slug']!r} is already used") from e
        return d

    def load_project(self, project_id: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT data FROM projects WHERE id = %s", (project_id,)).fetchone()
        return row["data"] if row else None

    def load_project_by_slug(self, slug: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT data FROM projects WHERE slug = %s", (slug,)).fetchone()
        return row["data"] if row else None

    def list_projects(self) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT data FROM projects ORDER BY updated_at DESC").fetchall()
        return [r["data"] for r in rows]

    def delete_project(self, project_id: str) -> bool:
        """Delete the project row; jobs keep their rows with project_id set to NULL."""
        with self._pool.connection() as conn:
            cur = conn.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            return cur.rowcount > 0

    # -- profiles (dict rows; vorch.profiles owns the model) --

    def save_profile(self, d: dict) -> dict:
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO profiles (id, name, kind, purpose, created_at, updated_at, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name, kind = EXCLUDED.kind, purpose = EXCLUDED.purpose,
                    updated_at = EXCLUDED.updated_at, data = EXCLUDED.data
                """,
                (d["id"], d["name"], d["kind"], d["purpose"],
                 _to_ts(d["created_at"]), _to_ts(d["updated_at"]), Jsonb(d)),
            )
        return d

    def load_profile(self, profile_id: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT data FROM profiles WHERE id = %s", (profile_id,)).fetchone()
        return row["data"] if row else None

    def list_profiles(self) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT data FROM profiles ORDER BY name").fetchall()
        return [r["data"] for r in rows]

    def delete_profile(self, profile_id: str) -> bool:
        with self._pool.connection() as conn:
            cur = conn.execute("DELETE FROM profiles WHERE id = %s", (profile_id,))
            return cur.rowcount > 0

    # -- lifecycle --

    def schema_version(self) -> int:
        with self._pool.connection() as conn:
            return max(db.applied_versions(conn), default=0)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pool.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
