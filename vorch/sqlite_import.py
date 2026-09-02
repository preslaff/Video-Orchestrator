"""One-off importer for the pre-PostgreSQL SQLite state files (``runs/*.db``).

Copies ``jobs`` and ``decisions`` rows into a :class:`~vorch.state.Store`,
preserving job ids and timestamps. Jobs that already exist in PostgreSQL are
skipped together with their decisions, so re-running is harmless. Decision ids
are *not* preserved (PostgreSQL assigns new ones) but insertion order follows
the old ids, so the log order is unchanged.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from psycopg.types.json import Jsonb

from .state import Store


@dataclass
class ImportReport:
    jobs: int = 0
    decisions: int = 0
    skipped_jobs: int = 0


def import_sqlite(path: str | Path, store: Store) -> ImportReport:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    report = ImportReport()
    try:
        jobs = src.execute(
            "SELECT id, status, idea, created_at, updated_at, data FROM jobs ORDER BY created_at"
        ).fetchall()
        with store._pool.connection() as conn:  # same package; raw rows need the pool
            for j in jobs:
                inserted = conn.execute(
                    """
                    INSERT INTO jobs (id, status, idea, created_at, updated_at, data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING id
                    """,
                    (
                        j["id"],
                        j["status"],
                        j["idea"],
                        datetime.fromisoformat(j["created_at"]),
                        datetime.fromisoformat(j["updated_at"]),
                        Jsonb(json.loads(j["data"])),
                    ),
                ).fetchone()
                if inserted is None:
                    report.skipped_jobs += 1
                    continue
                report.jobs += 1
                for d in src.execute(
                    "SELECT * FROM decisions WHERE job_id = ? ORDER BY id", (j["id"],)
                ).fetchall():
                    conn.execute(
                        """
                        INSERT INTO decisions
                            (job_id, scene_index, phase, attempt, action, score, reason, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            d["job_id"],
                            d["scene_index"],
                            d["phase"],
                            d["attempt"],
                            d["action"],
                            d["score"],
                            d["reason"],
                            datetime.fromisoformat(d["created_at"]),
                        ),
                    )
                    report.decisions += 1
    finally:
        src.close()
    return report
