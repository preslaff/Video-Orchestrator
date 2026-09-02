"""CLI checks that need no GPU: the `db` command group and the SQLite importer."""

from __future__ import annotations

import json
import sqlite3

from vorch import __main__ as cli
from vorch.state import Store, new_job, job_to_dict

_OLD_SCHEMA = """
CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, idea TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
    scene_index INTEGER, phase TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 1,
    action TEXT NOT NULL, score INTEGER, reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
"""


def _legacy_db(path, n_jobs=2):
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    ids = []
    for i in range(n_jobs):
        job = new_job(idea=f"old idea {i}", style="oil")
        job.status = job.status.__class__("done")
        conn.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?)",
            (job.id, "done", job.idea, job.created_at, job.updated_at,
             json.dumps(job_to_dict(job), ensure_ascii=False)),
        )
        for k in range(3):
            conn.execute(
                "INSERT INTO decisions (job_id, scene_index, phase, attempt, action, score, reason, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job.id, 0, "keyframe", k + 1, "retry" if k < 2 else "accept", 5 + k, f"r{k}", job.created_at),
            )
        ids.append(job.id)
    conn.commit()
    conn.close()
    return ids


def test_db_import_sqlite_copies_jobs_and_decisions(tmp_path, store):
    from vorch.sqlite_import import import_sqlite

    legacy = tmp_path / "old.db"
    ids = _legacy_db(legacy)

    r = import_sqlite(legacy, store)
    assert (r.jobs, r.decisions, r.skipped_jobs) == (2, 6, 0)

    loaded = store.load_job(ids[0])
    assert loaded is not None and loaded.idea == "old idea 0" and loaded.status.value == "done"
    log = store.decisions(ids[0])
    assert [d.action for d in log] == ["retry", "retry", "accept"]
    assert [d.attempt for d in log] == [1, 2, 3]

    # Idempotent: a second run skips everything already present.
    r2 = import_sqlite(legacy, store)
    assert (r2.jobs, r2.decisions, r2.skipped_jobs) == (0, 0, 2)
    assert len(store.decisions(ids[0])) == 3


def test_db_status_and_import_commands(tmp_path, store, monkeypatch, capsys):
    # Route the CLI at the per-test schema instead of the real database.
    monkeypatch.setattr(cli, "_open_store", lambda: _NoClose(store))
    legacy = tmp_path / "old.db"
    _legacy_db(legacy, n_jobs=1)

    assert cli.main(["db", "import-sqlite", str(legacy)]) == 0
    assert "imported 1 jobs, 3 decisions" in capsys.readouterr().out

    assert cli.main(["db", "status"]) == 0
    from vorch import db

    out = capsys.readouterr().out
    assert f"schema version: {db.latest_version()}" in out and "jobs: 1" in out


class _NoClose:
    """Proxy so the CLI's store.close() doesn't tear down the fixture's pool."""

    def __init__(self, inner: Store):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def close(self) -> None:
        pass
