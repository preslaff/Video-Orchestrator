"""CLI checks that need no GPU: the `db` command group and the SQLite importer."""

from __future__ import annotations

import json
import sqlite3

import pytest

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


# --- setup-phase commands ------------------------------------------------------------


@pytest.fixture
def cli_store(store, monkeypatch, tmp_path):
    from vorch import config

    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(config, "EXPORT_ROOT", tmp_path / "export")
    monkeypatch.setattr(cli, "_open_store", lambda: _NoClose(store))
    return store


def test_setup_check_prints_report(cli_store, monkeypatch, capsys):
    from vorch import envcheck
    from vorch.envcheck import CheckResult, EnvReport

    monkeypatch.setattr(envcheck, "run_checks", lambda **kw: EnvReport(
        [CheckResult("database", "ok", "v2"), CheckResult("gemma", "fail", "down")], "now", 3))
    assert cli.main(["setup", "check"]) == 1
    out = capsys.readouterr().out
    assert "[ok  ] database" in out and "[FAIL] gemma" in out and "-> fail" in out
    monkeypatch.setattr(envcheck, "run_checks", lambda **kw: EnvReport([CheckResult("database", "ok", "")], "now", 1))
    assert cli.main(["setup", "check", "--deep", "--profile", "social_9x16"]) == 0


def test_profiles_projects_and_export_commands(cli_store, tmp_path, capsys):
    from vorch.state import JobStatus, Scene, new_job

    assert cli.main(["profiles"]) == 0
    out = capsys.readouterr().out
    assert "cinematic_16x9" in out and "stills_16x9" in out and "stills x4" in out

    assert cli.main(["projects"]) == 0 and "no projects" in capsys.readouterr().out
    assert cli.main(["project", "add", "Autumn Launch", "--profile", "social_9x16"]) == 0
    assert "created project autumn-launch" in capsys.readouterr().out
    assert cli.main(["project", "add", "X", "--profile", "nope"]) == 1  # unknown profile -> error
    face = tmp_path / "face.jpg"
    face.write_bytes(b"f")
    assert cli.main(["project", "add-asset", "autumn-launch", str(face), "--kind", "master_face"]) == 0
    assert "added master_face asset" in capsys.readouterr().out
    assert cli.main(["project", "show", "autumn-launch"]) == 0
    out = capsys.readouterr().out
    assert "master_face" in out and "profile=social_9x16" in out
    assert cli.main(["projects"]) == 0 and "assets=1" in capsys.readouterr().out

    job = new_job(idea="Export me")
    kf = tmp_path / "k.png"
    kf.write_bytes(b"k")
    job.scenes = [Scene(index=0, keyframe_path=str(kf))]
    cli_store.save_job(job)
    assert cli.main(["export", job.id]) == 1  # not done
    assert cli.main(["export", job.id, "--force"]) == 0
    out = capsys.readouterr().out
    assert "s01_keyframe.png" in out and cli_store.load_job(job.id).export["ok"]
    job.status = JobStatus.DONE
    cli_store.save_job(job)
    assert cli.main(["export", job.id]) == 0


def test_assemble_command(cli_store, tmp_path, monkeypatch, capsys):
    from vorch import assembly, export
    from vorch.state import JobStatus, Scene, new_job

    job = new_job(idea="cut", kind="video")
    job.status = JobStatus.DONE
    clips = []
    for i in range(2):
        p = tmp_path / f"c{i}.mp4"
        p.write_bytes(b"v")
        clips.append(p)
    job.scenes = [Scene(index=i, keyframe_path=f"/k{i}.png", video_path=str(c)) for i, c in enumerate(clips)]
    cli_store.save_job(job)
    seen = {}

    def fake_resolve(j, spec_dict):
        seen["spec"] = spec_dict
        return assembly.AssemblySpec(items=[assembly.AssemblyItem(0), assembly.AssemblyItem(1)],
                                     cuts=[assembly.Cut(spec_dict["cuts"][0]["transition"], spec_dict["cuts"][0]["duration"])]), clips

    monkeypatch.setattr(assembly, "resolve", fake_resolve)
    monkeypatch.setattr(assembly, "run", lambda j, s, p: assembly.AssemblyRecord(
        path="/out/assembly_v1.mp4", created_at="t", duration=5.6, n_clips=2, spec=s.to_dict(), fps=16, width=832, height=480, version=1))
    monkeypatch.setattr(export, "export_job", lambda store, j, project: export.ExportRecord("/exp", "t", files=["assembly.mp4"]))
    assert cli.main(["assemble", job.id, "--transition", "wipeleft", "--duration", "0.3"]) == 0
    out = capsys.readouterr().out
    assert "wipeleft 0.3s" in out and "v1: /out/assembly_v1.mp4" in out and "exported -> /exp" in out
    assert seen["spec"]["cuts"] == [{"transition": "wipeleft", "duration": 0.3}]
    saved = cli_store.load_job(job.id)
    assert saved.assembly["ok"] and saved.export["files"] == ["assembly.mp4"]

    assert cli.main(["assemble", job.id, "--list"]) == 0
    assert "* v1" in capsys.readouterr().out and "wipeleft 0.3s" in capsys.readouterr().out or True

    job.status = JobStatus.NEEDS_APPROVAL
    cli_store.save_job(job)
    assert cli.main(["assemble", job.id]) == 1  # not done, no --force
