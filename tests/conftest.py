"""Shared fixtures: a PostgreSQL-backed Store isolated per test.

Every test that needs persistence gets its own throwaway schema in the test
database (``VORCH_TEST_DATABASE_URL``, default ``postgresql:///vorch_test``),
created before the test and dropped after it. If PostgreSQL is unreachable the
session stops with a clear message; set ``VORCH_TEST_SKIP_DB=1`` to skip the
DB-backed tests instead.
"""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from psycopg import sql

from vorch import backends, config, db, registry
from vorch.state import Store

TEST_DSN = os.environ.get("VORCH_TEST_DATABASE_URL", "postgresql:///vorch_test")


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    if os.environ.get("VORCH_TEST_SKIP_DB") == "1":
        pytest.skip("VORCH_TEST_SKIP_DB=1: skipping PostgreSQL-backed tests")
    try:
        db.ensure_database(TEST_DSN)
    except db.DatabaseUnavailable as e:
        pytest.exit(
            f"PostgreSQL is required for the test suite.\n{e}\n"
            "Start it, or set VORCH_TEST_DATABASE_URL, or VORCH_TEST_SKIP_DB=1.",
            returncode=3,
        )
    return TEST_DSN


@pytest.fixture
def pg_schema(pg_dsn: str):
    name = f"t_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
    try:
        yield name
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))


@pytest.fixture
def make_store(pg_dsn: str, pg_schema: str):
    """Factory: each call opens a fresh Store on the same per-test schema
    (for 'reopen' tests). All instances are closed at teardown."""
    made: list[Store] = []

    def _make() -> Store:
        s = Store(pg_dsn, schema=pg_schema)
        made.append(s)
        return s

    try:
        yield _make
    finally:
        for s in made:
            s.close()


@pytest.fixture
def store(make_store) -> Store:
    return make_store()


@pytest.fixture(autouse=True)
def _isolated_user_workflows(tmp_path, monkeypatch):
    """Never read the developer's ~/vorch/workflows (or backends.yaml) in tests: a
    user entry there would change which workflows/backends exist. Each test gets
    an empty dir and no backends file."""
    user_dir = tmp_path / "_isolated_user_workflows"
    user_dir.mkdir()
    monkeypatch.setattr(config, "USER_WORKFLOW_DIR", user_dir)
    # Same for ~/vorch/backends.yaml: tests see only the local backend unless they
    # point config.BACKENDS_FILE at their own file.
    monkeypatch.setattr(config, "BACKENDS_FILE", tmp_path / "_isolated_backends.yaml")
    registry.reset_cache()
    backends.reset_cache()
    yield
    registry.reset_cache()
    backends.reset_cache()
