"""PostgreSQL plumbing: DSN, database bootstrap, versioned SQL migrations.

Kept separate from ``state.py`` so the Store stays about jobs/decisions and this
module stays about *connections*. Everything is synchronous psycopg 3; the pool
is thread-safe, which is what lets the orchestrator (executor thread) and the
API (event loop) share one Store.

Migrations are plain SQL files in ``vorch/migrations/NNNN_name.sql``, applied in
order under an advisory lock and recorded in ``schema_migrations``. Tables are
created unqualified so they land in the first ``search_path`` schema — the
production default (``public``) or a per-test schema.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import tuple_row

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_RE = re.compile(r"^(\d{4})_[A-Za-z0-9_]+\.sql$")
# Arbitrary constant; all vorch processes agree on it so only one migrates at a time.
_LOCK_KEY = 0x766F7263  # "vorc"


class DatabaseUnavailable(RuntimeError):
    """PostgreSQL could not be reached (service down, wrong DSN, no peer auth)."""


def dbname_of(dsn: str) -> str:
    return str(conninfo_to_dict(dsn).get("dbname") or "")


def ensure_database(dsn: str) -> bool:
    """Create the database named in ``dsn`` if it does not exist.

    Connects to the maintenance database ``postgres`` with the same host/user.
    Returns True when the database was created, False when it already existed.
    Raises :class:`DatabaseUnavailable` when the server cannot be reached.
    """
    name = dbname_of(dsn)
    if not name:
        raise ValueError(f"DSN has no dbname: {dsn!r}")
    admin_dsn = make_conninfo(dsn, dbname="postgres")
    try:
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
            ).fetchone()
            if exists:
                return False
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            log.info("created database %s", name)
            return True
    except psycopg.OperationalError as e:
        raise DatabaseUnavailable(_unavailable_msg(dsn, e)) from e


def available_migrations() -> list[tuple[int, Path]]:
    """``[(version, path)]`` sorted by version, from ``MIGRATIONS_DIR``."""
    out: list[tuple[int, Path]] = []
    for p in sorted(MIGRATIONS_DIR.glob("*.sql")):
        m = _MIGRATION_RE.match(p.name)
        if not m:
            raise ValueError(f"bad migration filename: {p.name} (want NNNN_name.sql)")
        out.append((int(m.group(1)), p))
    versions = [v for v, _ in out]
    if len(set(versions)) != len(versions):
        raise ValueError(f"duplicate migration versions: {versions}")
    return out


def latest_version() -> int:
    migs = available_migrations()
    return migs[-1][0] if migs else 0


def applied_versions(conn: psycopg.Connection) -> set[int]:
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'schema_migrations' "
            "AND table_schema = current_schema()"
        )
        if cur.fetchone() is None:
            return set()
        cur.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def migrate(conn: psycopg.Connection) -> int:
    """Apply every pending migration in one transaction; return the current version.

    Idempotent: re-running with nothing pending is a cheap read.
    """
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = applied_versions(conn)
        current = max(applied, default=0)
        for version, path in available_migrations():
            if version in applied:
                continue
            log.info("applying migration %04d (%s)", version, path.name)
            conn.execute(path.read_text())  # multi-statement: no params -> simple protocol
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            current = version
    return current


def _unavailable_msg(dsn: str, exc: Exception) -> str:
    return (
        f"cannot connect to PostgreSQL ({dsn}): {exc}\n"
        "Is the service running (`pg_lsclusters`)? Does the OS user have a role "
        "(peer auth) or does VORCH_DATABASE_URL carry credentials?"
    )
