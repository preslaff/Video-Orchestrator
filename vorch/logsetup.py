"""One place that configures logging, for the CLI and for ``vorch serve`` alike.

Before this module there was a single ``logging.basicConfig`` inside ``__main__.main``, which
meant anything that did *not* enter through the CLI — ``uvicorn vorch.api:app``, an embedded
``create_app``, a notebook — got no configuration at all and fell back to the root logger's
``lastResort`` handler at WARNING. Every INFO line the orchestrator writes about what it decided
was simply dropped.

Two deliberate choices:

**Handlers go on the ``"vorch"`` logger, not the root.** Every module in the package already uses
``logging.getLogger("vorch.<module>")``, so one handler catches all of them without hijacking
uvicorn's, psycopg's or httpx's loggers — which is what ``basicConfig`` on the root does.

**The stderr format is unchanged.** Terminal habits (and the odd test that reads captured output)
survive; the *file* is the one that gains timestamps and a pid, because that is the copy you read
days later while working out which run did what.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from . import config

LEVELS = ("debug", "info", "warning", "error")

#: The logger every vorch module hangs off (``vorch.orchestrator``, ``vorch.tts``, …).
ROOT = "vorch"

STDERR_FORMAT = "%(levelname)s %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(process)d] %(message)s"

_configured = False


def resolve_level(level: str | None = None) -> int:
    """``level`` -> ``VORCH_LOG_LEVEL`` -> INFO, as a logging constant."""
    name = (level or os.environ.get("VORCH_LOG_LEVEL") or "info").strip().lower()
    return getattr(logging, name.upper(), logging.INFO) if name in LEVELS else logging.INFO


def resolve_log_file(log_file: str | Path | None = None) -> Path | None:
    """Where the rotating file handler writes, or ``None`` for stderr only.

    An explicitly empty ``VORCH_LOG_FILE`` (or ``log_file=""``) means *no file* — that is how the
    test suite keeps ``pytest -q`` from writing into ``~/vorch/logs``.
    """
    if log_file is not None:
        return Path(log_file).expanduser() if str(log_file) else None
    env = os.environ.get("VORCH_LOG_FILE")
    if env is not None:
        return Path(env).expanduser() if env else None
    return config.LOG_DIR / "vorch.log"


def configure(
    level: str | None = None, *, log_file: str | Path | None = None,
    stderr: bool = True, force: bool = False,
) -> Path | None:
    """Install vorch's handlers. Idempotent — a second call is a no-op unless ``force``.

    Returns the log file in use, or ``None`` when logging only to stderr. A file that cannot be
    opened (read-only home, bad path) is reported on stderr and skipped: logging must never be the
    reason a run fails.
    """
    global _configured
    if _configured and not force:
        return _current_file()

    log = logging.getLogger(ROOT)
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    log.setLevel(resolve_level(level))
    # Propagation is left ON deliberately. uvicorn configures its own loggers rather than the root,
    # so there is nothing up there to double-print to — and pytest's ``caplog`` captures by adding a
    # handler to the root, so switching it off would silently blind every test that asserts on a log
    # line. A host application that configures the root itself will see vorch's lines twice; that is
    # the conventional price and cheaper than breaking the tests.

    if stderr:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(STDERR_FORMAT))
        log.addHandler(sh)

    path = resolve_log_file(log_file)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                path, maxBytes=config.LOG_MAX_BYTES, backupCount=config.LOG_BACKUPS,
                encoding="utf-8", delay=True,
            )
            fh.setFormatter(logging.Formatter(FILE_FORMAT))
            log.addHandler(fh)
        except OSError as e:
            print(f"warning: cannot write the log file {path} ({e}); logging to stderr only",
                  file=sys.stderr)
            path = None

    _configured = True
    return path


def _current_file() -> Path | None:
    for h in logging.getLogger(ROOT).handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            return Path(h.baseFilename)
    return None


def reset() -> None:
    """Drop the handlers and the idempotency flag — for tests."""
    global _configured
    log = logging.getLogger(ROOT)
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    _configured = False
