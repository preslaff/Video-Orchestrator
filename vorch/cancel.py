"""Cooperative stop for a running job segment.

A run segment executes the blocking orchestrator in a worker thread, and a thread
cannot be killed — cancelling the asyncio task would let the thread keep saving the
job behind a client's back. Instead ``JobManager`` binds a ``threading.Event`` for the
segment (``call_bound``) and the pipeline checks it at safe points: between units of
work that are already persisted, and inside the ComfyUI poll/retry loops, where a stop
also removes the prompt from ComfyUI's queue and interrupts it.

The event lives in a ContextVar set *inside* the worker thread (``run_in_executor``
does not copy context), the same pattern as ``provenance.set_context``. Outside a
bound segment (CLI runs, tests) ``requested()`` is always False.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("vorch_stop", default=None)


class JobStopped(Exception):
    """Raised at a safe point when the user asked to stop the job.

    An ``Exception`` (not ``BaseException``) so it unwinds through ``finally`` blocks
    normally; ``JobManager`` catches it *before* its generic failure handler.
    """


def requested() -> bool:
    """True when the segment running in this thread has been asked to stop."""
    ev = _event.get()
    return ev is not None and ev.is_set()


def check() -> None:
    """Raise ``JobStopped`` if a stop was requested — call only where nothing unsaved is lost."""
    if requested():
        raise JobStopped("stopped by the user")


def call_bound(event: threading.Event, fn: Callable[..., T], *args: Any) -> T:
    """Run ``fn(*args)`` with ``event`` as this thread's stop signal."""
    token = _event.set(event)
    try:
        return fn(*args)
    finally:
        _event.reset(token)
