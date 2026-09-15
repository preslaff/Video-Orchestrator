"""The cooperative stop token: bound per worker thread, checked at safe points."""

from __future__ import annotations

import threading

import pytest

from vorch import cancel


def test_unbound_is_never_stopped():
    assert cancel.requested() is False
    cancel.check()  # no-op outside a bound segment


def test_call_bound_scopes_the_event_to_the_call_in_that_thread():
    ev = threading.Event()
    seen = {}

    def work():
        seen["before"] = cancel.requested()
        ev.set()
        seen["after"] = cancel.requested()
        with pytest.raises(cancel.JobStopped):
            cancel.check()
        return "ok"

    result = {}
    t = threading.Thread(target=lambda: result.update(r=cancel.call_bound(ev, work)))
    t.start()
    t.join()
    assert result["r"] == "ok" and seen == {"before": False, "after": True}
    assert cancel.requested() is False  # never leaked into this thread


def test_call_bound_resets_even_when_the_call_raises():
    ev = threading.Event()
    ev.set()
    with pytest.raises(cancel.JobStopped):
        cancel.call_bound(ev, cancel.check)
    assert cancel.requested() is False
