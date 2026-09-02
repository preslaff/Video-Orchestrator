"""Smoke check for the NiceGUI dashboard wiring.

The UI itself is validated manually in a browser; here we only assert that
``setup()`` registers the API routes and the pages without error (NiceGUI is a
heavy import, so this is the one test that pulls it in).
"""

from __future__ import annotations

from vorch import app as dashboard
from vorch.state import Store


def test_setup_registers_api_and_pages(store):
    dashboard.setup(store=store)  # idempotent; registers once per process

    from nicegui import app as nicegui_app

    paths = {getattr(r, "path", None) for r in nicegui_app.routes}
    assert "/api/jobs" in paths            # server routes mounted
    assert "/job/{job_id}" in paths        # detail page registered
    assert dashboard._ctx.get("manager") is not None


def test_pending_gate_inference(store):
    from vorch.state import Decision, new_job

    job = new_job(idea="x")
    store.save_job(job)
    assert dashboard._pending_gate(job, store) == "storyboard"
    store.log_decision(Decision(job_id=job.id, scene_index=None, phase="storyboard",
                                attempt=1, action="approved"))
    assert dashboard._pending_gate(job, store) == "keyframes"
