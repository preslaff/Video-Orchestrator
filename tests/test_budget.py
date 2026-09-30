"""Per-job spending on hosted models.

A hosted call costs money and the evaluator's retry loop multiplies it, so the ceiling is
checked before the call rather than reconciled after. Estimates are all there is: the real
charge never reaches /history (see vorch/budget.py), so the log records what the workflows
declared they would cost.
"""

from __future__ import annotations

import pytest

from vorch import budget, registry
from vorch.state import job_to_dict, job_from_dict, new_job


def _cost(lo, hi=None):
    return registry.parse_cost("w", {"usd": lo} if hi is None else {"min_usd": lo, "max_usd": hi})


def _job(cap=None):
    j = new_job(idea="x")
    j.budget_usd = cap
    return j


# --- parsing -------------------------------------------------------------------

def test_cost_accepts_a_number_a_flat_usd_and_a_range():
    assert _cost(0.04) == registry.Cost(0.04, 0.04)
    assert registry.parse_cost("w", 0.04) == registry.Cost(0.04, 0.04)
    assert _cost(0.03, 0.06) == registry.Cost(0.03, 0.06)
    assert registry.parse_cost("w", None) is None


def test_a_range_reserves_its_upper_bound():
    assert _cost(0.03, 0.06).worst == 0.06


@pytest.mark.parametrize("raw,msg", [
    ({"usd": -1}, ">= 0"),
    ({"usd": 0.1, "min_usd": 0.2}, "either usd or"),
    ({}, "needs usd"),
    ({"min_usd": 0.5, "max_usd": 0.1}, "below"),
    ("nope", "number or a mapping"),
    ({"usd": "free"}, "must be a number"),
])
def test_bad_cost_blocks_are_rejected(raw, msg):
    with pytest.raises(registry.RegistryError, match=msg):
        registry.parse_cost("w", raw)


def test_describe_reads_naturally():
    assert _cost(0.04).describe() == "$0.04"
    assert _cost(0.03, 0.06).describe() == "$0.03-$0.06"


# --- accounting ----------------------------------------------------------------

def test_spend_accumulates_and_survives_a_round_trip():
    j = _job(1.0)
    budget.record(j, "wf", _cost(0.04), scene=0, attempt=1)
    budget.record(j, "wf", _cost(0.03, 0.06), scene=1, attempt=1)
    s = budget.spent(j)
    assert s.calls == 2 and s.min_usd == pytest.approx(0.07) and s.max_usd == pytest.approx(0.10)
    assert budget.remaining(j) == pytest.approx(0.90)   # reserves the worst case
    assert budget.spent(job_from_dict(job_to_dict(j))) == s


def test_every_retry_is_billed_because_every_retry_is_a_call():
    j = _job(1.0)
    for attempt in (1, 2, 3):
        budget.record(j, "wf", _cost(0.04), scene=0, attempt=attempt)
    assert budget.spent(j).max_usd == pytest.approx(0.12)
    assert [e["attempt"] for e in j.spend] == [1, 2, 3]


def test_an_unpriced_call_is_counted_as_unknown_not_as_free():
    j = _job(1.0)
    budget.record(j, "wf", None)
    s = budget.spent(j)
    assert s.calls == 1 and s.unpriced == 1 and s.max_usd == 0.0
    assert "unpriced" in s.describe()


def test_the_ceiling_refuses_the_call_that_would_cross_it():
    j = _job(0.10)
    for _ in range(2):
        budget.check(j, "wf", _cost(0.04))
        budget.record(j, "wf", _cost(0.04))
    with pytest.raises(budget.BudgetExceeded, match="only \\$0.02"):
        budget.check(j, "wf", _cost(0.04))
    assert len(j.spend) == 2, "a refused call must not be billed"


def test_an_uncapped_job_and_an_unpriced_workflow_never_block():
    budget.check(_job(None), "wf", _cost(99.0))
    budget.check(_job(0.01), "wf", None)


def test_account_is_a_no_op_outside_a_run():
    budget.account("wf", _cost(0.04))          # no ambient job: nothing to bill


def test_account_bills_the_ambient_job_and_refuses_before_recording():
    j = _job(0.05)
    token = budget.set_job(j)
    try:
        budget.account("wf", _cost(0.04), scene=2, attempt=1)
        assert j.spend[0]["scene"] == 2
        with pytest.raises(budget.BudgetExceeded):
            budget.account("wf", _cost(0.04))
        assert len(j.spend) == 1
    finally:
        budget.reset_job(token)


# --- the gate ------------------------------------------------------------------

def test_a_budget_stop_is_what_pending_gate_reports(make_store):
    """The check for a known trap: `gates_for` is a fixed itinerary, so a stop that happens
    mid-run would otherwise be reported as whichever static gate came next."""
    from vorch.orchestrator import GATE_BUDGET, GATE_STORYBOARD, Orchestrator
    from vorch.state import Decision, JobStatus

    store = make_store()
    try:
        from vorch import profiles
        prof = profiles.apply_overrides(profiles.default_profile(),
                                        {"face_swap": {"enabled": False}})
        o = Orchestrator(store, profile=prof)
        job = o.create_job("idea")
        store.log_decision(Decision(job_id=job.id, scene_index=None, phase=GATE_STORYBOARD,
                                    attempt=1, action="approved"))
        assert Orchestrator.next_gate(job, store) == GATE_KEYFRAMES_NAME

        o._budget_stop(job, RuntimeError("would cost $0.04 but only $0.01 is left"))
        assert job.status is JobStatus.NEEDS_APPROVAL
        assert Orchestrator.budget_blocked(job, store)
        assert Orchestrator.next_gate(job, store) == GATE_BUDGET
        assert Orchestrator.pending_gate(job, store) == GATE_BUDGET

        # approving clears the stop; the run re-checks and will stop again if still over
        store.log_decision(Decision(job_id=job.id, scene_index=None, phase=GATE_BUDGET,
                                    attempt=1, action="approved"))
        assert Orchestrator.budget_blocked(job, store) is None
        assert Orchestrator.next_gate(job, store) == GATE_KEYFRAMES_NAME
    finally:
        store.close()


GATE_KEYFRAMES_NAME = "keyframes"


# --- the API surface ------------------------------------------------------------


@pytest.fixture
def client(tmp_path, make_store, monkeypatch):
    """A server whose orchestrator does nothing: these tests are about the budget, not a run."""
    from fastapi.testclient import TestClient

    from vorch import config
    from vorch.api import create_app
    from vorch.queue import JobManager

    class _NoOpOrch:
        def __init__(self, store, job):
            self.store, self.params = store, None

        def run(self, job):
            return job

    store = make_store()
    manager = JobManager(store, orchestrator_factory=lambda s, j: _NoOpOrch(s, j), auto_export=False)
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(config, "COMFY_OUTPUT", media)
    with TestClient(create_app(store=store, manager=manager, media_dir=media)) as c:
        c.store = store
        yield c
    store.close()


def test_a_job_can_be_submitted_with_a_ceiling_and_reports_its_spend(client):
    r = client.post("/api/jobs", json={"idea": "a bridge", "budget_usd": 0.10,
                                       "face_swap": False, "force": True})
    assert r.status_code in (200, 201), r.text
    job = r.json()
    assert job["budget_usd"] == 0.10
    assert job["spend_usd"] == 0.0 and job["spend_calls"] == 0 and job["spend"] == []
    assert job["budget_reason"] == ""


def test_the_route_raises_and_lifts_the_ceiling(client):
    """A job paused on its budget needs this before its gate is approved: approving alone
    would stop it again at the very next billed call."""
    jid = client.post("/api/jobs", json={"idea": "x", "budget_usd": 0.10,
                                         "face_swap": False, "force": True}).json()["id"]
    assert client.put(f"/api/jobs/{jid}/budget", json={"budget_usd": 5.0}).json()["budget_usd"] == 5.0
    assert client.put(f"/api/jobs/{jid}/budget", json={"budget_usd": None}).json()["budget_usd"] is None
    assert client.put(f"/api/jobs/{jid}/budget", json={"budget_usd": -1}).status_code == 422
    assert client.put("/api/jobs/nope/budget", json={"budget_usd": 1.0}).status_code == 404


def test_the_ceiling_cannot_be_set_below_what_is_already_spent(client):
    jid = client.post("/api/jobs", json={"idea": "x", "budget_usd": 1.0,
                                         "face_swap": False, "force": True}).json()["id"]
    job = client.store.load_job(jid)
    budget.record(job, "wf", _cost(0.40))
    client.store.save_job(job)
    bad = client.put(f"/api/jobs/{jid}/budget", json={"budget_usd": 0.20})
    assert bad.status_code == 400 and "already spent" in bad.text
    shown = client.get(f"/api/jobs/{jid}").json()
    assert shown["spend_usd"] == pytest.approx(0.40) and shown["spend_calls"] == 1
