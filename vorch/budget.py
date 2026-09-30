"""Per-job spending on hosted models: what a call will cost, and refusing to exceed a ceiling.

Every other failure in vorch costs GPU seconds. A hosted call costs money, and the evaluator's
accept/retry loop multiplies it — a scene rejected twice is billed three times. So the ceiling is
checked **before** the call, not reconciled after.

Estimates are all there is. The real charge comes back from api.comfy.org as a response header
that ComfyUI keeps in a process-local dict and emits only as a websocket progress line; it never
reaches ``/history``, which is vorch's only transport. So ``spend`` is what the workflows declared
they would cost (``registry.Cost``), never what was actually billed — and a workflow whose price
nobody declared is counted as unknown rather than as zero.

The job is reached through a contextvar rather than threaded through every step, the same way
``provenance`` stamps its manifests: the orchestrator sets it for a phase and ``steps.run_step``
consults it without knowing what a job is.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from . import registry

log = logging.getLogger("vorch.budget")

# Set by the orchestrator around a phase; read by run_step. Holds the live Job.
_job: contextvars.ContextVar[object | None] = contextvars.ContextVar("vorch_budget_job", default=None)


class BudgetExceeded(RuntimeError):
    """A hosted call would take the job past its ceiling; nothing was submitted."""


@dataclass(frozen=True)
class Spend:
    min_usd: float = 0.0
    max_usd: float = 0.0
    calls: int = 0
    unpriced: int = 0      # calls whose workflow declared no cost

    def describe(self) -> str:
        if not self.calls:
            return "nothing spent"
        amount = (f"${self.min_usd:.2f}" if abs(self.max_usd - self.min_usd) < 1e-9
                  else f"${self.min_usd:.2f}-${self.max_usd:.2f}")
        out = f"{amount} over {self.calls} call{'s' if self.calls != 1 else ''}"
        return out + (f" (+{self.unpriced} unpriced)" if self.unpriced else "")


def total(entries) -> Spend:
    """Sum a job's spend log."""
    lo = hi = 0.0
    calls = unpriced = 0
    for e in entries or []:
        calls += 1
        if e.get("max_usd") is None:
            unpriced += 1
            continue
        lo += float(e.get("min_usd") or 0.0)
        hi += float(e.get("max_usd") or 0.0)
    return Spend(round(lo, 6), round(hi, 6), calls, unpriced)


def spent(job) -> Spend:
    return total(getattr(job, "spend", None))


def remaining(job) -> float | None:
    """What is left of the ceiling, reserving the worst case. None when uncapped."""
    cap = getattr(job, "budget_usd", None)
    if cap is None:
        return None
    return round(float(cap) - spent(job).max_usd, 6)


def check(job, workflow: str, cost: "registry.Cost | None") -> None:
    """Raise BudgetExceeded if this call cannot be afforded. Unpriced calls never block.

    An unpriced hosted call is a hole in the accounting, not a reason to stop a run the user
    asked for — `registry validate --online` is where a missing `cost:` is reported.
    """
    left = remaining(job)
    if left is None or cost is None:
        return
    if cost.worst > left + 1e-9:
        raise BudgetExceeded(
            f"{workflow!r} would cost {cost.describe()} but only ${max(left, 0.0):.2f} of the "
            f"${float(job.budget_usd):.2f} budget is left ({spent(job).describe()} so far) — "
            "raise the budget to continue"
        )


def record(job, workflow: str, cost: "registry.Cost | None", *, scene=None, attempt=1) -> dict:
    """Append one billed call to the job's spend log and return the entry."""
    entry = {
        "workflow": workflow,
        "scene": scene,
        "attempt": attempt,
        "min_usd": cost.min_usd if cost else None,
        "max_usd": cost.max_usd if cost else None,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    getattr(job, "spend").append(entry)
    log.info("job %s: %s billed %s (%s)", getattr(job, "id", "?"), workflow,
             cost.describe() if cost else "unpriced", spent(job).describe())
    return entry


# --- the ambient job, for steps that must not know what a job is -----------------


def set_job(job) -> contextvars.Token:
    return _job.set(job)


def reset_job(token: contextvars.Token) -> None:
    _job.reset(token)


def current_job():
    return _job.get()


def account(workflow: str, cost: "registry.Cost | None", *, scene=None, attempt=1) -> None:
    """Check the ambient job's budget, then record the call. A no-op outside a job.

    Raises BudgetExceeded *before* recording, so a refused call is not billed.
    """
    job = current_job()
    if job is None:
        return
    check(job, workflow, cost)
    record(job, workflow, cost, scene=scene, attempt=attempt)
