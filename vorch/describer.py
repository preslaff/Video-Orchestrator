"""Background asset description — Gemma captions new assets and suggests their labels.

Uploads return at once; each asset is queued here and described one at a time with
``asset_vision.describe`` (CPU Gemma, 20–40 s each). The result is stored as the asset's
note with a suggested (``auto``) label, which the user reviews in the UI — the same loop as
``vorch project label-assets``, without the terminal.

- **Never competes with a job.** Before each asset the worker waits while any job run
  segment is in flight (``JobManager.busy``): a storyboard or evaluator call on the same
  CPU-pinned model would otherwise slow down.
- **Never overwrites a person.** The label is written only if the asset still exists and
  its label is not ``human`` — checked under the project edit lock at save time, so an
  edit made while Gemma was looking at the image wins.
- **In memory.** Per-asset state (queued / describing / failed + error) lives in this
  process and is reported on ``AssetOut``; a restart drops the queue, and the
  "describe unlabelled" endpoint picks those assets up again.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config
from . import projects as projects_mod

log = logging.getLogger("vorch.describer")

QUEUED, DESCRIBING, FAILED = "queued", "describing", "failed"


def _describe_with_gemma(path: str, kind: str, hint: str) -> Any:
    from . import asset_vision  # local: pulls in the Gemma client

    return asset_vision.describe(path, hint=hint, kind=kind)


class AssetDescriber:
    def __init__(
        self,
        store: Any,
        *,
        describe: Callable[[str, str, str], Any] | None = None,
        busy: Callable[[], bool] = lambda: False,
        auto: bool | None = None,
        poll_seconds: float = 3.0,
    ):
        """``describe(path, kind, hint)`` returns an ``AssetNote`` (injectable for tests);
        ``busy()`` is true while a job segment runs; ``auto`` enables describe-on-upload."""
        self.store = store
        self.auto = config.AUTO_DESCRIBE if auto is None else auto
        self._describe = describe or _describe_with_gemma
        self._busy = busy
        self._poll = poll_seconds
        self._state: dict[str, tuple[str, str]] = {}
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None

    # -- public -------------------------------------------------------------------

    def state(self, asset_id: str) -> tuple[str, str]:
        """(``queued`` | ``describing`` | ``failed`` | ``""``, error message)."""
        return self._state.get(asset_id, ("", ""))

    def enqueue(self, project_id: str, asset_ids: list[str]) -> int:
        """Queue assets for description (call from the event loop); returns how many were added.
        Assets already queued or being described are not added twice."""
        added = 0
        for aid in asset_ids:
            if self._state.get(aid, ("", ""))[0] in (QUEUED, DESCRIBING):
                continue
            self._ensure_worker()
            self._state[aid] = (QUEUED, "")
            self._queue.put_nowait((project_id, aid))
            added += 1
        return added

    async def join(self) -> None:
        """Wait until everything queued so far has been handled (tests, shutdown)."""
        if self._queue is not None:
            await self._queue.join()

    # -- worker -------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            project_id, aid = await self._queue.get()
            try:
                while self._busy():
                    await asyncio.sleep(self._poll)
                target = self._target(project_id, aid)
                if target is None:
                    self._state.pop(aid, None)
                    continue
                self._state[aid] = (DESCRIBING, "")
                try:
                    note = await loop.run_in_executor(None, self._describe, *target)
                    await loop.run_in_executor(None, self._store, project_id, aid, note)
                except Exception as e:  # one bad image or a Gemma outage must not stop the queue
                    log.warning("describe %s: %s", aid, e)
                    self._state[aid] = (FAILED, str(e))
                    continue
                self._state.pop(aid, None)
            finally:
                self._queue.task_done()

    def _target(self, project_id: str, aid: str) -> tuple[str, str, str] | None:
        """(path, kind, hint) to describe, or None when the asset is gone or labelled by a person."""
        project = projects_mod.load(self.store, project_id)
        asset = project.asset(aid) if project is not None else None
        if asset is None or asset.label_source == "human" or not Path(asset.path).is_file():
            return None
        return asset.path, asset.kind, project.describe_hint

    def _store(self, project_id: str, aid: str, note: Any) -> None:
        from . import asset_vision

        with projects_mod.editing(self.store, project_id) as project:
            asset = project.asset(aid) if project is not None else None
            if asset is None or asset.label_source == "human":
                return  # deleted, or a person labelled it while Gemma was looking
            label = asset_vision.label_for(note, kind=asset.kind, fallback=Path(asset.name).stem)
            projects_mod.set_asset_note(project, aid, note.model_dump(), label=label)
            projects_mod.save(self.store, project)
        log.info("described %s: %s", aid, label)
