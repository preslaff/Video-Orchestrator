"""NiceGUI dashboard (DESIGN §11) — job form, live gallery, HITL approval gates.

Runs in one process with the FastAPI API (server.register_routes) and the
JobManager, so pages call the manager directly and subscribe to its in-memory
event Broker for live refreshes — no polling, no separate client. Generated
media is served from ComfyUI's output dir via a static mount.

Entry points: ``setup()`` registers pages/routes on the global NiceGUI app once;
``run()`` starts the server (used by ``vorch dashboard``).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from nicegui import app as nicegui_app
from nicegui import ui

from .. import config, provenance
from ..orchestrator import Orchestrator
from ..queue import JobManager
from ..server import register_routes
from ..state import JobStatus, Store

log = logging.getLogger("vorch.app")

GATE_AFTER = {  # which gate clears the current NEEDS_APPROVAL pause
    "storyboard": "storyboard",
    "keyframes": "keyframes",
}

_ctx: dict = {}  # holds the Store + JobManager once setup() runs


def _store() -> Store:
    return _ctx["store"]


def _manager() -> JobManager:
    return _ctx["manager"]


def _pending_gate(job, store) -> str:
    """Which gate a paused job is waiting on (inferred from approvals so far)."""
    return Orchestrator.next_gate(job, store) or "keyframes"


def _outputs_url(path: str | None, *, video: bool = False) -> str | None:
    if not path:
        return None
    name = Path(path).name
    return f"/outputs/video/{name}" if video else f"/outputs/{name}"


# --- pages -------------------------------------------------------------------


def _index_page() -> None:
    store, manager = _store(), _manager()
    ui.label("Video Orchestrator").classes("text-2xl font-bold")

    with ui.card().classes("w-full max-w-2xl"):
        ui.label("New job").classes("text-lg")
        idea = ui.textarea("Idea").classes("w-full")
        with ui.row().classes("w-full"):
            style = ui.input("Style", value="oil painting, impasto, visible brushstrokes").classes("flex-grow")
            master = ui.input("Master face", value="assets/jena-cropped.jpeg").classes("flex-grow")
        with ui.row():
            n_scenes = ui.number("Scenes", value=2, min=1, max=20, format="%d")
            seed = ui.number("Seed (0=random)", value=0, format="%d")
            autonomy = ui.select(["semi", "full"], value="semi", label="Autonomy")
            evaluate = ui.switch("Evaluate", value=True)
            face_swap = ui.switch("Face swap", value=True)
        master.bind_visibility_from(face_swap, "value")  # master only matters when swapping

        async def submit() -> None:
            if not idea.value:
                ui.notify("Idea is required", type="warning")
                return
            try:
                job = await manager.submit(
                    idea.value,
                    style=style.value,
                    master_face=master.value,
                    autonomy=autonomy.value,
                    n_scenes=int(n_scenes.value) if n_scenes.value else None,
                    seed=int(seed.value or 0),
                    evaluate=evaluate.value,
                    face_swap=face_swap.value,
                )
            except FileNotFoundError as e:
                ui.notify(str(e), type="negative")
                return
            ui.navigate.to(f"/job/{job.id}")

        ui.button("Create & run", on_click=submit).props("color=primary")

    ui.separator()
    ui.label("Jobs").classes("text-lg")

    @ui.refreshable
    def job_list() -> None:
        rows = store.list_jobs()
        if not rows:
            ui.label("No jobs yet").classes("text-gray-500")
        for r in rows:
            with ui.row().classes("items-center gap-3"):
                ui.badge(r.status.value).props("outline")
                ui.link(r.idea or r.id, f"/job/{r.id}")

    job_list()
    ui.timer(3.0, job_list.refresh)  # cheap list poll; detail pages use the broker


def _job_page(job_id: str) -> None:
    store, manager = _store(), _manager()

    @ui.refreshable
    def view() -> None:
        job = store.load_job(job_id)
        if job is None:
            ui.label("No such job")
            return

        with ui.row().classes("items-center gap-3"):
            ui.link("← jobs", "/")
            ui.badge(job.status.value).props("outline")
        ui.label(job.idea).classes("text-xl font-bold")

        editing = job.status is JobStatus.NEEDS_APPROVAL
        edits: dict[int, dict] = {}  # populated in the scene loop below

        if editing:
            gate = _pending_gate(job, store)

            def save() -> None:
                payload = [
                    {"index": idx, **{name: ta.value for name, ta in flds.items()}}
                    for idx, flds in edits.items()
                ]
                try:
                    manager.update_scenes(job_id, payload)
                except (RuntimeError, KeyError) as e:
                    ui.notify(str(e), type="negative")
                    return
                ui.notify("Saved", type="positive")
                view.refresh()

            async def approve(g=gate) -> None:
                await manager.approve(job_id, g)
                view.refresh()

            with ui.row().classes("items-center gap-2"):
                ui.button("Save changes", on_click=save).props("outline")
                ui.button(f"Approve gate: {gate}", on_click=approve).props("color=positive")

        with ui.row().classes("flex-wrap gap-4"):
            for s in job.scenes:
                with ui.card().classes("w-96"):
                    ui.label(f"scene {s.index} · {s.status.value}").classes("font-semibold")
                    if s.setting or s.mood:
                        ui.label(f"{s.setting} · {s.mood}").classes("text-xs text-gray-500")

                    kf = _outputs_url(s.keyframe_path)
                    if kf:
                        ui.image(kf).classes("w-full")
                    vid = _outputs_url(s.video_path, video=True)
                    if vid:
                        ui.video(vid).classes("w-full")

                    if editing:
                        flds: dict[str, object] = {}
                        flds["description"] = ui.textarea("description", value=s.description) \
                            .classes("w-full").props("autogrow dense")
                        ta_kf = ui.textarea("keyframe prompt", value=s.keyframe_prompt) \
                            .classes("w-full").props("autogrow dense")
                        if s.keyframe_path is None:  # keyframe not made yet -> editable
                            flds["keyframe_prompt"] = ta_kf
                        else:
                            ta_kf.props("readonly")  # already generated; edit has no effect
                        flds["animation_prompt"] = ui.textarea("animation prompt", value=s.animation_prompt) \
                            .classes("w-full").props("autogrow dense")
                        edits[s.index] = flds
                    else:
                        if s.description:
                            ui.label(s.description).classes("text-sm")
                        ui.label(f"keyframe: {s.keyframe_prompt}").classes("text-xs text-gray-600")
                        ui.label(f"animation: {s.animation_prompt}").classes("text-xs text-gray-600")

                    m = provenance.read_manifest(s.keyframe_path) if s.keyframe_path else None
                    if m:
                        seed = m.get("params", {}).get("seed")
                        ui.label(f"seed={seed}").classes("text-xs text-gray-500")
                    if s.keyframe_verdict is not None:
                        v = s.keyframe_verdict
                        ui.label(f"verdict: {v.decision} (score {v.score})").classes("text-xs")

        with ui.expansion("decision log", icon="history").classes("w-full"):
            for d in store.decisions(job_id):
                who = "job" if d.scene_index is None else f"s{d.scene_index}"
                score = "" if d.score is None else f" score={d.score}"
                ui.label(f"{who} · {d.phase}/{d.action}{score} {d.reason}").classes("text-xs")

    view()

    # Live refresh: subscribe to the job's event stream while it's active.
    job = store.load_job(job_id)
    if job is not None and job.status not in (JobStatus.DONE, JobStatus.FAILED):
        q = manager.events(job_id)

        async def listen() -> None:
            try:
                while True:
                    ev = await q.get()
                    view.refresh()
                    if ev.get("type") == "end":
                        break
            finally:
                manager.broker.unsubscribe(job_id, q)

        asyncio.create_task(listen())


# --- setup / run -------------------------------------------------------------


def setup(store: Store | None = None) -> None:
    """Wire the Store, JobManager, API routes, static media mount, and pages."""
    if _ctx:
        return  # idempotent — register once per process
    if store is None:
        store = Store.open()
    _ctx["store"] = store
    _ctx["manager"] = JobManager(store)
    register_routes(nicegui_app, _ctx["manager"], store)
    if config.COMFY_OUTPUT.exists():
        nicegui_app.add_static_files("/outputs", str(config.COMFY_OUTPUT))
    else:
        log.warning("COMFY_OUTPUT %s missing; media won't render", config.COMFY_OUTPUT)

    ui.page("/")(_index_page)
    ui.page("/job/{job_id}")(_job_page)


def run(host: str = "127.0.0.1", port: int = 8090, **kw) -> None:
    """Start the dashboard server (blocking)."""
    setup()
    ui.run(host=host, port=port, reload=False, show=False, title="Video Orchestrator", **kw)
