"""vorch CLI — run pipeline steps individually or a full single scene.

Individual step commands isolate each ComfyUI endpoint for testing; ``run-scene``
chains them. Image arguments are local paths and are staged into ComfyUI input/
automatically.

    python -m vorch keyframe "oil painting, woman kneading bread, impasto"
    python -m vorch animate keyframe.png "slow camera push-in, hands kneading"
    python -m vorch faceswap scene.png assets/jena-cropped.jpeg
    python -m vorch run-scene --keyframe-prompt "..." --animation-prompt "..." \
        --master assets/jena-cropped.jpeg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config, provenance
from .clients import comfy
from .clients.gemma import GemmaError
from .evaluator import EvaluationError, evaluate_image
from .media import MediaError
from .orchestrator import GATE_KEYFRAMES, GATE_STORYBOARD, Orchestrator
from .pipeline import Scene, run_scene, run_sequence
from .db import DatabaseUnavailable
from .state import JobStatus, Store
from .steps.animate import animate
from .steps.faceswap import faceswap
from .steps.keyframe import keyframe
from .storyboard import StoryboardError

GATES = {GATE_STORYBOARD, GATE_KEYFRAMES}


def _cmd_keyframe(args) -> int:
    path = keyframe(
        args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        unet=args.unet,
    )
    print(path)
    return 0


def _cmd_animate(args) -> int:
    name = comfy.upload_image(args.image)
    path = animate(name, args.prompt, length=args.length, fps=args.fps, seed=args.seed)
    print(path)
    return 0


def _cmd_faceswap(args) -> int:
    input_name = comfy.upload_image(args.input_image)
    source_name = comfy.upload_image(args.source_image)
    path = faceswap(input_name, source_name, restore_model=args.restore_model)
    print(path)
    return 0


def _cmd_evaluate(args) -> int:
    verdict = evaluate_image(args.image, args.task)
    print(verdict.model_dump_json(indent=2))
    print(f"decision: {verdict.decision}", file=sys.stderr)
    if verdict.needs_review:
        print(f"flag: accepted but score={verdict.score} — review", file=sys.stderr)
    return 0 if verdict.decision == "accept" else 2


def _cmd_run_scene(args) -> int:
    if not args.no_face_swap and not args.master:
        raise ValueError("--master is required unless --no-face-swap is given")
    result = run_scene(
        args.keyframe_prompt,
        args.animation_prompt,
        args.master or "",
        seed=args.seed,
        evaluate=args.evaluate,
        task=args.task,
        max_attempts=args.max_attempts,
        face_swap=not args.no_face_swap,
    )
    print(f"keyframe: {result.keyframe}")
    print(f"swapped:  {result.swapped}")
    print(f"video:    {result.video}")
    if result.keyframe_verdict is not None:
        v = result.keyframe_verdict
        print(f"verdict:  {v.verdict} (score={v.score})")
    return 0


def _load_scenes(path: str) -> list[Scene]:
    """Parse a YAML scenes file into Scene objects.

    Expected shape: a list of mappings, each with ``keyframe_prompt`` and
    ``animation_prompt`` (required) and optional ``continuous`` (bool).
    """
    import yaml  # local import: pyyaml is only needed for run-sequence

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list of scenes")
    scenes: list[Scene] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: scene {i} is not a mapping")
        try:
            scenes.append(
                Scene(
                    keyframe_prompt=item["keyframe_prompt"],
                    animation_prompt=item["animation_prompt"],
                    continuous=bool(item.get("continuous", False)),
                )
            )
        except KeyError as e:
            raise ValueError(f"{path}: scene {i} missing required field {e}") from e
    return scenes


def _cmd_run_sequence(args) -> int:
    if not args.no_face_swap and not args.master:
        raise ValueError("--master is required unless --no-face-swap is given")
    scenes = _load_scenes(args.scenes)
    results = run_sequence(
        scenes,
        args.master or "",
        seed=args.seed,
        evaluate=args.evaluate,
        relock_identity=not args.no_relock,
        max_attempts=args.max_attempts,
        face_swap=not args.no_face_swap,
    )
    for i, r in enumerate(results):
        tag = "continuous" if r.continued_from_previous else "fresh"
        print(f"scene {i + 1} [{tag}]")
        print(f"  start:  {r.start_image}")
        print(f"  video:  {r.video}")
        if r.keyframe_verdict is not None:
            v = r.keyframe_verdict
            print(f"  verdict: {v.verdict} (score={v.score})")
    return 0


def _open_store() -> Store:
    return Store.open()


def _print_job(job, store: Store) -> None:
    print(f"job {job.id}  [{job.status.value}]  {job.idea!r}")
    for s in job.scenes:
        kf = s.keyframe_path or "-"
        vid = s.video_path or "-"
        print(f"  scene {s.index} [{s.status.value}]  kf={kf}  video={vid}")
    if job.status is JobStatus.NEEDS_APPROVAL:
        gate = Orchestrator.next_gate(job, store)
        print(f"  -> paused. approve with:  python -m vorch approve {job.id} {gate}")


def _print_provenance(job) -> None:
    """Show the recorded seed + prompt for each scene's media (from sidecars)."""
    rows = []
    for s in job.scenes:
        for label, path in (("keyframe", s.keyframe_path), ("video", s.video_path)):
            if not path:
                continue
            m = provenance.read_manifest(path)
            name = Path(path).name
            if not m:
                rows.append((s.index, label, None, None, name))
                continue
            p = m.get("params", {})
            seed = p.get("seed", p.get("seed_high"))
            rows.append((s.index, label, seed, p.get("positive", ""), name))
    if not rows:
        return
    print("provenance (seed + prompt per output):")
    for idx, label, seed, prompt, name in rows:
        if seed is None and prompt is None:
            print(f"  s{idx} {label:8} {name}  (no manifest)")
            continue
        print(f"  s{idx} {label:8} seed={seed}  {name}")
        if prompt:
            print(f"           prompt: {prompt}")


def _orchestrator(store: Store, spec: dict) -> Orchestrator:
    retry = spec.get("retry") or {}
    return Orchestrator(
        store,
        n_scenes=spec.get("n_scenes"),
        seed=int(spec.get("seed", config.SEED_RANDOM)),
        max_attempts=int(retry.get("max_attempts", 2)),
        evaluate=bool(spec.get("evaluate", True)),
        face_swap=bool(spec.get("face_swap", True)),
    )


def _cmd_run(args) -> int:
    import yaml  # local import: pyyaml only needed for job specs

    with open(args.spec, encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    if not spec.get("idea"):
        raise ValueError(f"{args.spec}: 'idea' is required")
    if spec.get("face_swap", True) and not spec.get("master_face"):
        raise ValueError(f"{args.spec}: 'master_face' is required when face_swap is on")

    store = _open_store()
    try:
        o = _orchestrator(store, spec)
        job = o.create_job(
            spec["idea"],
            style=spec.get("style", ""),
            master_face=spec.get("master_face", ""),
            autonomy=spec.get("autonomy", "semi"),
        )
        o.run(job)
        _print_job(job, store)
    finally:
        store.close()
    return 0


def _cmd_approve(args) -> int:
    if args.gate not in GATES:
        raise ValueError(f"unknown gate {args.gate!r}; expected one of {sorted(GATES)}")
    store = _open_store()
    try:
        job = store.load_job(args.job_id)
        if job is None:
            raise ValueError(f"no job with id {args.job_id}")
        o = Orchestrator.from_job(store, job)  # run-config restored from the job
        o.approve(job, args.gate)
        _print_job(job, store)
    finally:
        store.close()
    return 0


def _cmd_jobs(args) -> int:
    store = _open_store()
    try:
        rows = store.list_jobs()
        if not rows:
            print("no jobs")
        for r in rows:
            print(f"{r.id}  [{r.status.value}]  {r.updated_at}  {r.idea!r}")
    finally:
        store.close()
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    from .server import create_app

    app = create_app(_open_store())
    print(f"serving on http://{args.host}:{args.port}  (API under /api)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_dashboard(args) -> int:
    from .app import run  # lazy: NiceGUI only needed for the dashboard

    print(f"dashboard on http://{args.host}:{args.port}")
    run(host=args.host, port=args.port)
    return 0


def _cmd_db(args) -> int:
    from . import db

    if args.db_command == "init":
        created = db.ensure_database(config.DATABASE_URL)
        store = Store(config.DATABASE_URL)
        try:
            print(f"database {db.dbname_of(config.DATABASE_URL)} "
                  f"{'created' if created else 'exists'}; schema version {store.version}")
        finally:
            store.close()
        return 0
    if args.db_command == "status":
        store = _open_store()
        try:
            print(f"dsn: {config.DATABASE_URL}")
            print(f"schema version: {store.schema_version()} (latest {db.latest_version()})")
            print(f"jobs: {len(store.list_jobs())}")
        finally:
            store.close()
        return 0
    if args.db_command == "import-sqlite":
        from .sqlite_import import import_sqlite

        store = _open_store()
        try:
            r = import_sqlite(args.path, store)
        finally:
            store.close()
        print(f"imported {r.jobs} jobs, {r.decisions} decisions "
              f"({r.skipped_jobs} jobs already present, skipped)")
        return 0
    return 2


def _cmd_vram(args) -> int:
    from . import vram

    s = vram.snapshot()
    if s is None:
        print("GPU: unavailable (no nvidia-smi)")
    else:
        pct = 100 * s["free"] / s["total"] if s["total"] else 0
        print(f"GPU: used={s['used']} free={s['free']} total={s['total']} MB ({pct:.0f}% free)")
    if args.free:
        after = vram.reclaim()
        print(f"reclaimed — free now: {after if after is not None else 'unknown'} MB")
    return 0


def _cmd_job(args) -> int:
    store = _open_store()
    try:
        job = store.load_job(args.job_id)
        if job is None:
            raise ValueError(f"no job with id {args.job_id}")
        _print_job(job, store)
        print("decision log:")
        for d in store.decisions(job.id):
            scene = "job" if d.scene_index is None else f"s{d.scene_index}"
            score = "" if d.score is None else f" score={d.score}"
            reason = f"  {d.reason}" if d.reason else ""
            print(f"  [{d.created_at}] {scene} {d.phase}/{d.action}{score}{reason}")
        _print_provenance(job)
    finally:
        store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vorch", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    kf = sub.add_parser("keyframe", help="generate a keyframe from a prompt")
    kf.add_argument("prompt")
    kf.add_argument("--width", type=int, default=config.KEYFRAME_WIDTH)
    kf.add_argument("--height", type=int, default=config.KEYFRAME_HEIGHT)
    kf.add_argument("--steps", type=int, default=config.KEYFRAME_STEPS)
    kf.add_argument("--cfg", type=float, default=config.KEYFRAME_CFG)
    kf.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    kf.add_argument("--unet", default=None, help="override the diffusion model name")
    kf.set_defaults(func=_cmd_keyframe)

    an = sub.add_parser("animate", help="animate a keyframe into a video")
    an.add_argument("image", help="local path to the start image")
    an.add_argument("prompt", help="animation/motion prompt")
    an.add_argument("--length", type=int, default=config.WAN_LENGTH, help="frames (4n+1)")
    an.add_argument("--fps", type=int, default=config.WAN_FPS)
    an.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    an.set_defaults(func=_cmd_animate)

    fs = sub.add_parser("faceswap", help="swap a source face onto an image")
    fs.add_argument("input_image", help="local path to the scene/frame")
    fs.add_argument("source_image", help="local path to the master face")
    fs.add_argument("--restore-model", default=config.REACTOR_RESTORE_MODEL)
    fs.set_defaults(func=_cmd_faceswap)

    ev = sub.add_parser("evaluate", help="score an image against a task (Gemma vision)")
    ev.add_argument("image", help="local path to the image to evaluate")
    ev.add_argument("task", help="the intent to judge the image against")
    ev.set_defaults(func=_cmd_evaluate)

    rs = sub.add_parser("run-scene", help="full chain: keyframe -> swap -> animate")
    rs.add_argument("--keyframe-prompt", required=True)
    rs.add_argument("--animation-prompt", required=True)
    rs.add_argument("--master", default=None, help="master face image path (omit with --no-face-swap)")
    rs.add_argument("--no-face-swap", action="store_true", help="animate the keyframe without a face swap")
    rs.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    rs.add_argument("--evaluate", action="store_true", help="judge + retry the keyframe")
    rs.add_argument("--task", default=None, help="eval intent (defaults to keyframe prompt)")
    rs.add_argument("--max-attempts", type=int, default=2)
    rs.set_defaults(func=_cmd_run_scene)

    rq = sub.add_parser("run-sequence", help="chain scenes with last-frame continuity")
    rq.add_argument("scenes", help="YAML file: list of {keyframe_prompt, animation_prompt, continuous?}")
    rq.add_argument("--master", default=None, help="master face image path (omit with --no-face-swap)")
    rq.add_argument("--no-face-swap", action="store_true", help="animate keyframes/frames without a face swap")
    rq.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    rq.add_argument("--evaluate", action="store_true", help="judge + retry fresh keyframes")
    rq.add_argument("--no-relock", action="store_true", help="don't re-swap master onto carried frames")
    rq.add_argument("--max-attempts", type=int, default=2)
    rq.set_defaults(func=_cmd_run_sequence)

    rn = sub.add_parser("run", help="drive a full job from a YAML spec (storyboard -> clips)")
    rn.add_argument("spec", help="job spec YAML (idea, master_face, n_scenes, style, autonomy)")
    rn.set_defaults(func=_cmd_run)

    ap = sub.add_parser("approve", help="clear a semi-auto gate and continue a job")
    ap.add_argument("job_id")
    ap.add_argument("gate", help=f"one of {sorted(GATES)}")
    ap.set_defaults(func=_cmd_approve)

    js = sub.add_parser("jobs", help="list persisted jobs")
    js.set_defaults(func=_cmd_jobs)

    jb = sub.add_parser("job", help="show a job's scenes and decision log")
    jb.add_argument("job_id")
    jb.set_defaults(func=_cmd_job)

    vr = sub.add_parser("vram", help="show GPU memory; --free to reclaim ComfyUI VRAM")
    vr.add_argument("--free", action="store_true", help="POST /free and report new headroom")
    vr.set_defaults(func=_cmd_vram)

    sv = sub.add_parser("serve", help="run the HTTP API + SSE server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_cmd_serve)

    dbp = sub.add_parser("db", help="database: init | status | import-sqlite <file>")
    dbs = dbp.add_subparsers(dest="db_command", required=True)
    dbs.add_parser("init", help="create the database if missing and apply migrations")
    dbs.add_parser("status", help="show DSN, schema version and job count")
    imp = dbs.add_parser("import-sqlite", help="copy jobs + decisions from a legacy runs/*.db")
    imp.add_argument("path", help="path to the SQLite file (e.g. runs/vorch.db)")
    dbp.set_defaults(func=_cmd_db)

    db = sub.add_parser("dashboard", help="run the NiceGUI web dashboard")
    db.add_argument("--host", default="127.0.0.1")
    db.add_argument("--port", type=int, default=8090)
    db.set_defaults(func=_cmd_dashboard)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        comfy.ComfyError, EvaluationError, MediaError, GemmaError, StoryboardError,
        FileNotFoundError, ValueError, DatabaseUnavailable,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
