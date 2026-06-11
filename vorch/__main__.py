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

from . import config
from .clients import comfy
from .clients.gemma import GemmaError
from .evaluator import EvaluationError, evaluate_image
from .media import MediaError
from .orchestrator import GATE_KEYFRAMES, GATE_STORYBOARD, Orchestrator
from .pipeline import Scene, run_scene, run_sequence
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
    result = run_scene(
        args.keyframe_prompt,
        args.animation_prompt,
        args.master,
        seed=args.seed,
        evaluate=args.evaluate,
        task=args.task,
        max_attempts=args.max_attempts,
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
    scenes = _load_scenes(args.scenes)
    results = run_sequence(
        scenes,
        args.master,
        seed=args.seed,
        evaluate=args.evaluate,
        relock_identity=not args.no_relock,
        max_attempts=args.max_attempts,
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
    config.STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    return Store(config.STATE_DB)


def _print_job(job, store: Store) -> None:
    print(f"job {job.id}  [{job.status.value}]  {job.idea!r}")
    for s in job.scenes:
        kf = s.keyframe_path or "-"
        vid = s.video_path or "-"
        print(f"  scene {s.index} [{s.status.value}]  kf={kf}  video={vid}")
    if job.status is JobStatus.NEEDS_APPROVAL:
        done = {d.phase for d in store.decisions(job.id) if d.action == "approved"}
        gate = GATE_KEYFRAMES if GATE_STORYBOARD in done else GATE_STORYBOARD
        print(f"  -> paused. approve with:  python -m vorch approve {job.id} {gate}")


def _orchestrator(store: Store, spec: dict) -> Orchestrator:
    retry = spec.get("retry") or {}
    return Orchestrator(
        store,
        n_scenes=spec.get("n_scenes"),
        seed=int(spec.get("seed", config.SEED_RANDOM)),
        max_attempts=int(retry.get("max_attempts", 2)),
        evaluate=bool(spec.get("evaluate", True)),
    )


def _cmd_run(args) -> int:
    import yaml  # local import: pyyaml only needed for job specs

    with open(args.spec, encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    if not spec.get("idea"):
        raise ValueError(f"{args.spec}: 'idea' is required")
    if not spec.get("master_face"):
        raise ValueError(f"{args.spec}: 'master_face' is required")

    store = _open_store()
    try:
        o = _orchestrator(store, spec)
        job = o.create_job(
            spec["idea"],
            style=spec.get("style", ""),
            master_face=spec["master_face"],
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
    rs.add_argument("--master", required=True, help="master face image path")
    rs.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    rs.add_argument("--evaluate", action="store_true", help="judge + retry the keyframe")
    rs.add_argument("--task", default=None, help="eval intent (defaults to keyframe prompt)")
    rs.add_argument("--max-attempts", type=int, default=2)
    rs.set_defaults(func=_cmd_run_scene)

    rq = sub.add_parser("run-sequence", help="chain scenes with last-frame continuity")
    rq.add_argument("scenes", help="YAML file: list of {keyframe_prompt, animation_prompt, continuous?}")
    rq.add_argument("--master", required=True, help="master face image path")
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

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        comfy.ComfyError, EvaluationError, MediaError, GemmaError, StoryboardError,
        FileNotFoundError, ValueError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
