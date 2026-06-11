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
from .evaluator import EvaluationError, evaluate_image
from .media import MediaError
from .pipeline import Scene, run_scene, run_sequence
from .steps.animate import animate
from .steps.faceswap import faceswap
from .steps.keyframe import keyframe


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
    return 0 if verdict.verdict == "accept" else 2


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

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (comfy.ComfyError, EvaluationError, MediaError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
