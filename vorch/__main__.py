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
from .pipeline import run_scene
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


def _cmd_run_scene(args) -> int:
    result = run_scene(
        args.keyframe_prompt,
        args.animation_prompt,
        args.master,
        seed=args.seed,
    )
    print(f"keyframe: {result.keyframe}")
    print(f"swapped:  {result.swapped}")
    print(f"video:    {result.video}")
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

    rs = sub.add_parser("run-scene", help="full chain: keyframe -> swap -> animate")
    rs.add_argument("--keyframe-prompt", required=True)
    rs.add_argument("--animation-prompt", required=True)
    rs.add_argument("--master", required=True, help="master face image path")
    rs.add_argument("--seed", type=int, default=config.SEED_RANDOM)
    rs.set_defaults(func=_cmd_run_scene)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (comfy.ComfyError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
