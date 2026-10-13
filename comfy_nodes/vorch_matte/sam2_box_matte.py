"""SAM 2 video matte from one rectangle — the node vorch needs and Impact Pack does not have.

Impact Pack ships a SAM2 video path, but its only video node (``SAM2VideoDetectorSEGS``) takes a
``BBOX_DETECTOR`` and finds the subject itself on frame 0. vorch already has the box: the user drew
it on the keyframe (``Scene.locked_regions``). This node takes that rectangle, segments frame 0
from it, propagates through the clip and returns one mask per frame.

It calls ``build_sam2_video_predictor`` directly rather than going through Impact Pack's
``SAM2Wrapper``, because Impact's ``sam2_config_table`` only knows the SAM **2.0** filenames and
would not recognise a 2.1 checkpoint.

Measured on an RTX 5070 Ti (16303 MiB), ``sam2.1_hiera_small``, 81 frames, every frame tracked:
1747 MiB resident, or **742 MiB with offload at the same wall time (7.5 s)** — and independent of
the clip's resolution, because SAM2 resizes every frame to the model's own ``image_size``. So
``offload`` defaults to on: on this card it is free. See ``docs/layered-compositing-plan.md``.
"""

from __future__ import annotations

import logging
import os
import tempfile

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# SAM2 ships its hydra configs inside the package; the checkpoint filename picks one. Both
# generations are listed because `models/sams/` may hold either, and 2.1 is what is current.
CONFIGS = {
    "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_hiera_tiny.pt": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small.pt": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base_plus.pt": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large.pt": "configs/sam2/sam2_hiera_l.yaml",
}


def _checkpoints() -> list[str]:
    """The SAM2 checkpoints in ``models/sams``, newest generation first.

    Only names this node has a config for: a SAM v1 ``.pth`` in the same folder is not one.
    """
    try:
        import folder_paths
        names = folder_paths.get_filename_list("sams")
    except Exception:                                  # outside ComfyUI (import for tests)
        names = []
    found = [n for n in names if os.path.basename(n) in CONFIGS]
    return sorted(found, reverse=True) or ["sam2.1_hiera_small.pt"]


def _dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Grow the mask by ``px`` pixels (a max filter), so a matte that cuts just inside the object
    covers its edge instead of leaving a one-pixel seam of render around it."""
    if px <= 0:
        return mask
    k = 2 * px + 1
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=k, stride=1, padding=px).squeeze(1)


def _feather(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Blur the mask edge over ``px`` pixels, separably.

    Two reasons this belongs here and not in the compositor: the blend wants a soft edge anyway,
    and a hard binary mask rings badly once the matte is encoded as video.
    """
    if px <= 0:
        return mask
    sigma = px / 2.0
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=mask.dtype, device=mask.device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    m = mask.unsqueeze(1)
    m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return m.squeeze(1).clamp(0, 1)


class VorchSAM2BoxMatte:
    """One rectangle on frame 0 -> a per-pixel mask for every frame of the clip."""

    @classmethod
    def INPUT_TYPES(cls):
        frac = {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}
        return {
            "required": {
                "images": ("IMAGE",),
                # `ckpt_name`, not `ckpt`: the name is in `registry.MODEL_FIELDS`, so vorch
                # derives a model requirement from the template literal and
                # `registry validate --online` reports an uninstalled checkpoint by name.
                "ckpt_name": (_checkpoints(), {"tooltip": "SAM2 checkpoint in models/sams"}),
                # The box is in fractions of the frame, exactly as `media.Region` stores it, so a
                # mark survives any change of render size.
                "box_x": ("FLOAT", frac | {"tooltip": "left edge, fraction of the width"}),
                "box_y": ("FLOAT", frac | {"tooltip": "top edge, fraction of the height"}),
                "box_w": ("FLOAT", frac | {"default": 1.0, "tooltip": "width, fraction"}),
                "box_h": ("FLOAT", frac | {"default": 1.0, "tooltip": "height, fraction"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilate": ("INT", {"default": 2, "min": 0, "max": 64}),
                "feather": ("INT", {"default": 4, "min": 0, "max": 64}),
                "offload": ("BOOLEAN", {"default": True,
                                        "tooltip": "keep frames and state in RAM: 742 MiB instead "
                                                   "of 1747, at the same speed"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "matte"
    CATEGORY = "vorch"
    DESCRIPTION = __doc__

    def matte(self, images, ckpt_name, box_x, box_y, box_w, box_h,
              threshold, dilate, feather, offload):
        from sam2.build_sam import build_sam2_video_predictor

        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        if n == 0:
            raise ValueError("no frames to matte")
        if box_w <= 0 or box_h <= 0:
            raise ValueError(f"the box has no area: {box_w}x{box_h}")

        base = os.path.basename(ckpt_name)
        if base not in CONFIGS:
            raise ValueError(f"no SAM2 config for {base!r}; expected one of {sorted(CONFIGS)}")
        try:
            import folder_paths
            path = folder_paths.get_full_path("sams", ckpt_name) or ckpt_name
        except Exception:
            path = ckpt_name                            # outside ComfyUI: an explicit path
        if not os.path.isfile(path):
            raise ValueError(
                f"SAM2 checkpoint {base!r} not found in ComfyUI's models/sams "
                f"(looked at {path!r}). Download it with:\n"
                f"  curl -L -o ~/comfy/ComfyUI/models/sams/{base} "
                f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{base}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # SAM2 reads a clip from an MP4 or a folder of JPEGs only — never from a tensor — so the
        # frames ComfyUI already decoded go back out to a scratch folder. 81 frames cost well
        # under a second, against ~7 s for the propagation itself.
        with tempfile.TemporaryDirectory(prefix="vorch_matte_") as work:
            self._write_jpegs(images, work)
            pred = build_sam2_video_predictor(CONFIGS[base], path, device=device)
            try:
                state = pred.init_state(work, offload_video_to_cpu=bool(offload),
                                        offload_state_to_cpu=bool(offload))
                # `add_new_points_or_box` wants the box in the clip's own pixels (x0, y0, x1, y1).
                x0, y0 = box_x * w, box_y * h
                box = [x0, y0, min(w, x0 + box_w * w), min(h, y0 + box_h * h)]
                pred.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)

                masks = torch.zeros((n, h, w), dtype=torch.float32)
                seen = 0
                for idx, _obj_ids, logits in pred.propagate_in_video(state):
                    if idx >= n:
                        continue
                    # logits: [objects, 1, H, W] at the clip's resolution. One object here.
                    m = torch.sigmoid(logits[0, 0].float())
                    masks[idx] = (m > threshold).float().cpu()
                    seen += 1
                if seen < n:
                    log.warning("vorch matte: %d of %d frames tracked; the rest stay empty",
                                seen, n)
            finally:
                del pred
                if device == "cuda":
                    torch.cuda.empty_cache()

        masks = _feather(_dilate(masks, int(dilate)), int(feather))
        covered = float(masks.mean())
        log.info("vorch matte: %d frames, %.2f%% of the frame covered", n, covered * 100)
        return (masks,)

    @staticmethod
    def _write_jpegs(images, work: str) -> None:
        """IMAGE [B,H,W,3] float 0..1 -> ``00000.jpg``… , which is what SAM2 globs for."""
        from PIL import Image
        arr = (images.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        for i, frame in enumerate(arr):
            Image.fromarray(frame[..., :3]).save(os.path.join(work, f"{i:05d}.jpg"), quality=95)


NODE_CLASS_MAPPINGS = {"VorchSAM2BoxMatte": VorchSAM2BoxMatte}
NODE_DISPLAY_NAME_MAPPINGS = {"VorchSAM2BoxMatte": "SAM2 matte from a box (vorch)"}
