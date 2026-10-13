# vorch's own ComfyUI nodes

Versioned here, installed into ComfyUI by symlink — the same arrangement as [`modal/`](../modal),
where assets live in the repo and are copied or deployed elsewhere. Keeping the node in this repo
is what makes it reviewable, and what lets `vorch registry validate --online` police it.

## Install

```bash
ln -s "$PWD/comfy_nodes/vorch_matte" ~/comfy/ComfyUI/custom_nodes/vorch_matte
```

Then restart ComfyUI and check it loaded:

```bash
vorch registry validate matte_sam2 --online
```

That command is the install check, and it covers **both** halves:

- **the symlink** — `_finalize` puts every `class_type` in a template into the spec's
  `requires_nodes`, so a missing node is named rather than discovered at render time;
- **the checkpoint** — the node calls its input `ckpt_name` rather than `ckpt` precisely because
  that name is in `registry.MODEL_FIELDS`. vorch then derives a model requirement from the
  template's literal and reports an uninstalled checkpoint by filename.

The second one is worth knowing: `enum_from` does *not* give you this. It offers the list in the
UI and checks the node class exists, but an **empty** option list — which is exactly what a
missing checkpoint looks like — passes validation silently. `tests/test_workflows.py` pins both
negatives.

## `VorchSAM2BoxMatte`

One rectangle on frame 0 → a per-pixel mask for every frame of the clip.

`(IMAGE frames, ckpt_name, box_x, box_y, box_w, box_h, threshold, dilate, feather, offload) → MASK`

The box is in **fractions of the frame**, exactly as `media.Region` stores it, so a mark survives
any change of render size.

### Why this exists when Impact Pack has SAM2

Impact Pack does ship a SAM2 video path (`SAM2Wrapper.predict_video_segs`), but its only video
*node* — `SAM2VideoDetectorSEGS` — takes a `BBOX_DETECTOR` and finds the subject itself on frame 0.
vorch already has the box: the user drew it. The only bbox model on this machine is
`face_yolov8m.pt`, which finds faces and is no use for a perfume label.

This node also calls `build_sam2_video_predictor` directly rather than going through Impact's
wrapper, because Impact's `sam2_config_table` knows only the SAM **2.0** filenames and would not
recognise the 2.1 checkpoint that upstream now ships.

### Requirements

- `sam2` in ComfyUI's interpreter (`~/WORK/LTX-Colab/venv`):
  ```bash
  SAM2_BUILD_CUDA=0 ~/WORK/LTX-Colab/venv/bin/python -m pip install \
      --no-build-isolation git+https://github.com/facebookresearch/sam2
  ```
  `--no-build-isolation` reuses the installed torch instead of downloading a second one;
  `SAM2_BUILD_CUDA=0` skips the optional CUDA extension, which upstream says only limits
  post-processing. Adds `SAM-2`, `hydra-core`, `iopath`, `portalocker` and **nothing else** —
  torch is untouched. Back out with `pip uninstall SAM-2 hydra-core iopath portalocker`.
  Do **not** `pip install sam2` from PyPI: that is an unrelated third-party fork.
- A checkpoint in `models/sams/`:
  ```bash
  curl -L -o ~/comfy/ComfyUI/models/sams/sam2.1_hiera_small.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
  ```

### Cost (measured, RTX 5070 Ti / 16303 MiB, `sam2.1_hiera_small`, 81 frames)

| | peak VRAM | wall time |
|---|---|---|
| `offload=False` | 1747 MiB | 7.5 s |
| `offload=True` (default) | **742 MiB** | 7.5 s |

Independent of the clip's resolution — SAM2 resizes every frame to the model's own `image_size`,
so 960×960 costs the same as 832×480. Offload is free in time, which is why it defaults to on.

## Tests

`tests/test_comfy_nodes.py`. The static checks run anywhere; the helper tests need torch *and* a
pytest in ComfyUI's venv, which it does not have by default. See that file's docstring.
