# Adding a ComfyUI workflow without touching Python

Every generation step is a ComfyUI **API-format** template (`*.json`) plus a sidecar
**registry YAML** that names the fields the orchestrator may replace. The five built-ins
live in `workflows/`; your own go in `~/vorch/workflows/` (`VORCH_USER_WORKFLOW_DIR`).
A user entry with the same `name` overrides a built-in.

## 1. Export the workflow from ComfyUI

Build and test the graph in the ComfyUI UI, then **Export (API)** (not the plain export —
the API format is a flat dict keyed by node id and is what `/prompt` accepts). Save it as
e.g. `gimm_interp.json`.

## 2. Inspect it

```bash
python -m vorch registry inspect gimm_interp.json --online
```

Prints every node with its literal inputs and a role guess (`<- positive`, `<- seed`,
`<- width`, `<- loader:unet_name`, `-> output:video`, …). With `--online` each loader shows
how many installed options it has and flags values that are not installed.

## 3. Scaffold the YAML

```bash
python -m vorch registry add gimm_interp.json --kind postprocess
```

Copies the JSON next to a generated `gimm_interp.yaml` in the user dir and prints it.
The scaffold uses the canonical parameter names for the chosen kind (see below); unmapped
numeric literals are listed as comments so you can promote the ones you want to control.
Edit names/description, then:

```bash
python -m vorch registry validate gimm_interp --online
```

`validate` checks that every target `(node, field)` exists in the template and is a literal
(not a link), that the kind's required params are present, and — online — that every node
class is installed in ComfyUI and every model file named in the template is available.

## 4. Use it

Reference the workflow from an **output profile**: `keyframe.workflow`, `video.workflow`,
`face_swap.workflow`, `edit_workflow`, or as an entry in `post_steps`
(`[{workflow: gimm_interp, params: {multiplier: 2}}]`). Nothing else changes.

## YAML reference

```yaml
name: gimm_interp                  # default: file stem
kind: postprocess                  # keyframe | animate | faceswap | edit | postprocess | custom
template: gimm_interp.json         # relative to this YAML, or absolute
description: GIMM-VFI 2x frame interpolation
tags: [interpolation, video]
timeout: 900                       # seconds; default config.RUN_TIMEOUT
inputs: {input_video: video}       # params that are FILES staged into ComfyUI input/
outputs: {type: video, nodes: ["12"]}
params:
  input_video: {node: "1", field: video, type: str, required: true}
  multiplier:  {node: "5", field: interpolation_factor, type: int, min: 1, max: 8}
  fps:         {node: "11", field: frame_rate, type: int, min: 1, max: 60}
  seed:        {node: "5", field: seed, type: int}
  filename_prefix: {node: "12", field: filename_prefix, type: str}
requires: {nodes: [GIMMVFI_interpolate, VHS_LoadVideo, VHS_VideoCombine]}   # optional
```

Param fields: `node` + `field` (or `targets: [{node, field}, …]` to write several nodes),
`type` (`int|float|str|bool|enum|any`), `required`, `min`/`max`, `constraint`
(`4n+1`, `multiple_of_16`, `multiple_of_8`, `positive`), `enum_from: {class_type, input}`
(validated against `/object_info`). Loader model files (`unet_name`, `vae_name`,
`swap_model`, …) are discovered from the template automatically for the environment check.

### Canonical names per kind

| kind | required | optional (dropped with a warning if the workflow lacks them) |
|---|---|---|
| keyframe | positive, seed, width, height | negative, steps, cfg, unet, batch_size, sampler, filename_prefix |
| animate | start_image (image input), positive, length | negative, width, height, fps, seed, steps, cfg, split_step, filename_prefix |
| faceswap | input_image, source_image (image inputs) | restore_model, filename_prefix |
| edit | positive, seed, ≥1 image input (`reference`, `reference_2`, …) | width, height, steps, cfg, unet, negative, batch_size, sampler, filename_prefix |
| postprocess | ≥1 input (`input_video` or `input_image`) | fps, seed, filename_prefix |
| custom | — | anything |

The pipeline passes the optional names it knows; a workflow that does not declare one just
ignores it. That is what lets a leaner or richer user workflow replace a built-in.

### Postprocess chains

A `postprocess` workflow takes one media file (`input_video` or `input_image`) and produces
one. Profiles list them in `post_steps`; the orchestrator runs them in order after each clip
or still, staging the previous output into ComfyUI `input/`, filling `fps`/`seed` from the
profile when declared, and freeing VRAM between different workflows. Steps whose input type
does not match the media are skipped (an interpolator is skipped for stills).

Example graph for GIMM-VFI (installed here as `GIMMVFI_interpolate`):
`VHS_LoadVideo` → `GIMMVFI_interpolate` (`interpolation_factor`, `seed`) → `VHS_VideoCombine`
(`frame_rate`, `filename_prefix`, `format: video/h264-mp4`). Export it as API JSON and run
the three commands above; the node ids in the YAML come from `inspect`.

## Gotchas

- **Templates are never restructured.** Only listed literal fields are replaced. If a field
  is a link (`["75:68", 0]`), map the node that feeds it (e.g. the `PrimitiveInt`) instead.
- **Wan length must be 4n+1** (49, 81, 121…). Declare `constraint: 4n+1` on it.
- **Wan `steps` without `split_step`** keeps the HighNoise/LowNoise split at the template
  value (10) — set both.
- **KV-edit output size follows the first reference** (the graph reads it via
  `GetImageSize`), so it has no `width`/`height`; pre-fit references instead.
- The `filename_prefix` of a Save node decides the output subfolder (`video/...`); the app
  resolves output paths from the run history, so any prefix works.
