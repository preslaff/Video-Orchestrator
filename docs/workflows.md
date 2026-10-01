# Adding a ComfyUI workflow without touching Python

Every generation step is a ComfyUI **API-format** template (`*.json`) plus a sidecar
**registry YAML** that names the fields the orchestrator may replace. The five built-ins
live in `workflows/`; your own go in `~/vorch/workflows/` (`VORCH_USER_WORKFLOW_DIR`).
A user entry with the same `name` overrides a built-in.

## 0. The curated hosted models

`workflows/hosted/` ships a small set, one model per provider, each with a price and each
validating against a live ComfyUI:

| workflow | kind | ~cost/call | why this one |
|---|---|---|---|
| `api_wan_t2i` | keyframe | $0.03 | the cheap workhorse; literal sizes 768–1440 |
| `api_gpt_image_2_t2i` | keyframe | $0.006–0.02 | cheapest per image at `quality: low`; strong instruction-following |
| `api_dalle3_t2i` | keyframe | $0.04–0.12 | three sizes only, but reliable |
| `api_seedream5_pro_t2i` | keyframe | $0.045–0.09 | literal sizes up to 4K |
| `api_qwen3_t2i` | keyframe | $0.057–0.107 | literal sizes, and a real negative prompt |
| `api_ideogram_v4_t2i` | keyframe | $0.043–0.143 | by far the best at type **inside** the image, Cyrillic included |
| `api_nano_banana2_t2i` | keyframe | $0.084–0.185 | Gemini; six aspects |
| `api_flux_erase` | edit | $0.03–0.06 | erase a region; constant price |
| `api_nano_banana2_edit` | edit | $0.041–0.185 | reference-locked edit |
| `api_seedream5_pro_edit` | edit | $0.045–0.09 | SeedEdit |
| `api_flux_kontext_max_edit` | edit | $0.08–0.12 | Kontext Max, the reference-locked standard |

They need `VORCH_COMFY_API_KEY` (see the README). Prices were read from ComfyUI's own
`price_badge` on 2026-09-30 and are what **api.comfy.org** bills, which is not the provider's
list price; each file records where its figure came from. `api_flux_kontext_max_edit` is the one
exception — ComfyUI states no price for it, so its figure is BFL's list rate with room for markup.

They live apart from the local workflows so a ComfyUI built without `comfy_api_nodes` still has a
clean registry, and none of them takes a local workflow's name — a hosted entry called `keyframe`
would quietly send every job to a paid API.

## 1. Or import any of the others in bulk

ComfyUI ships an example workflow for every hosted provider it proxies (`api_*`), and serves them
over HTTP. `import` converts them to API format and scaffolds a YAML for each:

```bash
vorch registry import --match api_ --kinds keyframe edit     # into ~/vorch/workflows
vorch registry import --match api_openai --dry-run           # see first
vorch registry validate --online
```

It needs ComfyUI running — widget *order* is not recorded in a UI export, so the node schemas from
`/object_info` are what line the values up — but no API key and no credits: converting and
registering costs nothing. The generated entry takes its description from ComfyUI's own template
index, builds an `aspect`/`from_aspect` table from the node's size enum, and declares whatever the
kind requires but the graph cannot offer (see `targets: []` below). What it cannot guess is the
price, so add a `cost:` before running anything you care about.

Hosted `animate` entries mostly do **not** validate yet: these APIs bill a clip in **seconds**
(`duration`), while vorch's `animate` contract wants a frame `length` of 4n+1. Keyframe, edit and
postprocess import cleanly.

A single UI export works too — `vorch registry add my_export.json` converts it automatically
rather than telling you to re-export.

## 1a. Choosing a model per job and per scene

A profile sets the keyframe workflow for every scene (`setup/profiles`). Two narrower choices sit
on top:

- **per job** — the new-job wizard's *still model*, folded into `profile_overrides` so the job's
  snapshot records what it actually used. When the chosen model costs money the wizard also shows
  the estimated spend (scenes × variants × attempts — the retries are what make it bite) and takes
  a **budget**.
- **per scene** — *render with*, at the keyframes gate, on t2i scenes. `Scene.keyframe_workflow`
  (`""` = the profile's), the same shape as `Scene.edit_workflow`. This is the one worth having:
  one shot needs type drawn inside the picture and can afford Ideogram, the rest do not and
  should not pay for it.

An unknown or wrong-kind name is a 400 when the scene is saved, not a failure halfway through the
run. If the chosen model cannot render the job's aspect, the gate says so and names the aspects it
does offer, rather than letting the render come back the wrong shape.

## 2. Export the workflow from ComfyUI

Build and test the graph in the ComfyUI UI, then **Export (API)** (not the plain export —
the API format is a flat dict keyed by node id and is what `/prompt` accepts). Save it as
e.g. `gimm_interp.json`.

## 3. Inspect it

```bash
python -m vorch registry inspect gimm_interp.json --online
```

Prints every node with its literal inputs and a role guess (`<- positive`, `<- seed`,
`<- width`, `<- loader:unet_name`, `-> output:video`, …). With `--online` each loader shows
how many installed options it has and flags values that are not installed.

## 4. Scaffold the YAML

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

## 5. Use it

Reference the workflow from an **output profile**: `keyframe.workflow`, `video.workflow`,
`face_swap.workflow`, `edit_workflow`, or as an entry in `post_steps`
(`[{workflow: gimm_interp, params: {multiplier: 2}}]`). Nothing else changes.

## YAML reference

```yaml
name: gimm_interp                  # default: file stem
kind: postprocess                  # keyframe | animate | faceswap | edit | postprocess | custom
backend: local                     # or a name from ~/vorch/backends.yaml (e.g. modal-h100)
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

`backend:` (optional, default `local`) names the ComfyUI that runs this workflow — an entry of
`~/vorch/backends.yaml` (see `docs/modal-backend.md`). Outputs of remote backends are downloaded
into the local output layout with the backend name in the file name.

Param fields: `node` + `field` (or `targets: [{node, field}, …]` to write several nodes,
or `targets: []` to declare a param that writes nowhere — see below),
`type` (`int|float|str|bool|enum|any`), `required`, `min`/`max`, `constraint`
(`4n+1` and `multiple_of_16` for Wan, `8n+1` and `multiple_of_32` for LTX-2.x, `multiple_of_8`, `positive`), `enum_from: {class_type, input}`
(validated against `/object_info`). Loader model files (`unet_name`, `vae_name`,
`swap_model`, …) are discovered from the template automatically for the environment check.

### Canonical names per kind

| kind | required | optional (dropped with a warning if the workflow lacks them) |
|---|---|---|
| keyframe | positive, seed, width, height | negative, steps, cfg, unet, batch_size, sampler, filename_prefix |
| animate | start_image (image input), positive, length | negative, width, height, fps, seed, steps, cfg, split_step, filename_prefix |
| faceswap | input_image, source_image (image inputs) | restore_model, filename_prefix |
| edit | positive, seed, ≥1 image input (`reference`, `reference_2`, …) | width, height, steps, cfg, unet, negative, batch_size, sampler, filename_prefix |
| postprocess | ≥1 input (`input_video` or `input_image`) | fps, seed, filename_prefix, source_image (auto-filled with the job's master face) |
| custom | — | anything |

The pipeline passes the optional names it knows; a workflow that does not declare one just
ignores it. That is what lets a leaner or richer user workflow replace a built-in.

The **required** names are different: a workflow of that kind which does not declare one fails to
load (`RegistryError`), and a user YAML that fails to load is skipped and reported instead of
registered. Some graphs genuinely cannot offer one — a hosted API node may have no seed input at
all. For those, declare the param with an **empty target list**:

```yaml
params:
  positive: {node: "58", field: prompt, type: str}
  seed:     {targets: [], type: int}     # this graph has nowhere to put a seed
```

An empty `targets` satisfies the contract, is still type-checked and constraint-checked like any
other param, and writes nothing into the template. Use it only when the graph really has no such
field — not to silence a mapping you have not done yet, because the result is a value the user sets
and the render ignores. Steps send a param only when the workflow declares it (`spec.accepts`), so
a workflow may simply omit an *optional* name rather than declaring it targetless.

### `cost:` — what a hosted call is billed

A workflow that calls a hosted model declares its price, and `validate --online` treats a
missing one as a problem: every call spends money, and the evaluator's retry loop multiplies it.

```yaml
cost: {usd: 0.03}                          # a flat price
cost: {min_usd: 0.04, max_usd: 0.12}       # when it depends on size or quality
```

A job carries `budget_usd`; the ceiling is checked **before** each billed call and reserves the
**upper** bound, so a range never overshoots. Crossing it pauses the job on a `budget` gate with
what it would have cost and what is left — raise the ceiling (`PUT /api/jobs/{id}/budget`) before
approving, or it stops again at the very next call. Every call is logged on the job, retries
included, so a scene the evaluator rejected twice shows three entries.

`registry import` fills this in from the node's own `price_badge` when ComfyUI states a plain
number — about 70 of 241 do. The rest express it as a JSONata expression (megapixel maths,
branching on the chosen model) which vorch does not evaluate, so those need the figure copied
from the provider. Where a literal *is* available, `validate --online` compares it with what the
YAML claims and reports drift.

Two honest limits. The figures are **estimates**: the real charge comes back from api.comfy.org
as a header ComfyUI keeps in memory and emits only on its websocket, never in `/history`, which
is vorch's only transport — so nothing reconciles estimate against actual. And a workflow with no
`cost:` is counted as *unpriced* rather than free: it is never blocked, but it is not silently
treated as costing nothing either.

### Size as an enum: `from_aspect`

Local samplers take literal `width`/`height`. Many hosted models instead offer a fixed set of
sizes, each in its own dialect — `"1792x1024"`, `"16:9"`, `"1080:1920"`, `"1440x2560 (9:16)"`.
Declare an `aspect` param whose `from_aspect` table maps the job's aspect onto that node's label,
and leave `width`/`height` targetless:

```yaml
params:
  positive: {node: "58", field: prompt, type: str}
  seed:     {targets: [], type: int}
  width:    {targets: [], type: int}
  height:   {targets: [], type: int}
  aspect:
    node: "58"
    field: size
    type: enum
    from_aspect:            # vorch aspect -> this node's own label
      "16:9": "1792x1024"
      "9:16": "1024x1792"
      "1:1":  "1024x1024"
```

The table is also the workflow's **declaration of what it can render**. An aspect it does not list
is refused when the profile is saved and when the job is validated — not at render time — because
the alternative is a 9:16 job silently coming back square, visible only in the picture.
`registry validate --online` additionally checks every label against the node's real enum, so a
typo is caught before a paid call. A workflow that genuinely takes ints just wires `width`/`height`
as usual and declares no `from_aspect`; its own `min`/`max`/`constraint` then set the size limits
(which is how a model that renders above the local 2048 ceiling becomes usable).

### Postprocess chains

A `postprocess` workflow takes one media file (`input_video` or `input_image`) and produces
one. Profiles list them in `post_steps`; the orchestrator runs them in order after each clip
or still, staging the previous output into ComfyUI `input/` (on the step's backend), filling `fps`/`seed` from the
profile when declared, and freeing VRAM between different workflows. Steps whose input type
does not match the media are skipped (an interpolator is skipped for stills).
A step that declares a `source_image` param (video face swap, Wan-Animate-2 character replacement) receives the job's master face, staged once per backend; an explicit `params: {source_image: …}` wins. Built-in example: `faceswap_video` (ReActor over every frame, keeps fps and audio).

Example graph for GIMM-VFI (installed here as `GIMMVFI_interpolate`):
`VHS_LoadVideo` → `GIMMVFI_interpolate` (`interpolation_factor`, `seed`) → `VHS_VideoCombine`
(`frame_rate`, `filename_prefix`, `format: video/h264-mp4`). Export it as API JSON and run
the three commands above; the node ids in the YAML come from `inspect`.

## Gotchas

- **Templates are never restructured.** Only listed literal fields are replaced. If a field
  is a link (`["75:68", 0]`), map the node that feeds it (e.g. the `PrimitiveInt`) instead.
- **Wan length must be 4n+1** (49, 81, 121…). Declare `constraint: 4n+1` on it. **LTX-2.x length is 8n+1** (97, 121…) and its sizes are multiples of 32 — declare `8n+1` / `multiple_of_32`; profiles and the wizard read these constraints from the animate workflow you pick (`formats.workflow_rules`), so a profile on `animate_ltx25` rejects 81 frames while one on `animate` accepts them.
- **Wan `steps` without `split_step`** keeps the HighNoise/LowNoise split at the template
  value (10) — set both.
- **KV-edit output size follows the first reference** (the graph reads it via
  `GetImageSize`), so it has no `width`/`height`; pre-fit references instead.
- The `filename_prefix` of a Save node decides the output subfolder (`video/...`); the app
  resolves output paths from the run history, so any prefix works.
