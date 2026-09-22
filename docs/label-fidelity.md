# Keeping printed text intact through animation

Measured on the Perfume Echoes job `b8de371f` (MAH "FLOWER BLOOMING", 5 assets, LTX 960×960),
September 2026. Every animated clip came back with the brand label misspelt. Three separate
causes, only one of them a property of the model.

## 1. The resolution was thrown away before the model saw it

The assets are 1024×1024. `_install_user_keyframe` fitted them to the profile's **keyframe**
size (768×768) and the clip then rendered at **960×960** — a quarter of the linear detail
discarded and asked back from the generator.

The keyframe size is a budget for *generating* a frame. For an image made elsewhere there is
nothing to generate, so `Orchestrator.install_size()` now takes the larger of the keyframe and
clip sizes. The aspects are already validated as equal, so this only ever changes the scale.

## 2. The prompt writer aimed the camera at the label

Left to itself Gemma wrote *"The camera performs a slow, elegant zoom-in towards the product
label"* for **every** scene of the job — pointing the camera at the one thing the model cannot
draw. `storyboard.TEXT_SAFE_RULE` now tells it to keep printed text still and put the movement
in the surroundings: petals, dust, liquid, light, shadow.

This also makes the fix below exact rather than approximate.

## 3. The type is below what a video VAE can carry

Wan and LTX both encode 8× in each spatial direction. Measured line heights on the s00 keyframe:

| line | at 768 (keyframe) | at 960 (render) | in latent cells |
|---|---|---|---|
| FLOWER / BLOOMING | 9 px | 11 px | 1.4 |
| COEUR DE PARFUM | 4 px | 5 px | 0.6 |
| MAH | 6 px | 7.5 px | 0.9 |
| PARIS | 2 px | 2.5 px | 0.3 |

A letter occupying less than one latent cell has no representation; it is reconstructed as a
plausible smear. This is why the fine subline was **already wrong in frame 0** — not drift. No
prompt, seed, sampler or model setting changes it. As a rule of thumb text needs **≥ 32 px of
cap height in the render** to survive, which for this bottle means either a 1280-class render or
much tighter framing.

### The durable answer: don't ask the model for it

A scene carries `locked_regions` — rectangles in fractions of the frame, so they survive any
render size. After each render `media.lock_regions` scales the scene's own keyframe to the clip,
masks it to those rectangles with a feathered edge, and overlays it on every frame in one ffmpeg
pass (a second for a 5 s clip, audio copied through).

- Mark them in the gate panel: **Lock the label…** under the clip, drag over the label.
- `PUT /api/jobs/{id}/scenes/{i}/locks` re-applies to the clip that already exists, so a smeared
  label costs a paste, not a re-render.
- Each pass starts from the **unlocked** render (`post_paths[-1]` or `raw_video_path`), never
  from a previous paste; clearing the marks gives the plain render back.
- A failed paste keeps the clip and logs a warning decision — losing a render over an overlay
  would be the worse trade.

This is exact wherever the marked object does not move, which is what rule 2 enforces. **A
camera move onto a locked label is not supported and should be added in post instead** (zooming
a sharp composite), because a static paste cannot follow a push-in — visible in the pre-fix clip
`s00_clip`, where the pasted patch is correct but the render beneath it had zoomed away.

## What not to try

- **Green screen**: keying an olive jacket out of a green screen fails; `despill` browns it, and
  on blue it lilacs a grey layer. Documented, not a bug.
- **Blending in YUV**: `blend=all_mode=multiply` turned a plate bright green. Force `format=gbrp`
  before any blend.
- **Regenerating the label with an image model**: no model tested preserved it — FLUX kv_edit
  produced "BEOWER BHORMNCE", an external render "COURE BESRNOM".
