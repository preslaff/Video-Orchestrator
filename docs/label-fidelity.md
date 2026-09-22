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
render size. After each render `media.lock_regions` puts those pixels back from the frame the
model was given.

**The paste follows the shot.** A fixed rectangle is only right while the camera holds still,
which is not the shot anyone wants for a product — so `vorch/tracking.py` finds each region
again in every frame: a similarity search (uniform scale plus translation) seeded from the
previous frame, a few hundred normalised correlations per frame, no optical flow and no model.
Verified on the live clip through a **3× push-in**: the label stays on the bottle, at the right
size, reading correctly, at every scale.

**Three things stop it flickering**, all found by measuring a track that did:

1. **The template is cut from the clip's own first frame**, not from the source image. That
   reads backwards — the first frame is where the label is already wrong — but it is the point:
   every later frame is the model's continuation of *that*. Matching render against render
   correlates around **0.85**; matching a pristine label against a smeared one never passed
   **0.45**, lost the region for the first 29 frames, then lurched to catch up.
2. **The trajectory is smoothed** (`tracking.smooth`): frames the search could not trust are
   interpolated across rather than held, then the whole path is low-passed — in centre-and-scale,
   so a growing region does not creep sideways. A camera move is smooth; the search's output
   is not.
3. **The paste is sub-pixel and bilinear.** A rectangle that can only land on whole pixels has
   to step, and nearest-neighbour swaps whole rows of the source in and out as it grows, so the
   lettering crawls. Placements are fractional and the patch is resampled straight onto the
   destination grid, with edge pixels painted only as far as the rectangle covers them.

Measured on the live clip, before → after: stalls 37 → 11, reversals 3 → 0, jerk 2.96 → 2.30.
The old track also under-shot badly, ending at 468 px where the label is really 624 px.

Deliberately not estimated: rotation and perspective. A product hero shot does not roll, and a
wrong warp is worse than none. The search is local by construction, so it cannot jump to a false
match; when nothing correlates the placement holds and the frame is counted as *weak* in the log.

Details that matter:

- **The pixels come from the largest matching source** (`Orchestrator.lock_source`): the project
  asset when it is bigger than the keyframe *and shares its aspect* — `fit_image` centre-crops,
  so on a differently shaped asset the same fractions point somewhere else. That headroom is
  what a push-in resamples into.
- **The feather grows with the patch**, or the seam that is invisible at 1× becomes a bright
  edge at 3×.
- **A static shot must not shiver**: a region only moves when moving explains the frame better
  by a margin (`tracking.STICKINESS`), so noise alone never shifts the paste.
- Two passes: one greyscale decode to track (a third of the bytes), one to composite. Frames are
  streamed, so a long clip costs one frame of memory. About 12 s for 81 frames at 960², against
  minutes to render.
- `track=False` keeps the original exact path: one ffmpeg overlay at a fixed rectangle.

- Mark them in the gate panel: **Lock the label…** under the clip, drag over the label.
- `PUT /api/jobs/{id}/scenes/{i}/locks` re-applies to the clip that already exists, so a smeared
  label costs a paste, not a re-render.
- Each pass starts from the **unlocked** render (`post_paths[-1]` or `raw_video_path`), never
  from a previous paste; clearing the marks gives the plain render back.
- A failed paste keeps the clip and logs a warning decision — losing a render over an overlay
  would be the worse trade.

Rule 2 still matters, for a different reason than it first appeared: tracking keeps the label
*correct* through a dolly, but a 2.6× push-in resamples the patch up, so the type ends up
slightly soft. Keeping the camera off the label means it is both correct **and** sharp. Tracking
is what makes the feature usable when a shot does move — not a licence to zoom into type.

## What not to try

- **Green screen**: keying an olive jacket out of a green screen fails; `despill` browns it, and
  on blue it lilacs a grey layer. Documented, not a bug.
- **Blending in YUV**: `blend=all_mode=multiply` turned a plate bright green. Force `format=gbrp`
  before any blend.
- **Regenerating the label with an image model**: no model tested preserved it — FLUX kv_edit
  produced "BEOWER BHORMNCE", an external render "COURE BESRNOM".
