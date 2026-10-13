# Layered compositing — can we build the layered architecture tool?

> Status: in progress, 2026-10-13. **P1.0 and P1.1 are done** (the lock endpoint runs off the
> event loop; SAM2 is installed and the VRAM gate passed — see the gate result below). P1.2 onward
> is specified; phases 2+ are sketched with their cost, not designed. Machine facts were checked on
> the box and are marked as measured; anything estimated says so.

## The question, and the answer

The question came out of the Vera layered-diffusion paper (`2606.23610v1.pdf`), whose three-layer
representation — edit layer, alpha matte, composite — read like something you would have to train a
Mixture-of-Transformers to obtain. Can vorch build the layered tool?

Yes, and most of it is already built. **vorch already composites with an alpha matte.** It just
synthesises the matte from a rectangle.

`vorch/tracking.py:274-300`, `paste()`:

```python
alpha = (_edge_alpha(yc, at.y, at.y + at.h, feather)[:, None]
         * _edge_alpha(xc, at.x, at.x + at.w, feather)[None, :])[..., None]
region = frame[y0:y1, x0:x1]
region[:] = region * (1.0 - alpha) + sub * alpha
```

That is alpha-over, textbook. The only rectangular thing about it is where `alpha` comes from:
`_edge_alpha` (`tracking.py:261-271`) is a per-axis ramp computed from the **box bounds**. Give that
line a real per-pixel matte and `paste()` is a layer compositor.

### Why this beats reproducing Vera, for this pipeline

Vera's headline result is background preservation under editing — PSNR 36.2 against 31.7 for the
best baseline. That number has to exist because Vera re-generates the whole frame and tries not to
disturb the parts it shouldn't. Compositing does not have that problem to solve: **the background
is never shown to a model**, so model-induced drift is not 36 dB down, it is absent. The pixels are
exact up to the one `crf 16` encode every pixel already passes through.

Vera needs its architecture because it edits video it did not make, from a text instruction.
**vorch made the video.** It controls generation, so it can author the layers up front instead of
training a model to infer them afterwards. That is a real architectural advantage of being the
generator rather than the editor.

What is *not* buildable without Vera's weights is Vera's actual product: "here is a finished clip
and a sentence, change it." That is a different tool from this one, and this document does not
claim it.

Why any of this matters is settled in [`label-fidelity.md`](label-fidelity.md): a video VAE
compresses 8x in each spatial direction, so a glyph occupying less than one latent cell has no
representation and comes back as a plausible smear. No prompt, seed, sampler or weight changes it.
Locked regions is the right answer; this plan makes it the right *shape*.

## What already exists

Phase 1 is small because the unglamorous majority of a layer compositor is written and tested.

| part | where | what it already does |
|---|---|---|
| the blend | `tracking.py:274-300` `paste` | alpha-over, with sub-pixel `area_resample` so scale and fractional position are one operation |
| the edge ramp | `tracking.py:261-271` `_edge_alpha` | never narrower than one pixel, so a fractional edge does not jump |
| the compositor | `media.py:1130+` `_lock_tracked` | streams frames — decode pipe to numpy to encode pipe — so a long clip costs **one frame** of memory |
| per-frame placement | `media.py:1087` `_follow_regions` | finds each region again in every frame, so the paste follows a dolly |
| the tracker | `tracking.py` `follow`/`smooth` | `MIN_CONFIDENCE = 0.25`, `STICKINESS = 0.004` — a locked-off shot does not shiver |
| detail stability | `media.py:1159-1167` | reconstructs each patch **once** at the largest it will ever be drawn; measured **~3x steadier** lettering than rescaling per frame |
| lighting match | `media.py:1045`, `:1066` `_ring_level`/`_relight` | matches the patch to the brightness of the ring around it |
| the ffmpeg path | `media.py:978-991` | the codebase's only `alphamerge` — a `drawbox` mask fed to `overlay`, flattened in the same graph |

The same four floats currently have **six representations**, and phase 1 extends all six rather
than replacing any:

1. `media.Region` — `media.py:892-915`, frame fractions, `.box()` is the only fraction-to-pixel step
2. `Scene.locked_regions: list[dict]` — `state.py:184`, the persisted form
3. `schemas.LockedRegion` / `SceneLocks` — `api/schemas.py:212`, `:277`
4. `tracking.Placement` — `tracking.py:41`, the per-frame pixel form
5. the frontend `LockedRegion` + `Drag` — `LockRegions.tsx:8`, plus CSS-percent rendering
6. the ffmpeg `drawbox` string in the untracked path — `media.py:978-991`

Only **two** production sites construct a `Region`, both `Region(**r)` from a dict:
`orchestrator.py:1010` (inside `_lock_regions`) and `orchestrator.py:1038` (inside
`set_locked_regions`). Both matter in P1.5 below, for a reason that is easy to miss.

## Two facts that set the staging

**Persistence is free.** There is no `scenes` table — the whole `Job`, scenes included, is one
`data jsonb` column (`migrations/0001_init.sql:5-13`). *No scene field has ever had a SQL column*:
`locked_regions`, `texts`, `clips`, `variants`, `keyframe_workflow` all arrived with no migration,
because `_scene_from_dict` uses `d.get(...)` defaults and old rows load for free. A migration is
needed only for a new top-level `jobs` column, a JSONB backfill, or an expression index. **Phase 1
needs none of the three.**

**Alpha persistence is not free.** Nothing in vorch can write, probe, serve or round-trip an alpha
channel:

- every encode hardcodes `-c:v libx264 ... -pix_fmt yuv420p -movflags +faststart -f mp4`
  (`media.py:859`, `:999-1000`, `:1177-1178`)
- `_lock_tracked`'s `.reshape(h, w, 3)` hardcodes three channels; every raw pipe is `rgb24` or `gray`
- `MediaInfo` (`media.py:272+`) carries `width`, `height`, `fps`, `duration`, `nb_frames`, `codec`
  and `has_audio` — but **no `pix_fmt`, no `color_range`, no `has_alpha`**, and `probe`'s
  `-show_entries` list (`media.py:304-306`) does not request them. P1.4 fixes this, because the
  missing one bites.
- `/media` is a `StaticFiles` mount that guesses MIME from the extension (`api/media.py`), and
  `MediaThumb.tsx:52` is a bare `<video src>` that will not play ProRes 4444
- `grep -i rgba` over `vorch/` and `frontend/src/` returns **zero hits**; no prores, no vp9, no
  yuva420p, no PNG-sequence path

So **phase 1 deliberately never persists alpha.** The matte is a luma intermediate: generated,
consumed in the composite, flattened to h264. The alpha stays a separate `HxW` float array inside
`paste`, never a channel — so `.reshape(h, w, 3)` stays, every encode stays `yuv420p`, and the zero
`rgba`/`yuva420p`/prores hits stay zero. Introducing the first alpha-capable container is phase 2's
cost, and it is a real one.

---

# Phase 1 — the matte

Replace the tracked rectangle with a tracked **shape**. Local, free, and it fixes a current failure:
a label on a curved bottle, a logo on fabric, a product panel that is not a quadrilateral.

Commits are ordered so the first one is a standalone fix, the second is a go/no-go, and nothing
expensive is built before the thing it depends on is proven.

## P1.0 — make the lock endpoint safe to block. A prerequisite, not a nicety.

`JobManager.set_locked_regions` (`queue.py:595-601`) is a **plain `def`**, and `set_scene_locks`
(`api/jobs.py:348`) calls it directly from an `async def` route. So the paste runs **on the event
loop thread**, and it does **not** take `self._lock` — the single-GPU lock. `_editable_job`
(`queue.py:511-522`) says so itself: *"it says nothing about the GPU. Anything that needs the card
must take `self._lock` as well."*

Today that is merely bad: an 81-frame tracked ffmpeg paste freezes every request and every SSE
stream for its duration. With a SAM2 propagation in the path it is fatal — minutes of blocked event
loop, a GPU run outside the lock, no `cancel.check()`, no `vram.reclaim`.

The correct shape is three methods above it. `speak_scene` (`queue.py:570-585`) is `async def`,
takes `async with self._lock:`, and hands the work to `run_in_executor` — and its docstring records
that this was a *correction* made for exactly these two reasons, after a Speak loaded 11268 MiB on
top of another job's render.

So: convert `set_locked_regions` to that shape and `await` it in `api/jobs.py`. **This is a real fix
for a bug that exists today**, independent of mattes, and it ships first on its own.

## P1.1 — install SAM2. Go/no-go gate.

This was the step feared to be the blocker. It is not. Measured in ComfyUI's venv on 2026-10-13:

| fact | value |
|---|---|
| ComfyUI's venv | `~/WORK/LTX-Colab/venv` — from `services_builtin/services.yaml:63`, **not** `~/comfy/ComfyUI/venv` |
| torch / torchvision | `2.11.0+cu130` / `0.26.0+cu130` — satisfy SAM2's `torch>=2.5.1`, `torchvision>=0.20.1` |
| already present | `omegaconf`, `tqdm`, `PIL`, `numpy`, `ultralytics`, `segment_anything` |
| **missing** | `sam2`, and only two transitive deps: **`hydra-core`** and **`iopath`** |
| nvcc | 13.1 present — and **not needed**, see below |
| on disk already | `models/sams/sam_vit_b_01ec64.pth` (SAM **v1**, 375 MB) and `models/ultralytics/bbox/face_yolov8m.pt` |

Install from **the official Meta repo**:

```bash
SAM2_BUILD_CUDA=0 ~/WORK/LTX-Colab/venv/bin/python -m pip install \
    --no-build-isolation git+https://github.com/facebookresearch/sam2
```

- `--no-build-isolation` matters: SAM2's `pyproject.toml` declares `torch>=2.5.1` as a *build*
  requirement, so without it pip builds in an isolated env and downloads a second, different torch.
- `SAM2_BUILD_CUDA=0` skips the optional `sam2/csrc/connected_components.cu` extension. Upstream's
  own `setup.py` says skipping it only limits post-processing and "doesn't affect the results in
  most cases", and `SAM2_BUILD_ALLOW_ERRORS=1` is already its default, so a build failure would not
  abort the install anyway. The package name is `SAM-2`; it imports as `sam2`.
- **`pip freeze` the venv first.** It is the venv every render runs in. If the install wants to move
  torch, `--no-deps` plus installing `hydra-core` and `iopath` by hand is the whole answer.

**Do not `pip install sam2` from PyPI.** PyPI `sam2` 1.1.0 is a third-party fork
(`JinsuaFeito-dev/segment-anything-2`) declaring **zero** dependencies. Impact Pack's own error
message points at the Meta repo; so does this plan.

Weights — **SAM 2.1**, not 2.0:

```
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

Verified HTTP 200, **184 MB**, into `~/comfy/ComfyUI/models/sams/`. Configs ship inside the package
at `sam2/configs/sam2.1/sam2.1_hiera_s.yaml`. The 2.0 URLs are **commented out** in upstream's
`checkpoints/download_ckpts.sh` — upstream now ships 2.1 only.

### Gate result: **GO**, and cheaper than feared (measured 2026-10-13)

The install went exactly as the dry run promised — `Would install SAM-2-1.0 hydra-core-1.3.7
iopath-0.1.10 portalocker-4.4.0`, **torch and torchvision untouched**. (`portalocker` is a
transitive of `iopath`, one more than this plan first predicted.) Back out with
`pip uninstall SAM-2 hydra-core iopath portalocker`.

Peak VRAM for an 81-frame propagation of `sam2.1_hiera_small`, prompted by one box on frame 0,
every frame tracked:

| render size | in VRAM | with CPU offload | wall time |
|---|---|---|---|
| 832x480 | 1747 MiB | **742 MiB** | 7.5 s |
| 480x832 | 1749 MiB | **741 MiB** | 7.4 s |
| 960x960 | 1751 MiB | **744 MiB** | 7.5 s |

Two findings that matter more than the absolute number:

- **Peak VRAM is independent of the render size.** SAM2 resizes every frame to the model's own
  `image_size` internally, so a 960x960 clip costs the same as 832x480. The worry that drove this
  into a gate — that propagation holds per-frame features for the whole sequence — is real but
  bounded, and bounded at the model's resolution, not the clip's.
- **`offload_video_to_cpu=True, offload_state_to_cpu=True` costs nothing in time** (7.4–7.6 s
  either way) and cuts peak VRAM by 58 %. So the node should offload by default; there is no
  trade to weigh.

Against a 16303 MiB card where ComfyUI idles at 222 MiB, **742 MiB is not a constraint** — it is
less than Gemma's 1838 MiB resident floor. The contingency of propagating at reduced resolution and
upscaling the mask is not needed and should not be built.

One expected message, confirmed benign: `cannot import name '_C' from 'sam2'` — the skipped CUDA
extension, exactly as upstream's `setup.py` says ("Skipping the post-processing step… doesn't
affect the results in most cases"). All 81 frames were still tracked.

**The go/no-go check** was `python -c "from sam2.build_sam import build_sam2_video_predictor"`
in that venv; it passed, and `torch.cuda.is_available()` is still `True`. Had it failed, the plan
was to stop. The fallback is the SAM **v1** already on disk, prompted per frame by the box
`_follow_regions` already tracks: no install at all, harder edges, temporal coherence from the
tracker rather than the model. It was the cheaper route and was considered; SAM2 was chosen for
genuine video propagation. Nothing but notes is committed in this step.

## P1.2 — a ComfyUI custom node, versioned in this repo

A new node is unavoidable, and it is worth being precise about why.

Impact Pack **already has a SAM2 video path** (paths below are in
`~/comfy/ComfyUI/custom_nodes/comfyui-impact-pack/`, not in this repo): Impact Pack's `modules/impact/core.py:639+` `class SAM2Wrapper`,
with `predict_video_segs(image_frames, segs)` building a `build_sam2_video_predictor` and
propagating, and a `predict(image, points, plabs, bbox, threshold)` that takes a bbox directly. But
its only shipped *video* node, `SAM2VideoDetectorSEGS` (Impact Pack's `modules/impact/detectors.py:303-348`),
requires a `BBOX_DETECTOR` and **auto-detects** the subject on frame 0:

```python
segs = bbox_detector.detect(image_frames[0].unsqueeze(0), bbox_threshold, 0, 0, drop_size)
```

vorch already has the box — the user drew it. And the only bbox model on disk is `face_yolov8m.pt`,
which finds faces and is useless for a perfume label. So the shipped node is the wrong shape.

The node should also **not couple to Impact Pack's config table**: `sam2_config_table`
(Impact Pack's `modules/impact/impact_pack.py:97-105`) maps only the **2.0** filenames (`sam2_hiera_small.pt` →
`configs/sam2/sam2_hiera_s.yaml`), so a 2.1 checkpoint downloaded today is not recognised by its
`SAMLoader`. Call `build_sam2_video_predictor` directly and target current weights.

Shape:

```
(IMAGE frames, STRING ckpt, FLOAT box_x, box_y, box_w, box_h,   # frame fractions, as Region
 FLOAT threshold, INT dilate, INT feather) -> MASK
```

Internals: box on frame 0 → SAM 2.1 video propagation → dilate → **gaussian feather** → per-frame
`MASK` batch. Feather in the node, for two reasons: it is what the blend wants anyway, and a binary
mask rings badly under h264.

**Segment the clip, never the keyframe.** The matte must be computed from the render's own frames,
in clip space — the same reason `_follow_regions` cuts its template from the clip's first frame. A
matte of the keyframe is in the wrong space and would then need warping.

**Home:** a new top-level directory in this repo, mirroring `modal/` — the established precedent for
repo-versioned assets installed elsewhere (`modal/` holds `workflows/`, `profiles/`,
`download_models.py`, `upload_local_models.sh`, `README.md`). Nothing is currently symlinked into
ComfyUI's `custom_nodes/`, so the symlink is an install step with its own README line.

Install verification comes free: `_finalize` (`registry.py:502-504`) puts **every** template
`class_type` into `requires_nodes`, and `vorch registry validate --online` iterates `reg.list()`
(`__main__.py:933`) — every registered spec, whatever its kind. So one command reports a missing
symlink.

## P1.3 — the matte workflow: `kind: custom`, not `postprocess`

`workflows/matte_sam2.yaml`. The kind choice is the one place where the obvious answer is wrong.

`postprocess` satisfies the contract — `Contract(frozenset(), frozenset({"fps", "seed",
"filename_prefix"}), min_inputs=1)` (`registry.py:130`), nothing required, one input of any type —
so it is *legal*. It is still wrong, because `ProfileEditor.tsx:195-201` offers **every**
registered `postprocess` workflow in the post-step picker, and `run_chain`'s only kind check is
`spec.kind != "postprocess"` → raise (`steps/postprocess.py:60`). A user who picked it from that
dropdown would get the clip staged as `input_video`, the workflow run with the YAML's default box,
and then `current = outs[0]` — **the deliverable clip replaced by a greyscale matte** and handed to
the next step. That is a footgun the UI advertises.

`custom` — `Contract(frozenset(), frozenset())` (`registry.py:131`), and already the default kind
(`:391`) — means exactly what this workflow is: registered, invoked by name with its own params,
driven by no chain. It is absent from the post-step picker and `run_chain` refuses it. And it adds
**no registry kind**, so none of the nine change sites a kind costs is touched (`registry.py:47`,
`api/workflows.py:19`, `__main__.py:1061-1064` plus three argparse `choices`, four functions in
`registry_tools.py`, `docs/workflows.md:124` and `:152-165`, two frontend pickers).

`custom`'s empty `optional` set is harmless here: `Registry.apply` consults
`KIND_CONTRACTS[kind].optional` **only** for params the spec does not declare
(`registry.py:550-555`), and this spec declares all of its own.

Template: copy the skeleton of `workflows/reactor_video_faceswap.json` — `LoadVideo` →
`GetVideoComponents` → (work) → `CreateVideo` → `SaveVideo` — which `workflows/faceswap_video.yaml`
already drives with `inputs: {input_video: video}` and
`input_video: {node: "1", field: file, type: str, required: true}`. No new input type and no new
param type: that YAML already uses `type: float` with `min`/`max`. Output a **greyscale matte
video** at near-lossless crf.

Invoked **directly via `run_step`** (`steps/__init__.py:18-47`), not through `run_chain`, because
the box is per-scene-per-region and cannot come from a profile's static `post_steps` params.

**Validate where the user is standing.** `envcheck` only sees workflows in
`profiles.workflows_used()` (`profiles.py:151-158`, feeding `envcheck.py:380`, `:477`), and this one
is not in it. Rather than add an envcheck row that would fail for every install with no SAM2 and no
wish for one, preflight at `set_locked_regions` time — the shape
[`motion-transfer-plan.md`](motion-transfer-plan.md) chose for its own case — and use the machinery
that already exists:

```python
spec = reg.get(MATTE_WORKFLOW)                 # KeyError -> ValueError -> 400
problems = reg.validate(spec, object_info=comfy.object_info(backend=spec.backend))
```

`reg.validate` with `object_info` checks both the node class and the model file
(`registry.py:577-584`), and `comfy.object_info` is process-cached (`clients/comfy.py:395`). This
fires only when an incoming region asks for a matte, cannot be skipped by a run that bypassed
preflight, and reports at the moment of the request.

## P1.4 — `probe` learns `pix_fmt` and `color_range` (done; the feared bug is not real)

This section was written to prevent "the quietest bug in the whole plan": an h264 matte stores
limited-range luma, so a white mask lands as 235, alpha tops out at 0.92, and 8 % of the render
ghosts through the middle of the lock — a faintly transparent brand label that reads as a bad
relight rather than a bug. The planned fix was to decode the matte with
`-vf scale=in_range=tv:out_range=pc`, chosen from the file.

**Measured instead of assumed, and the conclusion flipped.** Half the premise is true and half is
not:

| | measured |
|---|---|
| a white mask stored in an untagged yuv420p stream | **235** — the range *is* compressed |
| that file read back with `-pix_fmt gray` | **255** — ffmpeg's yuv→gray expands it itself |
| the same read with `scale=in_range=tv:out_range=pc` | **255** — idempotent, not a second expansion |
| mid-grey: stored / naive / with the filter | 126 / **128** / **128** |

So swscale already treats untagged YUV as limited and `gray` as full, and does the expansion on
the way out. `_lock_tracked` reads exactly that way, so **the naive read is correct and no range
correction is needed.** Worse, a `MediaInfo.full_range` property — which this plan originally
specified — would have invited exactly the correction that is already applied.

What shipped, therefore, is narrower than planned and honest about why: `pix_fmt` and
`color_range` on `MediaInfo`, fed by `pix_fmt,color_range` in `probe`'s `-show_entries` list, as
**description of the file rather than a correction anybody applies**. They would matter only to
code that took the Y plane straight from a `-pix_fmt yuv420p` pipe, which nothing here does. No
`full_range` property.

The test that records this is the valuable artefact
(`test_a_mask_survives_h264_even_though_it_is_stored_limited_range`): it pins all four numbers
above, so nobody "fixes" the non-bug by adding an expansion, and it fails loudly if a future
ffmpeg stops expanding on its own — the case in which masks really would go translucent.

## P1.5 — composite with the matte

**Tracker for placement, matte for shape.** Both, not either. Abandoning the tracker loses three
things a matte cannot supply:

- **Where the source pixels come from and how much to magnify them.** The matte says which *clip*
  pixels belong to the object; it says nothing about the keyframe. On a push-in that mapping is the
  whole job, and the reconstruct-once-at-max-scale fix depends on knowing the per-frame scale ahead
  of time.
- **`_relight`.** `_ring_level` (`media.py:1045`) reads the band *outside* the box and masks the box
  out, so a non-rectangular matte inside it changes nothing and relight keeps working untouched.
  That alone argues for keeping the box a first-class object even for a matte region.
- **`_follow_regions` and its tests** (`tests/test_media_lock.py:59-85`).

**Do not let the tracker resample the matte.** The matte is already frame-sized and frame-aligned,
because SAM2 propagated it against the render's own frames. So the alpha is a plain slice of the
frame-sized matte at the destination window:

```python
alpha = (edge_ramp_y[:, None] * edge_ramp_x[None, :] * matte[y0:y1, x0:x1])[..., None]
```

No `area_resample` of the matte, no cropping by `Placement`. Cropping the matte *using the
tracker's placement* would superimpose two independent motion estimates — NCC similarity search
against SAM2 propagation — and a 2 px disagreement puts a crescent of render inside the lock and a
crescent of keyframe outside it, **on the object's silhouette**, which is the most visible place it
could possibly land. Slicing keeps the alpha boundary exactly where SAM2 found the edge in that
frame. Multiplying rather than replacing the ramp keeps a patch leaving the frame fading correctly.

Signatures and boundaries:

- **`media.lock_regions` stays ComfyUI-free.** The orchestrator runs the workflow and passes a path;
  `media.py` only ever consumes paths. Otherwise `tests/test_media_lock.py` can no longer synthesise
  its own inputs and the module loses its pure-ffmpeg/numpy character.
- `lock_regions(..., mattes: Sequence[Path | None] | None = None)` with an explicit
  `len(mattes) == len(regions)` check — **not** a path hung off `Region`. `Region` is deliberately
  render-size-independent geometry; a cache path is job state.
- `_lock_tracked` gains a third decode pipe at `-pix_fmt gray` (a separate read from the `rgb24`
  frame pipe, whose reshape hardcodes three channels), asserts the matte's dimensions match the
  clip, compares `nb_frames` (which `MediaInfo` already has, and
  `tests/test_media_lock.py:42` already asserts on) and holds the last frame on a short matte rather
  than desyncing silently.
- The `track=False` path is a `drawbox` string (`media.py:978-991`) and **cannot** honour a matte.
  Nothing in production passes it, so raise `ValueError("a matte needs track=True")` rather than
  quietly pasting a rectangle.
- Count matte pixels falling outside the patch window and log them beside the existing `weak`
  count — that is the signal the drawn box is too tight for the object SAM2 found.

## P1.6 — state, API, UI

**`Region` must gain the field, or the feature fails silently.** Both production call sites do
`media.Region(**r)`, `Region` is frozen with exactly `x/y/w/h`, and **`TypeError` is in the
swallowed set** at `orchestrator.py:1011` — so an unrecognised key yields a `warning` Decision, an
unlocked clip and a `200`. Add `matte: bool = False` to the dataclass (`media.py:892-915`);
`__post_init__` and `box()` are unaffected and both call sites keep working verbatim. Extend the
clean-dict whitelist at `orchestrator.py:1035-1036` too, which currently rebuilds each region from
exactly four keys and drops anything else on the floor.

**Persist intent, never a derived path.** The naive `<stem>_matte_r0.mp4` is a stale-matte hole:
region identity in `Scene.locked_regions` is **positional**, so deleting region 0 in the UI makes
the old region 1 become index 0 — and `force=True` does not save you, because force bypasses the
`_locked.mp4` short-circuit (`orchestrator.py:1005-1007`), not a matte. Content-address it instead,
following `versions.py:135`:

```python
digest = sha1(f"{base.name}|{x}|{y}|{w}|{h}|{MATTE_WORKFLOW}|{THRESHOLD}|{DILATE}|{FEATHER}")[:8]
dest = base.with_name(f"{base.stem}_matte_{digest}.mp4")
```

The only new persisted state is then `matte: true` — a declaration of intent — and there is nothing
to invalidate. Keep `threshold`/`dilate`/`feather` out of the UI in phase 1 and fix them as module
constants, so the digest is stable and the UI is one checkbox. The lifecycle falls out:

| event | what happens | stale? |
|---|---|---|
| draw box, animate, `_record_clip` → `_lock_regions` | digest over the base clip + box; SAM2 runs; cached; pasted | no |
| regenerate the clip | ComfyUI's per-prefix counter gives the new render a different stem, so a new digest — the same property today's `_locked.mp4` short-circuit already relies on | no; old files orphaned exactly as today |
| regenerate the keyframe | `_marks_still_fit` false → marks dropped, `unlocked` Decision (`orchestrator.py:343-352`) | no |
| redraw the **same** box | digest hits → SAM2 skipped entirely | correct, and a win positional naming cannot give |
| `set_locked_regions(force=True)` with a nudged box | new digest → new SAM2 run | correct, and the expensive case the UI must warn about |

**Fail loudly on the PUT, degrade on the run.** `_lock_regions` swallowing failure is right during
`_record_clip` — keeping the unlocked render is the better trade — and wrong during
`set_locked_regions`, where the user explicitly asked and a SAM2 failure would return `200` with a
Decision nobody reads. Add `strict: bool = False` to `_lock_regions` and pass `strict=True` from
`set_locked_regions`, re-raising as `ValueError` → `400`, which the endpoint already maps
(`api/jobs.py:353-354`).

**UI.** `schemas.LockedRegion` (`api/schemas.py:212-224`) gains `matte: bool = False`; `make
api-types`; `LockRegions.tsx` gains a per-region toggle carried through `toRegion` (`:16-23`).
The PUT is about to go from seconds to minutes, so: emit a `"locking"` event before the matte run so
`GatePanel`/`events.ts` can say what is happening, keep the save button busy for the whole run, and
check `useSceneLocks` (`frontend/src/hooks/useJobs.ts:266-282`) for a fetch timeout. Note that nothing currently
listens for the existing `"locked"` event either.

Keep `locked_regions` out of both edit allow-lists (`orchestrator.SCENE_EDIT_FIELDS:282-284`,
`queue.py:417-419`) — its only mutation path stays the dedicated method behind `_editable_job`.

## P1.7 — tests

Real ffmpeg, in the style of `tests/test_media_lock.py`: module-level
`skipif(shutil.which("ffmpeg") is None)`, `lavfi`-synthesised clips, 2x2 pixel probes.

**The decisive test.** A red source over a blue clip, and a matte whose shape is a **circle** —
synthesised with numpy piped into one rawvideo encode rather than relying on `geq` being compiled
in. On the `track=True` path, assert:

- a pixel **inside the bounding box but outside the circle** comes from the **clip**;
- a pixel inside the circle comes from the **source**, and assert this one **tightly**, not at the
  existing tolerance of 14 — a loose centre assertion is exactly what would let the tv-range
  ghosting of P1.4 through.

The first of those is the whole feature; it is the one thing a rectangle cannot pass. Plus:
wrong-size matte → `MediaError`; short matte → defined behaviour; `track=False` + matte →
`ValueError`; `tracking.paste(alpha=...)` units in `tests/test_tracking.py`; orchestrator coverage
in the existing fake style (`tests/test_orchestrator.py:728-872` monkeypatches `media.lock_regions`
and records its arguments); a cache-hit/miss test per lifecycle row above; and the existing seven
lock tests unchanged.

---

# Phases 2+ — sketched, not designed

Costed here so the staging is an informed choice, not an omission.

### Persisting RGBA layers

The first alpha-capable container in a pipeline where every encode hardcodes
`-pix_fmt yuv420p -f mp4`. Needs: a choice of container (ProRes 4444 `.mov`, VP9+alpha `.webm`, or
PNG/EXR sequences), `has_alpha` on `MediaInfo` (P1.4 does half the work), a MIME story for `/media`,
and something other than a bare `<video src>` in the UI, because `MediaThumb.tsx:52` cannot play
ProRes.

### A `layer` kind and a per-scene layer stack

`Scene.video_path` is documented at `state.py:115-117` as the *projection* of the active clip "so
everything downstream keeps reading one clip". That sentence is the whole cost. The chokepoints:

- `assembly.clips_of` (`:247`) — `if s.video_path` is the gate for "is this scene in the timeline"
- `assembly._clip_path` (`:420-424`) — `SOURCES = ("final", "raw")` wants a third axis, orthogonal
  to `clip_id`
- `assembly.clip_infos` (`:618-650`) — probes exactly one path per scene and feeds the entire
  timeline UI (`AssemblePanel.tsx` reads `clip?.url`, `clip.duration`, `clip.fps` off it)
- `is_stale` — "a layer changed" and "the composite changed" become different questions

Weeks, and the state-model work is the bulk of it. The registry side is comparatively cheap:
`Contract` has only four fields (`required`, `optional`, `min_image_inputs`, `min_inputs`), and
there is no `min_video_inputs` and no `mask` input type (`INPUT_TYPES = ("image", "video")`,
`registry.py:49`), so a `layer` contract means extending `_contract_problems` the way
[`motion-transfer-plan.md`](motion-transfer-plan.md) proposed for `motion`.

### Export

`export.plan()` (`:74-101`) is a flat `(name, path)` list, and `build_manifest`'s per-scene bucket is
a prefix match — `{name: src for name, src in files.items() if name.startswith(f"s{n:02d}_")}`
(`:167`) — so layer files land in the right scene **for free**. What needs widening: `sources`
(`:168-169`) is a fixed four-key shape (`keyframe/swapped/clip/raw_clip`) and the `seeds` loop
(`:143-146`) a fixed three-tuple. Watch the naming collision against the existing `s01_post*`
namespace, and note `README.txt` (`:189-245`) hardcodes the one-clip, h264 story.

### Layer sources — recorded, not planned

Researched 2026-10-13. Both are genuinely released; neither is on the critical path.

| | **Casper / Generative Omnimatte** | **Wan-Alpha** |
|---|---|---|
| released | code **and** weights, Apache-2.0 | Apache-2.0, with a ComfyUI package |
| base | CogVideoX-Fun-5b, or **Wan2.1-Fun-1.3B** / 14B+LoRA | LoRA + DoRA on Wan2.1-T2V-14B, two VAE decoders (RGB + alpha) |
| resolution | **480x832** for the Wan variants — vorch's exact format | 832x480, 81 frames |
| what it gives | removes an object **together with its shadows and reflections** — a clean plate | generates a *transparent* clip from text |
| reported runtime | 1.3B ~10 min on an A6000 (48 GB); 14B ~55 min | not stated |
| the catch | standalone Python + Gradio, **not** a ComfyUI node; VRAM figures are the machines they used, not minimums — whether 1.3B fits 16 GB is **untested** | **text-to-video only**, so it cannot animate vorch's keyframes |
| where it would fit | a Modal `layer` source for clean plates | generating transparent *elements*, which have no keyframe — off the critical path |

For completeness: the Vera dataset itself is public — `netflix/Vera-Layered-Video-Dataset`,
Apache-2.0, **320 GB**, ~18K training samples (11,996 at 49 frames + 6,005 at 81). That makes
reproducing Vera possible rather than blocked, but it is still weeks of MoT implementation plus
four figures of Modal H100 per run, for a model needing **21.8 GB at inference** — so it would not
run on the 16303 MiB card even once trained. One licensing detail: Apache-2.0 overall, but sources
include DAVIS under **CC BY-NC 4.0**, which matters for commercial work.

---

## The standing conclusion

Phase 1 is worth doing even if phases 2+ never happen, because it fixes a real, current failure with
no new concepts and no new architecture: **a label on a curved bottle is not a rectangle.** P1.0 is
worth doing on its own merits whatever happens to the rest.

And the invariant from [`model-tuning-options.md`](model-tuning-options.md) stands unchanged: no
amount of training will fix brand text in video. That is VAE compression, and `locked_regions` is
already the right answer. This plan only makes it the right shape.
