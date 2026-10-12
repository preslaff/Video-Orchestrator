# Blender as the 3D stage (research, v0)

> Status: researched 2026-10-05; **the five open questions were answered 2026-10-06** and the
> direction below is now decided, though **nothing is implemented yet**. Written from Georgi's Local
> Video Studio package (`PRESLAV_LOCAL_VIDEO_STUDIO_2026-10-05`), from two external sources, and from
> measurements taken on this machine.
>
> **Decided 2026-10-06:** Blender is invoked rather than resident (C and D rejected) · the `.blend`
> files live locally beside vorch · **OpenPose control is kept** — no depth, no Wan 2.2 VACE · **S2V is
> kept** — Wan-Animate-2 motion transfer is parked · photogrammetry (§9) is parked. Consequence: the
> S2V set was installed locally and hash-verified on 2026-10-06 (§10).
>
> **Decided 2026-10-12:** the Blender side is **its own app in its own repository**, handing vorch a
> **shot package** on disk — media plus a `manifest.json` sidecar. vorch never runs Blender, which
> drops `vorch/blender.py` before it was written and makes the GPL/MIT boundary structural (§10,
> decision 6).

## Context

Georgi wants consistent video that profits from a real 3D composition: one `.blend` holding the
set, the character, the lighting and the cameras, so that shot 1, 2 and 3 share a world instead of
being argued into resembling one. He has already run the first pilot (a bar scene, an animated
stand-in, a Klein keyframe, Bulgarian SAPI audio and Wan 2.2 S2V) and **measured it**, which means
the design space is much smaller than it looks.

His machine: Windows 11, RTX 5060 Ti 16 GB, 32 GB RAM, Blender 5.2.2 **and** 4.5.14, a
`blender-mcp` at port 9876 (26 tools, read-only probe recorded), ComfyUI portable 0.28.0, DaVinci
Resolve, WhisperX, OTIO. vorch is on this machine. The two are joined by the **P0 parity contract**
(`parity-contract/schemas/`), whose `ShotCard` is the natural seam — see §4.

### Measured here, 2026-10-05

| | |
|---|---|
| Blender | 5.2.2 LTS at `/opt/blender-5.2.2-linux-x64/…/blender` — **not on `PATH`** |
| Bundled Python | 3.13.13 (vorch is 3.10+; `bpy` is absent from `.venv`, deliberately — see §5) |
| Headless EEVEE, 832×480, one frame, factory startup | **7.04 s, peak 596 MiB VRAM, released on exit** |
| The same, ten frames as a sequence | **1.27 s total — 0.127 s a frame**; the 7.04 s was one-off setup (§8) |
| Subject projection vs ground truth | a 0.23 m head, 85 mm, 832×480: **239.1 px at 1.89 m, 86.1 px at 5.25 m** — Georgi's two measured sizes (§4 ①) |

The 596 MiB is the default scene (a cube). A real set with textures will cost more. **Measure it on
the bar scene before writing a number into the service manifest** — an arbiter that under-reserves
admits work that then runs out of memory (`docs/services.md`).

## 1. What Georgi's measurements already decided

From `sources/S2V_SHOT02_RESULTS_2026-10-05.md`. Same seed (77), same reference, same prompt, one
variable per run — so these comparisons mean something.

**① A Blender OpenPose control video costs identity.** ArcFace to the photo fell to **0.350**
(min 0.141) with control, from **0.486** without. The gesture, meanwhile, obeyed the rig exactly:
the left arm rises in frames ~24–60 because the blocking skeleton does. So control works for
*motion* and is paid for in *face*.

**② Framing beats control by an order of magnitude.** Going from an 86 px face to a 239 px face
(85 mm close camera) took ArcFace from **0.486 → 0.789**. Nothing else in the entire report moved
the number that far. The most valuable thing Blender knows is therefore not an image: it is **how
many pixels the face will be**, computable before a single render is paid for.

**③ That 0.350 is partly a retargeting artifact, not a property of the method.** The report is
explicit: the reference subject's keypoints were placed **by hand** (`anchor_kp_ref.json`,
`MANUAL_ESTIMATE`). The Blender skeleton is geometrically exact; its correspondence to the
reference frame was not. A **depth pass needs no retargeting at all** — it is exact by
construction. And depth is the one thing the pilot did not try: *"RGB/depth превиз и локална сцена
с много говорещи персонажи не са част от тези измервания."*

Two further facts worth carrying: S2V held identity about **2× better than I2V** (0.486 vs 0.242),
and **lip-sync is unconfirmed, not absent** — the RMS-correlation criterion was invalidated against
a 500-sample null distribution, so `r ≥ 0.3` was retired. Blender does not fix lip-sync. A natural
Bulgarian voice and a tighter frame do. Keep the two problems apart.

## 2. What the external sources add

### flick.art, *Blender AI filmmaking* — independent confirmation of the two-part shape

The article's workflow is: animate a rough blockout and camera in 3D, export **a blockout video as
the motion reference** and **the first frame as the composition reference**, then generate. Its
stated principle is that Blender output is **instruction, not visual asset** — *"less detail is
often better"*, *"a clean, straightforward blockout lets it retain the motion"*, and too much detail
confuses the model.

That is the same two-part structure §4 arrives at from Georgi's numbers, reached independently: a
driving video for movement, a locked frame for composition. It also lists the local control route
(depth, OpenPose, canny, normals, segmentation into ControlNet) and names two things we did not have:

- **Wan 2.2 VACE** accepts depth and pose control video frame by frame. Georgi's inventory says
  plainly: *"В прочетения ComfyUI model каталог на G няма Wan VACE или пълни Fun-Control high/low
  модели."* **The model that would make depth control work is not installed.** If depth control is
  the experiment, this download is a prerequisite, not a detail.
- **toyxyz's "Character bones that look like OpenPose for Blender"** emits several passes from one
  render. Worth evaluating before writing our own pose/depth exporter.

It also quotes 16 GB VRAM for images and **24 GB+ for video models**, which is a useful outside view
on how tight this card is.

### Blender Foundation, *Guidelines for AI-enabled features* (devtalk, 2026-06-02, fsiddi)

Reads as policy, not draft. Two mandatory criteria for any AI feature:

1. **Local & sovereign** — *"The model can run locally, or be deployed anywhere by the user without
   restrictions."* Copyleft-friendly model licences preferred.
2. **IP conscious** — *"The model should be built on lawfully licensed content, respecting
   intellectual property."*

And the line that settles one question: *"There are no plans to develop or implement 'generative' AI
features (e.g. prompt-based artwork generation, mesh generation, etc.)"* Allowed examples are
assistive — denoising, upscaling, masking, pose detection, mocap, transcription, skinning.

**What this means for us.** The guidelines govern what *Blender itself* ships, not what an add-on
may do, so they do not forbid anything we want to build. But they are a clear signal about
`extensions.blender.org`: a generative add-on is not what that platform is cultivating. Treat
publication there as out of scope until someone asks for it. Internally it changes nothing.

Worth noting in the other direction: vorch already satisfies criterion 1 better than most of the
ecosystem — it runs entirely locally on open models, which is the stance this document is asking
for. That is a point in its favour, not an obstacle.

### Not read

`reddit.com/r/StableDiffusion/comments/1hutxwq/` (FLUX + Blender + Python scripts) could not be
fetched — both `www.` and `old.` hosts are blocked from this environment, and a search did not
recover it. It likely overlaps heavily with the flick.art material (Blender depth → ControlNet →
FLUX, scripted). **Nothing in this document rests on it.** If it matters, paste the text.

## 3. Architecture: four options, two acceptable

| | mechanism | for | against | verdict |
|---|---|---|---|---|
| **A. Blender as an asset producer** | vorch runs `blender -b shot.blend -P previz.py -- --shot S01`; output lands as project assets | no resident cost, deterministic, dies clean, no add-on needed | no interactivity; vorch must reach the `.blend` | **the spine** |
| **B. Blender add-on that pushes to vorch** | a "Send shot to vorch" operator POSTs a ShotCard + renders to `/api` | the artist leads, which is how Georgi actually works | reverses control; needs an API entry point for an externally composed job | **the interface** |
| **C. Blender resident as a service** (the `blender-mcp` shape, port 9876) | vorch talks to a live Blender over a socket | symmetry with ComfyUI/Gemma; Georgi already has it running | holds GPU and a process; `bpy` is not reentrant; precisely the class of service `docs/services.md` says vorch will never manage | **no** |
| **D. `bpy` as a library inside `.venv`** | `pip install bpy` in the API process | simplest code | pins Python to 3.13, ~1 GB wheel, and one segfault in `bpy` takes FastAPI down with it | **no** |

**Decided 2026-10-06: never C, never D.** **Superseded in part on 2026-10-12** (§10, decision 6): the
Blender side becomes its own application in its own repository, and vorch **never runs Blender at
all** — it reads a shot package off disk. So option A survives as "Blender produces assets", but its
subprocess wrapper does not: there is no Blender path in vorch's config and no `vorch/blender.py`.
Option B (an add-on that POSTs to the API) is moot for the same reason — a folder is the handoff.

What the table above still settles is C and D. A resident Blender holds a process and the card for no
benefit, and `bpy` inside the venv would put a non-reentrant library in the API process **and** pull a
GPL dependency into an MIT codebase. Two repositories make that boundary structural rather than a
convention someone has to remember.

The shared vocabulary is `ShotCard` (`parity-contract/schemas/shot-card.schema.json`):
`camera_movement`, `camera_speed`, `framing`, `angle`, `lens_fov`, `start_end`, `duration_fps`,
`identity_refs`, `resource_refs`. That is very nearly a description of a Blender camera, which is not a
coincidence — it is what the shot package's manifest should align its field names to, so the two
contracts do not drift.

## 4. What Blender should supply, ranked by measured payoff

Ranked by what has actually been measured, not by what sounds most capable. The first two items
address the project's two known generation failures; the last two change what downstream code has
to do at all.

### ① The subject pixel budget, before any generation — largest payoff, zero cost

Blender knows the lens, the camera's distance to any object and the render resolution, so it knows
**how many pixels a subject will get in the output** before anything is rendered. This is a **gate,
not a render**, and it lands on machinery vorch already has: `envcheck` blocks submit on failure, and
a render that cannot fit is refused by name rather than left to OOM. A shot whose subject comes out
too small should be refused the same way. One `blender -b` per job, seconds, no VRAM held.

**It is not a face gate.** Faces are one kind of subject; printed labels, logos and product detail
are others, and the project's two known generation failures are one failure each from the first two
kinds. The thresholds, with what is actually measured marked as such:

| kind | threshold | source |
|---|---|---|
| face, for identity | 180–250 px | **measured** — 86 px gave ArcFace 0.486, 239 px gave 0.789 (SHOT_02) |
| printed text | ≥ 32 px cap height | **measured** — `docs/label-fidelity.md` |
| anything at all | 8 px linear | the VAE's 8× spatial encode: under one latent cell there is no representation |
| logo, product detail | unmeasured | no run has established these; do not invent one in code |

**Verified 2026-10-05 against ground truth.** The projection is four API calls — `matrix_world`,
`bound_box`, `bpy_extras.object_utils.world_to_camera_view`, `resolution_y` — and run on a 0.23 m
head at 832×480 it returns **239.1 px at 1.89 m** and **86.1 px at 5.25 m** with an 85 mm lens:
Georgi's two measured face sizes, from geometry alone. The same function on a 11 mm label at 0.6 m
and 960×960 returns 24.5 px and refuses it, and on a 3 mm subline 6.7 px — the same verdict the
Perfume Echoes job earned by rendering it. So §8's acceptance test for this gate is already passed in
principle; what remains is reading real geometry instead of a stand-in primitive.

Two caveats before it becomes a gate. A bounding box is the wrong measure for most subjects — a
bottle's bbox is mostly air, and what must survive is the **cap height of the glyphs**, not the
extent of the label object. And Georgi's 86 px was a bounding box measured on a *rendered* face,
which is not exactly a head-bone projection: expect agreement within a few percent, not to the pixel.

**So the subject is tagged in Blender, not guessed by vorch.** Each thing that must hold up gets its
own marker — a plane or an empty sized to exactly the detail at risk — in one collection, carrying a
custom property:

```
Collection "vorch_subjects"
  ├─ face_georgi     vorch_kind = "face"
  ├─ label_flower    vorch_kind = "text"
  └─ logo_bag        vorch_kind = "logo"
```

Custom properties are set in Blender's Object Properties panel with no scripting, so Georgi tags what
matters and `previz.py` only walks the collection. He decides; vorch measures.

### ② A driving video for motion transfer — **parked 2026-10-06** (S2V kept instead)

`docs/motion-transfer-plan.md` (planned 2026-09-05, **not started**) specifies `Scene.driving_video`:
a project asset of `kind: video` supplies movement to Wan-Animate-2 while identity comes from the
scene's keyframe. Asset `kind: video` is already validated (`vorch/projects.py:233`).

**A Blender previz render is exactly such a driving video** — and this decouples the two things that
fought each other in SHOT_02, because Animate-2 takes motion from a video and identity from an image
by design, rather than asking one S2V pass for both. The plan was written for phone-shot reference
footage; Blender is the same input, directed. Note Animate-2 is deployed on Modal, i.e. paid.

### ③ OpenPose kept — and the retargeting problem dissolves

**Decided 2026-10-06: keep OpenPose, drop depth and Wan 2.2 VACE.** No 15 GB download.

That looked like it needed a pose detector to replace the hand-placed anchors — and it does not, if
**the keyframe is generated from the previz.** Render the Blender blocking, make the Klein keyframe
from that frame in edit mode, and the reference image's pose *is* the rig's pose. The 2D control
keypoints then come from projecting the same armature — the identical four API calls as ①.

```
Blender blocking ─┬─► previz frame ─► Klein edit-mode keyframe   (composition locked to the rig)
                 └─► projected 2D keypoints ─► OpenPose control
                                              └─► S2V   ← every correspondence exact by construction
```

No detector, no `MANUAL_ESTIMATE`, no retargeting step at all. Note this is the one place where
`identity_edit` **locking composition — normally the reason invariant 6 warns against it — is exactly
what is wanted**, because the composition should be pinned to the blocking.

The registry already expresses the second image input with no Python: `identity_edit.yaml` declares
`inputs: {reference: image, reference_2: image}`, and a control input is the same shape.

### ④ Consistency — the actual answer to the original question

Not per-frame control. **One source of truth**: the same `.blend`, the same set, the same lighting,
the same three cameras across all three shots, feeding the same composition into `mode: edit` as a
reference. The composition is locked *because it is 3D*, not because a model was persuaded. This is
also what flick.art means by composition lock, and it is the part that scales to a permanent studio.

### ⑤ Locked regions become analytic instead of tracked

The projection in ① returns the subject's **rectangle in frame fractions** — which is exactly the
shape of `Scene.locked_regions` (`x/y/w/h` as fractions, so they survive any render size).

Today that rectangle is drawn by hand on the keyframe, and `vorch/tracking.py` then finds it again in
every frame: a similarity search over uniform scale plus translation, a few hundred normalised
correlations per frame, seeded from the previous frame. It works — verified through a 3× push-in —
but it is a *search*, with a template cut from the clip's own first frame and three separate
anti-flicker fixes found by measuring a track that flickered.

For a Blender-sourced shot the camera path and the object are both known in 3D, so **the rectangle's
position in every frame is analytic**. No template, no correlation, no seed, no drift, nothing to
flicker. The tracker does not get better — it becomes unnecessary for these shots, while remaining
the only option for footage that came from anywhere else.

This is a stronger argument for the Blender route than the gate is, and it is worth validating early:
the same push-in clip that the tracker was verified on is the obvious comparison.

### ⑥ The other direction: an approved keyframe into a 2.5D scene

Everything above runs Blender → keyframe. The reverse also works, for a narrower purpose: take a
keyframe you have already approved, estimate monocular depth (Depth Anything v2, Marigold), displace
a subdivided plane by it, and put a Blender camera in front of the result. This is camera projection,
standard in VFX for decades — not research. What it yields is a **relief carved from one viewpoint**,
not a model of the room.

| what the 3D was for | survives a keyframe reconstruction? |
|---|---|
| camera motion with real parallax | **yes**, within a limited move budget |
| composition lock | yes, trivially — it came from the image |
| analytic `locked_regions` (⑤) | **moot, which is better than exact** — see below |
| the subject pixel budget (①) | **broken.** Monocular depth is relative, not metric; nothing is 0.23 m tall unless the scale is anchored on an assumed subject size |
| character performance — the arm rising on the rig | **no.** A depth mesh is a frozen statue and cannot be re-posed |

That last row is why this does not replace Georgi's rigged-character work. It replaces set and camera
work.

**Where it pays off is a shot that only needs a camera move** — an establishing shot, a product
reveal, B-roll, a push-in on a label. There the video model can be skipped entirely:

```
keyframe → depth → displaced mesh → Blender camera move → render
```

At the measured 0.127 s a frame, 77 frames is **~10 s of rendering** against **8–13 min** for a local
Wan clip (that rate was a trivial scene; a dense displaced mesh with a texture will be slower, still
seconds not minutes). And three problems disappear rather than being managed: **the label is intact
because the pixels are the original pixels** — projected, not generated, so neither `locked_regions`
nor the tracker nor the 32 px threshold applies; nothing resamples the face, so identity cannot
drift; and there is no 4n+1 rule, no 22 GB model load and no VRAM admission fight.

**The limit is disocclusion.** Move the camera and pixels hidden behind foreground objects become
visible with no data behind them — stretched smears at silhouette edges. This sets the move budget:
push-ins are safest (they reveal almost nothing), lateral dollies worst, an orbit impossible. Expect
single-digit-percent camera displacement relative to scene depth, not the 3× push the tracked label
survived.

Which suggests the strongest form: **render the 2.5D move in Blender, then hand it to Wan or LTX as a
control or low-strength pass, so the model only fills the holes instead of inventing the motion.**
Geometry owns the camera; the model owns the gaps. The registry already expresses a second image
input with no Python (③), so that path is open.

**Do not expect the two directions to compose into a loop.** A keyframe-derived scene cannot tell you
the keyframe was badly framed — it inherited the framing. ① belongs before generation; ⑥ belongs
after approval.

Cost to try: one depth model (~1–2 GB, local, fast, no licence friction — nothing is installed today,
`geometry_estimation/` is an empty placeholder) and a `previz.py` variant that builds the mesh and
animates a camera. Smaller than §8's VACE download, and it tests independently of everything there.

## 5. Licensing — and why it reinforces A/B

The Blender Foundation treats an add-on that imports `bpy` as a derivative work of Blender, so **an
add-on must be GPL**. vorch is MIT as of 2026-10-04.

This is not a problem; it is an argument. MIT is GPL-compatible in that direction, so **a GPL
add-on beside an MIT core works — provided the boundary is a process, not a link.** Options A
(subprocess) and B (HTTP) preserve that boundary by construction. Option D breaks it: `bpy` inside
vorch's venv pulls Blender into an MIT project's process and opens a question nobody needs.

The right architecture and the right licence point the same way.

## 6. VRAM: Blender is a phase, but a cheap one

596 MiB peak, released on exit, against OmniVoice's 5726 → 11268 MiB — and 0.127 s a frame once
the process is up (§8). Therefore:

- Blender is **not** a new invariant of the narration kind.
- It is **not** free beside a loaded Wan either: at ~12000 MiB for a render plus 1838 for Gemma,
  596 MiB counts.
- The rule follows the existing phasing: **render the previz for every shot at once, before the
  keyframe phase.** Blender exits, then ComfyUI loads. Same shape as "all keyframes, then all
  clips" (`NOTES.md` §5).

**Decision 6 moves who enforces that.** Since vorch no longer invokes Blender, it cannot sequence it
either — the phasing becomes the operator's discipline: export the package, close Blender, then
submit. vorch's side of it is unchanged and already built: a step declares what it needs, VRAM is
arbitrated, `/setup/services` attributes the card per process including processes vorch cannot name,
and a render that will not fit is refused by name. So a Blender left open does not corrupt anything —
it shows up as foreign VRAM and can cost you a refused render. That is the arbitration working, with
a cause that lives in another window.

## 7. Smallest first step

> Revised 2026-10-12 for decision 6. The work now splits across two repositories; the first step on
> each is small and they can proceed independently.

**In the Blender app** (its own repo): the exporter, shipping the manifest before the media.

```
previz.py / the add-on — per camera or marker range, emit a shot package:
   manifest.json          lens, sensor, resolution, fps, frame range, and per tagged
                          subject: kind, projected px, frame rect   ← ship this first
   s00_first_frame.png    the composition reference
   s00_previz.mp4         EEVEE at the profile's resolution and fps
   s00_control_*.mp4      OpenPose keypoints projected from the same armature
   s00_regions.ndjson     one rect per frame, for locked_regions
```

**In vorch**: a package importer plus the budget check. No Blender dependency of any kind.

Ship the **manifest** first, on both sides. One projection answers both of the project's known
generation failures — a face too small to hold identity, and type too small to survive a video VAE —
and the same rectangle is the `locked_regions` path (§4 ⑤). It needs no renders, no GPU and no models,
so it is testable on either side alone.

Then the experiment, under Georgi's own rule — *едно променено нещо на пуск, същият seed и ref*:
SHOT_02, seed 77, same audio, **OpenPose control with the keyframe generated from the previz** so the
pose correspondence is exact by construction (§4 ③). Falsifiable: if 0.350 was retargeting, this
should land at ≥ 0.486 with the gesture still obeyed. If it falls again, control costs identity on
this model and S2V should run uncontrolled while Blender supplies framing instead.

## 8. Timing plan for the experiment

> Measured on this machine 2026-10-05, and from Georgi's own run times. Effort is in working
> hours; no calendar dates, because availability on either machine is unknown.

### What previz actually costs — measured

A single frame at 832×480, factory startup, took **7.04 s**, but that figure is almost all
one-off: ten frames of the same scene render in **1.27 s total**, i.e. **0.127 s per frame**, and
the whole process including Blender's startup finished in 2.25 s. The first number was paying for
EEVEE shader compilation and CUDA context setup, which the shader cache then keeps.

So a 77-frame previz of that scene is **~10 s of rendering**, not nine minutes. Both numbers are the
default cube scene; a textured bar with real lighting will cost more per frame, and that is the one
previz number still to measure. **Previz is not the bottleneck and should never be scheduled as
one** — an S2V run is 100× its cost.

### Two tracks, different machines, genuinely parallel

| | track A — the subject budget | track B — the control experiment |
|---|---|---|
| repo | the Blender app + vorch's importer | vorch |
| blocked on | nothing | the `.blend` arriving |
| GPU needed | none | yes |
| gives | a pre-generation refusal | whether control costs identity |

They do not queue behind each other. Track A can start now; track B needs the `.blend`.

> **Revised 2026-10-06.** Both tracks run on this machine. Track B lost its depth/VACE branch
> entirely, and its pose-detector dependency dissolved — see §4 ③. The S2V set is installed (§10).
>
> **Revised 2026-10-12** for decision 6: track A now spans two repositories, and A2 changed from a
> subprocess wrapper to a package importer. Same effort, less coupling.

### Track A — the subject budget (no GPU, no models, nothing from Georgi)

| step | work | effort |
|---|---|---|
| A1 | **Blender app:** walk the `vorch_subjects` collection, emit `manifest.json` — lens, sensor, resolution, fps, frame range, and per subject its `kind`, projected px and frame rect | 3 h |
| A2 | **vorch:** read a shot package — install the first frame as the scene's keyframe, the previz as `driving_video`, the rects as `locked_regions`. No Blender dependency | 3 h |
| A3 | Wire the budget into submit as a refusal with a reason, following the `envcheck` pattern, + tests | 4 h |
| A4 | Measure the per-frame previz rate on a real textured set | 30 min |
| A5 | Emit the per-frame rect as `locked_regions` and compare against the tracker on the verified push-in clip (§4 ⑤) | 4 h |

**The acceptance test is already passed in principle, and it cost nothing.** The projection was run
on 2026-10-05 and returns **239.1 px** and **86.1 px** for an 85 mm lens at 1.89 m and 5.25 m —
Georgi's two measured face sizes, from geometry alone (§4 ①). A1 is therefore smaller than it looked:
the maths is settled, what remains is reading real geometry instead of a stand-in primitive and
emitting the JSON. If a real-geometry run disagrees with 86 and 239 by more than a few percent, the
exporter is wrong — not the report.

Roughly **a day and a half**, and nothing in it can fail in a way that costs GPU time.

### Track B — the control experiment

Georgi's rule holds: *едно променено нещо на пуск, същият seed и ref.* So the runs are serial.

Hypothesis ③ (§1) says the 0.350 came from hand-placed keypoints, not from control as such. The
retargeting is fixed **by construction** rather than by a detector: the keyframe is generated from the
previz, so the reference pose is the rig's pose and the control keypoints are a projection of the same
armature (§4 ③). One variable changes against `base_control`, and no new diffusion model is needed.

| step | work | wall clock |
|---|---|---|
| B1 | Register an S2V workflow with a YAML sidecar (control video + audio + reference) | 2 h |
| B2 | Previz frame → Klein edit-mode keyframe, so the reference pose is the rig's | 2 h |
| B3 | Project the armature to 2D OpenPose keypoints → control video. Reuses ①'s projection | 2 h |
| B4 | **Run:** S2V, OpenPose control, seed 77, 20/6, same audio | **~21 min** |
| B5 | Metrics: ArcFace + the null distribution already built in `23_face_metrics.py` | minutes |

Read-out, falsifiable: **≥ 0.486 with the gesture still obeyed** means retargeting was the problem and
Blender control is sound. **Still ~0.35** means control genuinely costs identity on this model, and the
honest conclusion is that S2V should run without control while Blender supplies framing and
composition instead — ①, ④ and ⑤ all stand regardless of how this lands.

**No B2 branch.** Depth and Wan 2.2 VACE were dropped on 2026-10-06, so the plan's largest unknown —
a download of unknown size — is gone.

**Prerequisites, both settled or in hand:** the S2V set is installed locally (§10), and the `.blend`
with the bar and the rigged character has to arrive from Georgi's `D:` once.

A faster variant exists if a run must be cheap: 4 steps / cfg 1 + LoRA was **150–711 s** rather than
~21 min, and scored *higher* on identity (0.508–0.545). It is a different sampler regime, so it
cannot be compared against the 20/6 baseline — use it to iterate, then confirm once at 20/6.

### Critical path and total

```
A1 → A2 → A3 ──────────────── gate shipped        ~1.5 days, no GPU
B1 → B2 → B3 → B4 → B5 ────── hypothesis answered ~6 h setup + 21 min GPU, needs the .blend
                └─ if inconclusive → B2a…B2e      + unknown download + ~2 h + 21 min GPU
```

**Total GPU time for the whole experiment is under an hour.** Everything else is setup. The schedule's
former largest risk — a VACE download of unknown size — was removed by the 2026-10-06 decisions; what
remains is a one-off dependency on the `.blend` arriving.

### What not to schedule

- **Lip-sync.** Unconfirmed, and its metric was invalidated against a null distribution. It is a
  voice-and-framing problem, not a Blender problem (§1). Keep it out of this plan entirely.
- **A resident Blender.** §3 option C. No step here needs one.
- **Previz as a phase to budget.** 0.127 s per frame. Render it, let Blender exit, move on.

## 9. Asset preparation: a real place into Blender (COLMAP) — **parked 2026-10-06**

> Status: **parked by decision on 2026-10-06** — the set will be modelled rather than photographed.
> Kept here because the reasoning does not expire and the permanent studio may want it later.
>
> COLMAP is a **source checkout, not an installation** — `~/WORK/colmap`, `3.12.0-621-g5e548461`, no
> binary under `build/`, no `pycolmap` in either Python. Licence is **new BSD**, so no friction with
> vorch being MIT.

§4 ⑥ answers "keyframe → scene". COLMAP answers a different and better-founded question: **real place
→ scene.** Photograph a location or a product from several positions and Structure-from-Motion
recovers camera intrinsics, camera poses and geometry. For a permanent studio — which Georgi's plan
wants — this is a stronger foundation than anything generated.

**One keyframe cannot be used.** The docs are explicit: *"Make sure that each object is seen in at
least 3 images – the more images the better"*, and single-image reconstruction is unsupported. SfM
matches features between views taken from different positions; one image has no second view.

**And generating several views to feed it does not work.** Generative models are not multi-view
consistent: ask for the same bar from three angles and the bottle count, ceiling height and wall
positions all move. SIFT either finds no matches or triangulates geometry that does not exist. This is
why the field built multi-view-consistent 3D generators (Zero123, TRELLIS, Hunyuan3D) instead of
pointing SfM at generated images. Do not spend a day on it.

### Why a rough reconstruction is enough

Counterintuitively, the ugliness of a photogrammetric mesh does not matter for what we need Blender
for. **A rough proxy with real scale beats a beautiful model with guessed scale**: the subject pixel
budget (①) is only as honest as the geometry's proportions, occlusion needs real volumes, camera moves
need real parallax, and the locked-region rectangles (⑤) need the object's real position. None of that
needs render quality.

| stage | output | useful for |
|---|---|---|
| sparse (`mapper`) | intrinsics, camera poses, sparse point cloud | **often enough** — model a clean proxy over the points, recover the real camera path |
| dense (`patch_match_stereo` → `stereo_fusion`) | per-view depth/normal maps, `fused.ply` | denser proxy; depth ground truth for ⑥ |
| mesh (`poisson_mesher`) | `meshed-poisson.ply` | straight into Blender as set geometry |

**Start with sparse.** Dense is the GPU-expensive stage and may not be needed: a point cloud you model
two boxes and a plane over is easier to block against than a four-million-triangle mesh with holes.

### Interiors are the harder case

The documented capture requirements read like a list of things interiors violate:

- **blank walls** — no texture, no features, no geometry, holes exactly where a wall was wanted;
- **windows** — blown highlights break matching, and the docs warn against high dynamic range;
- **mirrors and glass** — reflections produce confident *wrong* matches, worse than none;
- **repeated patterns** — tiles, identical chairs, regular panelling cause false correspondences;
- **tight spaces** — hard to get a wide baseline, and *"rotate position, not just camera orientation"*
  is harder indoors.

Exteriors are more forgiving on all five. A textured bar you can walk around in will work; a
white-walled room with a window means modelling over a sparse cloud rather than getting a usable mesh.

### The shoot

- **3+ views of every object**, generous overlap between consecutive shots.
- **Move your feet, don't pan.** Panning from one spot gives no parallax; initialisation may fail.
- **Lock exposure and white balance.** Auto-everything breaks the "similar illumination" requirement.
- **Stills over video.** Frames work, but rolling shutter and motion blur cost matches; for video use
  the sequential matcher.
- **Put a known length in the scene** — tape measure, metre stick, A4 sheet. Reconstruction is in
  arbitrary units, so one measured object scales the whole thing to metres in Blender. **Without
  metres the pixel budget gate is meaningless**, so this is the step not to skip.
- Shoot a closed loop and return to the starting view.

### Rough command path

```
colmap automatic_reconstructor --workspace_path ws --image_path ws/images
# or step by step, with the knobs that matter on a 16 GB card:
colmap feature_extractor  --SiftExtraction.max_image_size 1600 …
colmap exhaustive_matcher …                                  # requires a GPU
colmap mapper …                                              # sparse: minutes
colmap patch_match_stereo --PatchMatchStereo.max_image_size 1200   # dense: the expensive part
colmap stereo_fusion … && colmap poisson_mesher …
```

Into Blender: PLY imports natively. For cameras, SBCV's photogrammetry importer add-on reads COLMAP
models including poses — **check it against 5.2 before relying on it**; otherwise the text model
(`cameras.txt`, `images.txt`) is simple enough to read with a short `bpy` script, the same small API
surface as the gate.

### Where it belongs, and what it costs

**Asset preparation, not a pipeline stage.** Run it once per location or product, by hand; the output
becomes a project asset (a `.blend`, or a camera path). vorch consumes that and never invokes COLMAP
per job. That placement matters because:

- **Feature matching requires a GPU**, and dense reconstruction is VRAM-hungry — the docs warn it may
  crash without sufficient VRAM. On 16303 MiB that makes dense MVS another **exclusive** heavy phase,
  the same shape as the narration server. Offline, it needs no admission control at all; inside a job
  it would.
- **It is not built.** A CUDA build via vcpkg is hours of its own, and a CPU-only build will not get
  far given the matching requirement. Decide which of the three uses you want before starting it.
- **Scale is arbitrary** without calibrated intrinsics — hence the ruler in the shoot list.

### The three uses, ranked

| use | what it gives | why it beats the alternative |
|---|---|---|
| **A real place as the permanent studio** | real geometry, real proportions | the pixel budget gate becomes trustworthy; the set is reusable across jobs |
| **Matchmoving real footage** | intrinsics + a real camera path + sparse cloud from a phone clip | organic handheld motion is tedious to keyframe; here it is recovered, and then drives previz |
| **Ground truth for ⑥** | multi-view-consistent MVS depth | tells you whether monocular relative depth is good enough for a camera move, instead of guessing |

Honest sequence: build it, shoot one test location **with a scale reference**, run sparse only, get it
into Blender, and judge whether the proxy is good enough before investing in dense or in tooling.

## 10. Decisions

The five open questions were answered on 2026-10-06; a sixth decision followed on 2026-10-12.

| # | question | answer | consequence |
|---|---|---|---|
| 1 | Blender resident, or invoked? | **invoked, not resident** | C and D rejected (§3); refined by decision 6 |
| 2 | OpenPose, or depth via VACE? | **keep OpenPose** | no 15 GB download; retargeting solved by construction (§4 ③) |
| 3 | where do the `.blend` files live? | **locally, beside vorch** | no cross-machine transfer per job — but the bar/rig `.blend` must arrive from Georgi's `D:` once |
| 4 | S2V or Wan-Animate-2? | **keep S2V** | §4 ② parked; no Modal cost; Blender supplies a *control* video, not a driving video |
| 5 | photograph a real location? | **parked for now** | §9 parked; the set gets modelled |
| 6 | where does the Blender tooling live? | **its own app, its own repository**, handing vorch a shot package on disk | vorch never runs Blender; `vorch/blender.py` is dropped before it was written |

### Decision 6 — a separate app, and a shot package as the contract (2026-10-12)

The Blender side becomes its own application in its own directory and repository. vorch does not
invoke it, does not know its path, and has no Blender dependency: it reads a folder.

**Why this is better than the subprocess wrapper it replaces.** The GPL/MIT boundary stops being a
convention and becomes two repositories — the Blender app imports `bpy` and is therefore GPL, while
vorch stays MIT and only reads files (§5). Each side is testable without the other. And vorch's
surface shrinks rather than grows.

**The interface is a package, not two files.** "First frame plus clip" was the starting proposal and
it covers composition lock and motion reference, but it silently drops the two highest-value findings
in this document, because both are *data* rather than pixels: the subject pixel budget (§4 ①) needs
the lens, the resolution and the subject's projected size, and the analytic `locked_regions` (§4 ⑤)
needs the rectangle's path per frame. Keeping OpenPose (decision 2) also requires a third media file.
So:

```
shot-package/
  manifest.json                 what the pixels cannot carry
  s00_first_frame.png           composition reference / the installed keyframe
  s00_previz.mp4                motion reference
  s00_control_openpose.mp4      control for S2V, projected from the same armature
  s00_regions.ndjson            one rect per frame → locked_regions
```

```json
{ "format":   "vorch-shot-package/1",
  "scene_id": "S00",
  "camera":   { "lens_mm": 85, "sensor_mm": 36, "resolution": [832, 480], "fps": 16, "frames": 77 },
  "subjects": [ { "name": "face_georgi",  "kind": "face", "px_h": 239.1, "rect": [0.40, 0.25, 0.20, 0.50] },
                { "name": "label_flower", "kind": "text", "px_h": 24.5,  "rect": [0.44, 0.49, 0.12, 0.03] } ],
  "source":   { "blend": "bar_v07.blend", "sha256": "…", "exporter": "0.1.0" } }
```

This mirrors two patterns the codebase already uses: the workflow registry is a template plus a YAML
sidecar that is the executable source of truth (`docs/workflows.md`), and `vorch/export.py` hands off
a folder of numbered media plus `manifest.json` and `provenance/`. An **import** package shaped like
the export package is idiomatic here rather than novel.

Three things the sidecar buys beyond the media: the budget gate works without vorch ever opening
Blender; the locked regions stay analytic instead of tracked; and a clip is traceable to the `.blend`
that produced it, which is the provenance discipline vorch already keeps for seeds and prompts.

Field names should align with `ShotCard` (`parity-contract/schemas/shot-card.schema.json`) so the two
contracts do not drift — it already carries `lens_fov`, `framing`, `camera_movement`, `duration_fps`.

**Unsettled:** the app's name. `format: "vorch-shot-package/1"` is a placeholder too — it is the
actual contract between two repositories, so it wants a version field from the first commit, which
is cheap now and painful to retrofit.

### What was installed as a consequence

Keeping S2V while running everything locally required the model set, which was absent here. Fetched
2026-10-06 from the URLs in Georgi's own download receipt and **verified against his SHA-256**, so the
two machines run byte-identical weights:

| file | bytes | into | status |
|---|---|---|---|
| `wan2.2_s2v_14B_fp8_scaled.safetensors` | 16394832474 | `models/diffusion_models/` | **verified** |
| `wav2vec2_large_english_fp16.safetensors` | 630997322 | `models/audio_encoders/` | **verified** |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | 1226977424 | `models/loras/` | **verified** |

All three match the receipt's SHA-256, so the two machines run byte-identical weights. The LoRA
matters only for the fast 4-step variants (Georgi's `fast_bg` / `fast_en`), not for the 20/6 baseline.
Observed throughput on this link was about **2.4 MB/s** and it dropped twice mid-transfer, which is
worth remembering before committing to the 11 GB fp16 text encoder.

Already present and reused: `wan_2.1_vae.safetensors`, and a umt5 text encoder. **Note a parity
caveat:** Georgi runs `umt5_xxl_fp16` (11.4 GB); this machine has fp8-scaled and a Q5_K_M GGUF. That
is a different text-encoder precision, so expect small numeric differences against his baseline before
blaming anything else. Pulling the fp16 encoder is 11 GB and only worth it if the baseline fails to
reproduce.

ComfyUI here is 0.34.0 (Georgi: 0.28.0) and ships the S2V nodes natively — `comfy_extras/nodes_wan.py`
and `comfy_extras/nodes_audio_encoder.py` — so no custom node is needed.

### Still open

1. **The `.blend` transfer.** The bar, the rigged character and SHOT_02's blocking are on Georgi's
   `D:`. Everything in track B waits on that one copy.
2. **Three unmeasured timings** that are not Blender's fault but sit in the same critical path: Gemma's
   storyboard latency on CPU, one vision verdict, and a local Klein keyframe.

## Sources

- `~/Downloads/PRESLAV_LOCAL_VIDEO_STUDIO_2026-10-05/` — Georgi's package: the master document,
  `sources/S2V_SHOT02_RESULTS_2026-10-05.md`, `parity-contract/schemas/`, `data/model-inventory.json`
- https://flick.art/blog/blender-ai-filmmaking — read 2026-10-05
- https://devtalk.blender.org/t/guidelines-for-ai-enabled-features/45354 — read 2026-10-05
- https://colmap.readthedocs.io/en/latest/tutorial.html — read 2026-10-05 (§9)
- `reddit.com/r/StableDiffusion/comments/1hutxwq/` — **not read**, inaccessible from this environment
- Local measurements: Blender 5.2.2 headless EEVEE and the subject projection, 2026-10-05
- Local inventory: `~/WORK/colmap` (checkout, unbuilt), `~/comfy/ComfyUI/models/geometry_estimation`
  (empty) — checked 2026-10-05
