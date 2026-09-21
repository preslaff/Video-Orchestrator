# Video Orchestrator (vorch)

Local, self-contained AI raw-material generator for **cinematic and marketing video**. It
turns a text idea into consistent keyframes and clips in the format you need (16:9, 9:16, 1:1,
stills, longer clips), reviews every result with a vision model, and hands you a clean folder of
raw material plus an optional first cut — all on your own machine. Final editing (audio, titles,
colour) happens in an NLE such as Shotcut; this app prepares what goes into it.

Two local systems do the work, driven by a deterministic Python core:

- **Gemma 4** (`gemma-4-26B-A4B`, multimodal, via llama.cpp) — the "brain + eyes": writes the
  storyboard and prompts, and **evaluates** every generated image as a structured JSON verdict.
- **ComfyUI** — the "hands": FLUX.2 Klein keyframes, Wan 2.2 image-to-video, ReActor face swap,
  and any workflow you register yourself.

```
idea → storyboard ⏸ → keyframes (+evaluate/retry, variants) ⏸ → clips → post-steps → export (+ assembly)
```

`⏸` marks a human approval gate (semi-autonomous mode). Everything is persisted, resumable and
auditable.

## Contents

- [Feature overview](#feature-overview)
- [Architecture](#architecture)
- [Install](#install)
- [Running the services](#running-the-services)
- [The web UI](#the-web-ui)
- [Setup phase](#setup-phase): environment check · projects & assets · output profiles · workflows
- [Jobs](#jobs): wizard · gates · evaluator · regeneration · stills & variants · edit mode · post-steps
- [Deliverables](#deliverables): export folder · assembling clips · versions · speed
- [Command line](#command-line)
- [HTTP API](#http-api)
- [Configuration](#configuration)
- [Reproducibility & provenance](#reproducibility--provenance)
- [Development](#development)

## Feature overview

| Area | What you get |
|---|---|
| Formats | Output **profiles** decide aspect (16:9 / 9:16 / 1:1 / any W:H), keyframe and clip sizes, clip length (Wan 4n+1 frames), fps, sampler settings, style vocabulary and negatives. Built-ins: cinematic 16:9, social 9:16, social 1:1, stills 16:9. |
| Storyboarding | Gemma breaks the idea into scenes and writes format-aware keyframe and motion prompts (framing hints for vertical/square, motion amount for the clip length, style prefix, negatives). |
| Self-evaluation | Gemma vision judges each keyframe (action, identity, artifacts, composition, framing for the delivery format, 1–10 score) and the pipeline retries with the model's fix suggestion. Objective checks gate; the score only flags for review. |
| Human gates | Semi-autonomous runs pause after the storyboard and before the expensive video phase. Edit prompts, assign the face and references per scene, pick variants, apply the evaluator's suggestions, regenerate a scene, then approve. |
| Identity | **Per scene.** Each scene carries its own face for the ReActor swap (or none) and its own edit-mode references, chosen from the project's assets. Label faces by who/when ("Maria, age 8") and the storyboard assigns them to scenes; review and change them at the gates with thumbnail pickers. The evaluator judges `identity_match` against the scene's face. Reference-locked **edit mode** uses the Klein identity/KV edit workflows for "same composition, light edit" variants. |
| Stills jobs | Storyboard → N keyframe variants per scene, best chosen by verdict, face swap per variant, no video phase. |
| Post-processing | Any registered `postprocess` workflow (interpolation, upscale, …) runs after each clip or still, in order. Built-in `faceswap_video` imposes the master face on every frame of a clip. |
| Pluggable workflows | Register a newly exported ComfyUI graph with a YAML sidecar, no Python: `vorch registry inspect` → `add` → `validate --online`. |
| Cloud backends (optional) | Any workflow can declare `backend: modal-h100`; its runs go to a remote ComfyUI (e.g. on Modal) and the outputs are downloaded into the local layout. Ready-made cloud workflows: Wan 2.2 fp8/fp16 (480p/720p), **LTX-2.5 with native audio**, FLUX.2 Klein 4B/9B keyframes, **Wan-Animate-2** character replacement. Design, cost model and phases in [`docs/modal-backend.md`](docs/modal-backend.md); deployment files and runbook in [`modal/`](modal/README.md) (`pip install -e ".[cloud]"`). |
| Deliverables | A per-job **export folder** for Shotcut (numbered stills/clips, manifest, storyboard, README). **Assemble** selected clips into one video with per-cut ffmpeg `xfade` transitions, per-clip speed, optional audio (LTX clips), kept as versions. |
| Live control | Next.js UI with live progress (phase stepper, current step, elapsed time), server-sent events with replay, decision log, provenance (seed + prompt per file). |
| Operations | Environment check (database, ComfyUI, Gemma + vision probe, ffmpeg, GPU headroom, every workflow's node classes and model files), VRAM guard between heavy models, PostgreSQL state, CLI for everything. |

## Architecture

```
┌─ Next.js UI :3000 ──────────────┐    rewrites /api, /media
│  setup (env · workflows · profiles ·│ ─────────────────────────┐
│  projects) · new-job wizard · job   │                          ▼
│  page (SSE timeline, gallery, gates,│   ┌─ FastAPI :8000 (vorch serve) ─────────────────────┐
│  assemble, export)                  │   │  /api + SSE · /media (ComfyUI output)               │
└─────────────────────────────────────┘   │  JobManager (one GPU lock) → Orchestrator:         │
                                          │   STORYBOARD ─⏸─ KEYFRAMES ─⏸─ VIDEOS ─ POST ─ EXPORT│
                                          │  PostgreSQL: jobs (JSONB) · decisions · profiles ·  │
                                          │  projects        registry: workflows/*.yaml         │
                                          └────────┬────────────────────────┬──────────────────┘
                                                   │ HTTP (called as tools)  │ HTTP
                                                   ▼                         ▼
                                     Gemma · llama-server :8080     ComfyUI :8188
                                     storyboard/prompts · vision    /prompt · /history · /free
                                     eval → JSON verdict            = deterministic generator
```

Design principles (see [`DESIGN.md`](DESIGN.md) §2, §11): a deterministic Python state machine
drives everything and Gemma is called as a tool, never as a planner; ComfyUI workflows are
**templates** whose listed node fields are replaced by id, never restructured; generation is
**phased by stage** (all keyframes, then all videos) with VRAM freed between heavy models;
Wan clip length is always 4n+1 frames; evaluation returns JSON.

**Hardware target:** RTX 5070 Ti (16 GB VRAM), 64 GB RAM, Linux. The 16 GB ceiling drives the
VRAM strategy: Gemma runs on the CPU (`--n-gpu-layers 0 --no-mmproj-offload`) so ComfyUI owns
the GPU.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[app,dev]"      # core + FastAPI server + test tools
```

### PostgreSQL

State lives in a local PostgreSQL database (psycopg 3, JSONB, plain-SQL migrations in
`vorch/migrations/`). The default DSN `postgresql:///vorch` uses the local socket with peer
auth, so your OS user needs a role with `CREATEDB` (`sudo -u postgres createuser -s $USER` once).

```bash
python -m vorch db init                          # create the database + apply migrations
python -m vorch db status                        # DSN, schema version, job count
python -m vorch db import-sqlite runs/vorch.db   # one-off import of pre-PostgreSQL runs
```

### Node (UI only)

The web UI needs Node 22 (`frontend/.nvmrc`); `vorch serve` never does.

```bash
cd frontend && npm ci
```

## Running the services

1. **ComfyUI** on `:8188` with the FLUX.2 Klein, Wan 2.2 (GGUF) and ReActor models.
2. **Gemma** (llama-server) on `:8080`, on the CPU so it never competes with ComfyUI for VRAM,
   and **with the vision projector** — without `--mmproj` every verdict is meaningless:
   ```bash
   ./build/bin/llama-server \
     -m ~/models/gemma4/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
     --mmproj ~/models/gemma4/mmproj-BF16.gguf \
     --n-gpu-layers 0 --no-mmproj-offload --ctx-size 32768 \
     --host 0.0.0.0 --port 8080
   ```
3. **API**: `.venv/bin/python -m vorch serve` → `http://127.0.0.1:8000/api` (docs at `/docs`).
4. **UI**: `cd frontend && npm run dev` → `http://localhost:3000` (`make frontend && make start`
   for a production build).

Check everything at once: `python -m vorch setup check --deep` (or *Setup → Environment* in
the UI).

## The web UI

| Route | Purpose |
|---|---|
| `/jobs` | All jobs with id, status, project, kind and profile name; filters by project, unassigned and status, kept in the URL (`/jobs?project_id=…`, `?unassigned=1`); live status badges; delete. |
| `/jobs/new` | The **new-job wizard**, in two shapes. *Generate images*: project → profile → brief (idea, style, scenes, seed, autonomy, keyframe mode + references, face swap, evaluation) → review. *Animate my images*: the keyframes already exist as project assets — pick them in playing order, give a one-line **direction**, and vorch makes one scene per asset with that image installed as its keyframe and the motion written by Gemma **looking at the frame**. No storyboard, no keyframe generation; face swap must be off, and an asset of another aspect is flagged before it is centre-cropped. Submission is blocked, with reasons, while the environment check fails (a *force* switch overrides). |
| `/jobs/[id]` | The **job page**: progress panel, gate/review panel, scene grid (keyframes, variants, clips), assemble panel, export panel, collapsible decision log and provenance. Everything updates live over SSE; the stream resumes across approval gates and reloads. |
| `/setup/env` | Environment check with a re-run button and the effective configuration. |
| `/setup/workflows` | Registered ComfyUI workflows, their parameters and requirements; how to add one. |
| `/setup/profiles` | Output profiles: clone, edit (aspect presets, sizes, 4n+1 helper, style vocabulary, workflow pickers, post-steps), delete. Built-ins are read-only. |
| `/projects` | The projects list (default profile, assets, jobs, last activity, plus an **Unassigned** row for jobs without a project) and the project page: settings, asset uploads (master face, references, products, logos), the project's jobs and **New job in this project** (opens the wizard with the project preselected). `/setup/projects` redirects here. |

**Every preview opens the original.** Click an image (keyframe, variant, asset, face, reference chip)
to see the full file in a lightbox; videos (clips, clip versions, assemblies) and tiles whose click
selects something (asset pickers, version strips) have a ⤢ corner button instead. ← / → step
through a grid, Esc closes, *open original ↗* opens the file in a new tab.

## Setup phase

### Environment check

`GET /api/setup/env`, `vorch setup check [--deep] [--profile ID]`, or the Setup page. Each
check reports ok / warn / fail:

- **database** (reachable, migrations applied), **comfy** (`/system_stats`), **comfy_dirs**
  (input/output writable), **gemma** (`/v1/models`), **gemma_vision** (deep only: a tiny image
  probe that catches a llama-server started without `--mmproj`), **ffmpeg** / **ffprobe**,
  **gpu** (free VRAM vs the guard), one row per registered **workflow** (YAML vs template, and
  online: node classes and model files installed in ComfyUI), the selected **profile**, and the
  **export** / **projects** roots.

A job submit is refused while any check *fails* (warnings pass); the API returns the report with
the 409.

### Projects and assets

A project groups jobs and owns shared files: the **master face** used for identity, and
**reference** / **product** / **logo** images for edit mode. Assets are uploaded in the UI (or
`vorch project add-asset <slug> <path> --kind master_face`), stored under
`~/vorch/projects/<slug>/assets/` and content-hashed. A project also sets the default profile and
an optional export root. Deleting a project keeps its jobs (unlinked).

**Labels matter**, because the storyboard assigns a face and references to each scene by reading
them. Rather than typing them all, let Gemma look at the assets and propose one each.

**In the UI this happens on its own.** An uploaded asset without a label is described in the
background, one at a time (~20–40 s each on CPU Gemma), and its suggested label appears flagged
**AI**. Describing waits while a job is generating, so it never slows a storyboard or the evaluator.
- **Describe unlabelled** queues older assets, or any that a server restart dropped.
- **Describe again** redoes a suggestion.
- The project's **describe hint** (in its settings) gives Gemma the tag vocabulary.
- `VORCH_AUTO_DESCRIBE=0` turns describe-on-upload off.

Large projects stay manageable:
- The asset page and every picker search labels, captions, tags and sign text ("mizen" finds the
  plaque), and use small cached thumbnails.
- The asset page can show **unreviewed only**.
- In a job's scene cards the pickers stay collapsed to a summary until you **Choose…**.

An asset's **kind** can be changed from its tile (or with `vorch project set-kind <slug> <asset> <kind>`).
A suggested label is rebuilt for the new kind; a label you wrote is kept. Scenes already using the
asset are not changed.

From the command line:

```bash
vorch project label-assets <slug> --limit 3 --dry-run \
  --hint "a bicycle trip around Irish lighthouses; tag shots as lamp_room, fog_horn, camp, road"
vorch project label-assets <slug> --hint "…"          # ~20-40 s per asset on CPU Gemma
```

`--hint` says what the collection is and which tag vocabulary to use, so the schema stays
general-purpose while the tags fit the project. A suggestion is recorded as `label_source="auto"`
and flagged **AI** in the UI; editing or confirming it in `/projects` (or at a gate) makes it
`"human"`, after which no later run overwrites it — `--force` redoes only the suggestions, and
replacing a person's wording needs `--overwrite-human`. The full description is kept on the asset
and shown under the thumbnail in the per-scene pickers, so choosing a reference is informed rather
than a guess. `--relabel` rebuilds labels from the stored descriptions without calling the model.

### Output profiles

A profile is the complete generation preset for a job and is **snapshotted onto the job** at
submit, so later edits never change a running job.

| Block | Fields |
|---|---|
| identity | `purpose` (cinematic, social_vertical, social_square, ad_broll, stills), `kind` (video / stills), `aspect` |
| keyframe | workflow, width × height (16-aligned), steps, cfg, unet |
| video | workflow, width × height, `length` (4n+1: 49, 81, 121 …), fps, negative |
| stills | variants per scene (1–8) |
| style | keyframe prefix (technique first, e.g. "impasto oil painting" or "phone photo, natural light"), keyframe negative, animation suffix, animation negative |
| face_swap | enabled, workflow, restore model |
| evaluate | enabled, max attempts, review threshold |
| edit_workflow | which `edit` workflow reference-locked keyframes use |
| post_steps | ordered `postprocess` workflows with parameters |

The two sizes are **different things**: the first is the still Klein renders, the second the size
Wan animates it at (Wan downscales and centre-crops the still itself). They only have to share an
aspect — and the clip, the smaller one, is what you end up with. See
[Delivery](#delivery-what-you-actually-ship) for turning that into a file you can publish.

**Custom sizes are the point of cloning a profile.** Any 16-aligned size the animate workflow
allows works (the built-in Wan graph caps each side at 1280; raise it by copying
`workflows/animate.yaml` into `~/vorch/workflows/` and editing `max:`). The editor shows what a
size costs as you type it — **how long one clip takes on this GPU, and how far past the tested
16 GB budget it is** — because a legal size can still be one nobody wants to wait for: 832×1248 is
~2.6× the budget and roughly half an hour a clip, against ~8 minutes at 832×480. The new-job
wizard multiplies that by the scene count. Estimates come from measured runs
(`docs/modal-backend.md`) and are linear in pixels × frames, so treat them as a guide once RAM
offload starts.

Built-ins (YAML in `vorch/profiles_builtin/`, read-only): `cinematic_16x9` (still 1024×576, clip
832×480, 49 frames), `social_9x16` (still 576×1024, clip 480×832), `social_1x1` (768×768 → 640×640),
`stills_16x9` (4 variants, no video). Clone one to change anything (for example `length: 81`
for longer clips). Aspect presets compute the sizes; any `W:H` works.

### Workflows (pluggable ComfyUI graphs)

Every ComfyUI template ships with a YAML sidecar (`workflows/*.yaml`) that declares its
**kind** (keyframe, animate, faceswap, edit, postprocess, custom), which parameters are staged
files, its output nodes, and a semantic parameter → `(node, field)` map with types, ranges and
constraints. Per-kind contracts guarantee the pipeline can drive any workflow of a kind.

To use a new graph without touching Python: export it from ComfyUI as **API** JSON, then

```bash
python -m vorch registry inspect ~/vorch/workflows/my_graph.json --online   # nodes + role guesses
python -m vorch registry add ~/vorch/workflows/my_graph.json --kind postprocess
python -m vorch registry validate my_graph --online                          # vs the running ComfyUI
```

Entries in `~/vorch/workflows/` override built-ins by name; the UI lists them under
*Setup → Workflows* and offers them in profile pickers. Guide: [`docs/workflows.md`](docs/workflows.md).

## Jobs

### Creating a job

From the wizard, the API (`POST /api/jobs`) or the CLI (`vorch run job.yaml`). The profile is
resolved as `profile_id` → the project's default → `cinematic_16x9`, optionally deep-merged
with `profile_overrides`; the master face comes from the request or the project's asset.

`job.yaml` keys: `idea`, `style`, `n_scenes`, `seed` (0 = random, materialised and recorded),
`autonomy` (`semi` | `full`), `profile`, `profile_overrides`, `aspect` (shortcut for sizes),
`project` (slug), `mode` (`t2i` | `edit`), `references`, `master_face`, `face_swap`,
`evaluate`, `retry.max_attempts`, `video.length` / `video.fps`, `export`.

### Phases and gates

Video jobs: **storyboard** ⏸ **keyframes** ⏸ **videos** (swap → Wan → post-steps) → done →
export. Stills jobs: **storyboard** ⏸ **keyframes** (variants) → finish (swap per variant, image
post-steps) → done → export. In `semi` autonomy the run pauses at each `⏸`; `full` runs through.

At a gate (and after completion) the job page's review panel lets you edit the description,
prompts and negatives per scene, switch a scene to edit mode with references, pick a keyframe
variant, and **approve**. Prompts of scenes that already have a keyframe apply on regeneration.

### Evaluation

Each keyframe attempt is judged by Gemma vision against the scene's intent and the delivery
format: `action_correct`, `identity_match`, `style_match`, `artifacts`, `composition_ok`,
`framing_ok` (portrait/landscape/square framing), a 1–10 `score`, `verdict` and a
`fix_suggestion`. The **objective** booleans decide accept/retry; the score only flags an
accepted frame for review when below the profile's threshold. Every attempt, retry, review flag,
approval and regeneration lands in the **decision log**.

### Regenerating a scene

"Regenerate keyframe" discards a scene's keyframe (and its swap and clip), applies any prompt
edits, reopens the keyframes gate and reruns only that scene with a new seed. The evaluator's
suggestions for the scene are listed with an **Apply to prompt** button, which appends
`Correction: …` exactly as the automatic retry does. "Re-animate" keeps the keyframe and renders
the clip again. The export folder is rebuilt when the job completes again.

### The job page

Each scene is one row: its description and prompts, and directly **below them the media, keyframe
and clip side by side** — the keyframe with **Regenerate keyframe** / **Use my image…**, the clip
with **Re-animate** — so an edit and its result stay together as you scroll.

Under each, a **version strip** holds every keyframe and every clip the scene has ever had, newest
first, with the active one ringed. Click one to make it active: that is the keyframe the clip is
animated from, and the clip the export and **Assemble** use. It is a pointer change, applied at
once — nothing is regenerated. Choosing an older keyframe brings back the clip that was rendered
from it, if the history still has one. **✕** drops a version from the job and keeps the file on disk.

The scene chips in the progress panel jump to a scene, the Save/Approve bar stays in reach, and a
refetch (a finished step, someone else's edit) no longer throws away a half-typed prompt. The whole
storyboard is still available as a gallery under **All scenes**.

### Stopping, editing and resuming

**Stop** on a running job ends it at the next safe point, as **stopped**; finished work is kept.
The ComfyUI step in progress is withdrawn, while a Gemma call finishes first. **Resume** then
continues from the saved state. A job that was running when the server restarted shows as *not
running* and resumes the same way.

Scenes are editable whenever the job isn't running: at either gate, and after it is done, failed or
stopped.
- **Edit with**: a scene in `edit` mode can pick its own edit workflow instead of the profile's.
  `identity_edit` conditions a *fresh* render on the references — same subject, new composition;
  `kv_edit` keeps the base image and changes only what the prompt describes, which is what a small
  fix needs. Choosing an image-edit workflow also ticks **start from this scene's current
  keyframe**, so "remove the lightning, keep the rest" needs no reference hunting. Such a workflow
  takes its size from the reference and ignores the keyframe negative, and the panel says so.
- **What made a version**: the ⓘ on a keyframe or clip tile shows the prompt it was generated
  with, its negative, seed, steps/cfg/size, model and backend — and whether that prompt still
  matches the scene's. **Restore this prompt** puts it back into the scene so you can regenerate
  from an earlier take. Prompts are saved on the version itself (so they outlive the ComfyUI output
  folder) and read from the file's manifest for takes made before that.
- **Animate this scene** renders one clip immediately, whatever a gate is waiting for — reviewing a
  scene and asking for its clip is the review for that scene. The keyframes gate keeps its batch
  meaning (**Approve keyframes** renders every scene still missing a clip, and the panel says how
  many); a job whose last missing clip was rendered this way finishes and exports on its own.
- **Change the storyboard itself**: drag a row (or use ▲▼) to reorder, `✕` to delete — asking each
  time whether to delete what it rendered — and **+ add scene here** between any two rows. A new
  scene is described in one sentence; Gemma writes its keyframe and animation prompts in the
  background (~40 s), and the next run generates only it. Scenes are renumbered, so the export
  folder and the assembly follow the new order; a saved cut list is remapped rather than silently
  pointing at another scene. **Write prompts with Gemma** rewrites one scene's prompts after its
  description changed.
- **Collapsed rows**: above six scenes each one is a single row — number, keyframe thumbnail,
  description, and an **unsaved** badge when it holds edits you have not saved — and clicking it
  opens the full editor in place (several can be open; *Expand all* / *Collapse all* in the header).
  A 40-scene storyboard is then a list you can scan instead of forty stacked editors, and the
  progress panel's scene chips open the scene they jump to.
- **Mode (t2i/edit) and references**: pick project assets in priority order, or **Upload…** a file
  from the scene card, which adds it to the project. On a scene that already has a keyframe, they
  apply on *Regenerate keyframe*. The picker says how many references the edit workflow uses.
- **Suggested reference**: in a t2i job whose project has reference photos, Gemma matches the
  photos' labels to the scenes after writing the storyboard and suggests at most one per scene,
  with the reason and a fit: *same shot* (her photo is essentially this frame: edit mode keeps the
  scene) or *motif only* (same subject, different framing: edit mode would copy the photo's
  composition, so t2i is often better). **Use as reference** switches the scene to edit mode with that
  photo (save to keep it); nothing is applied by itself. Suggestions are logged as `suggested`
  decisions; a failed pass leaves the storyboard intact, just without hints.
- **Save as reference**: copies a scene's keyframe into the project as a reference asset, so
  another scene can be generated from it (the picker only offers project assets). It is a copy, so
  regenerating the source scene never changes the scenes using it; saving the same frame twice
  returns the asset already there, and an unnamed one is captioned in the background like any
  upload. This is how a tower or a staircase keeps the same look across several scenes.
  The picker lists assets **newest first**, so a frame you just saved is its first tile; while a
  search or a tag chip is active it shows a **Clear** button and an "N of M shown" count, so a
  narrowed list never reads as a missing asset.
- **Use my image as keyframe**: pick or upload an image. It is cropped to the job's frame size and
  replaces the keyframe without being generated or evaluated. The generated candidates stay
  selectable, and the gate reopens for review as for a regenerated keyframe.
- **Changing the face or the keyframe variant** discards what was built on the old one.
- **The Assemble panel says when clips changed** since the last assembly.

### Stills and variants

A stills profile generates `variants` keyframes per scene with distinct seeds, evaluates each,
and picks the best (accepted with the highest score). All variants are kept; choose a different
one at the gate or after completion (re-export follows).

### Edit mode (reference-locked keyframes)

`mode: edit` (job-wide, or per scene at the gate) generates keyframes with the profile's edit
workflow (`identity_edit` or `kv_edit`) from one or more reference images, which are fitted to
the target size and staged once per run. Reference editing locks the composition, so it is meant
for "same composition, light edit" variants and product placement, not for fresh scenes.

### Post-processing chains

A profile's `post_steps` run after each clip (and each still variant): the previous output is
staged into ComfyUI, `fps` / `seed` are filled in when the workflow declares them, VRAM is
reclaimed when the workflow changes, and steps whose input type does not match are skipped. The
raw Wan clip is kept alongside the final one.

### Live progress

The job page shows a phase stepper, a progress bar over keyframe/clip units, the current step
with elapsed time ("Animating scene 2 with Wan — 81 frames @ 16 fps (~5 s); this is the slow
step"), and per-scene state chips. Events are named SSE messages with ids; reloads replay the
timeline and reconnects resume from the last id.

## Deliverables

### Export folder (Shotcut handoff)

Written automatically when a job finishes, or on demand (`POST /api/jobs/{id}/export`,
`vorch export <job>`), under `~/vorch/export/<project slug or _unassigned>/<job8>_<idea>/`:

```
README.txt            fps, resolution, frames, clip length, scene order, Shotcut notes
manifest.json         profile snapshot, prompts, seeds, verdicts, sources, assembly versions
storyboard.md         per scene: description, prompts, verdicts, seeds
s01_keyframe.png      s01_keyframe_v2.png … (stills variants)
s01_swapped.png       (when the master face was imposed)
s01_clip.mp4          s01_clip_raw.mp4 (when post-steps changed the clip)
assembly.mp4          (optional: the current assembly)
provenance/           copied sidecar manifests
```

Files are hardlinked when possible (copied across filesystems); re-export is idempotent and the
UI flags a stale export after the job changed.

### Assembling clips

The **Assemble** panel joins selected clips of a job into one video:

- tick and reorder scenes (final or raw clip per scene);
- choose a **transition per cut**: any ffmpeg `xfade` transition (fades, wipes, slides,
  shapes, wind …) with its length, or a hard cut;
- set a **target duration per clip** to change its speed (0.25×–4×); slow motion repeats
  frames or, with **smooth slow motion**, uses motion-compensated interpolation (much slower to
  render);
- pick a **delivery target** (below), the fps, grain, or the NVENC encoder.

ffmpeg renders once (h264) to `assembly_v<N>.mp4` next to the clips,
with a provenance sidecar. **Every assembly is kept as a version**: play any, *Use for export*
(it becomes `assembly.mp4` in the export folder), or delete it. From the CLI:

```bash
python -m vorch assemble <job_id> --transition fade --duration 0.5   # same cut everywhere
python -m vorch assemble <job_id> --spec cuts.json                    # per-cut / per-clip spec
python -m vorch assemble <job_id> --list                              # versions (* = current)
python -m vorch assemble <job_id> --select 2                          # make v2 current
```

`cuts.json` mirrors the API spec, e.g.
`{"items":[{"scene_index":0,"target_seconds":6.1},{"scene_index":1}],"cuts":[{"transition":"wipeleft","duration":0.3}],"smooth":true}`.

### Delivery: what you actually ship

A 16 GB card animates at 480p, so a job's clips are 832×480 (or 480×832) at 16 fps. **The assembly
is where that becomes a deliverable file** — it is the only step that re-encodes. Choose a target in
the Assemble panel, or `--delivery` on the CLI:

| target | size | fps | for |
|---|---|---|---|
| `source` | the clips' own | the clips' own | an edit in an NLE, or when you want the raw thing |
| `master_1080` | the job's aspect at 1080 (1920×1080 / 1080×1920 / 1080×1080) | 24 | a film master, a screener, Vimeo |
| `social_1080` | same | 30 | Reels / TikTok / Shorts — **they accept 23–60 fps, so 16 is out of range** |
| `youtube_1080` | 1920×1080 | 30 | landscape upload |

Every target is scaled with lanczos, tagged Rec.709, and encoded H.264 High with a 2-second closed
GOP and `+faststart` — what the platforms ask for. **Grain** (light/medium) is worth turning on for
an upscale: it hides the plastic look that scaling 480p to 1080p produces. `--max-mbps` caps the
bitrate when a platform publishes a ceiling (Instagram documents 25 Mbps).

```bash
python -m vorch assemble <job_id> --delivery master_1080 --grain 1
python -m vorch assemble <job_id> --width 1080 --height 1920 --fps 30   # explicit beats the target
```

**Upscaling is a ceiling, not a fix.** Sharpness comes from the clips. When a shot matters, render
it natively bigger instead: the cloud profiles do 1280×720 (Wan fp16, ≈$1.50/clip) or 1280×704 @ 24
fps **with audio** (LTX-2.5, ≈$0.28/clip once warm) — see [`docs/modal-backend.md`](docs/modal-backend.md).
Two things worth knowing: platforms re-encode and serve ≤1080p, so 4K buys nothing for
Reels/TikTok/Shorts; and a 1080p master is a native 2K fit, so festivals do not reject a short for
being finished at 1080p.

## Command line

`python -m vorch <command>` (or `vorch` once installed):

| Command | What |
|---|---|
| `setup check [--deep] [--profile ID]` | environment report; exit 1 on any failure |
| `profiles` · `projects` · `project add / add-asset / show` | setup-phase objects |
| `registry list / validate [--online] / inspect <json> / add <json>` | workflow registry |
| `run job.yaml` · `approve <job> storyboard\|keyframes` · `jobs` · `job <id>` | drive and inspect jobs |
| `export <job> [--force]` · `assemble <job> …` | deliverables |
| `keyframe` · `animate` · `edit` · `faceswap` · `evaluate` | single steps (`--aspect`, `--workflow`, `--negative`, `--ref`) |
| `run-scene` · `run-sequence scenes.yaml` | one scene, or a sequence with last-frame continuity (`continuous: true`, `--no-relock`) |
| `vram [--free]` | GPU memory; reclaim ComfyUI's models |
| `serve [--host --port]` · `openapi` | the API server; the schema for the typed client |
| `db init / status / import-sqlite <file>` | database |

## HTTP API

All routes under `/api`; interactive docs at `/docs`. Errors use one envelope:
`{"error": {"code", "message", "details"}}`.

| Method | Path | Purpose |
|---|---|---|
| POST / GET | `/jobs` | create; list (`limit`, `offset`, `status`, `project_id`, `unassigned`) |
| GET / DELETE | `/jobs/{id}` | job with scenes, media URLs, pending gate, assembly; delete |
| PUT | `/jobs/{id}/scenes` | edit prompts/negatives/mode/references, choose a variant |
| POST | `/jobs/{id}/scenes/{i}/regenerate` | redo a keyframe or clip (optional edits, `apply_suggestion`) |
| POST | `/jobs/{id}/approve/{gate}` | clear `storyboard` / `keyframes` |
| GET | `/jobs/{id}/decisions` · `/provenance` | decision log; seed + prompt per file |
| GET | `/jobs/{id}/events` · `/events` | SSE (named events, ids, `Last-Event-ID` replay) |
| POST / GET | `/jobs/{id}/export` | run / inspect the export |
| GET / POST | `/jobs/{id}/assemble` | status (clips, transitions, versions) / run |
| POST / DELETE | `/jobs/{id}/assemble/versions/{n}/select` · `/versions/{n}` | make a version current / delete it |
| GET | `/setup/env` · `/setup/config` | environment check; effective configuration |
| GET | `/workflows` · `/workflows/{name}` | registry |
| GET / POST / PUT / DELETE, POST `…/clone` | `/profiles[/{id}]` | profiles (built-ins read-only) |
| GET / POST / PUT / DELETE | `/projects[/{id}]`, `/projects/{id}/assets[…]`, `/projects/{id}/jobs` | projects and assets (multipart upload) |
| GET | `/media/{path}` | generated files (Range requests, no-cache) |

## Configuration

Environment variables (see `vorch/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `VORCH_COMFY_URL` / `VORCH_GEMMA_URL` | `http://localhost:8188` / `:8080` | services |
| `VORCH_COMFY_ROOT` | `~/comfy/ComfyUI` | its `input/` and `output/` dirs |
| `VORCH_DATABASE_URL` / `VORCH_TEST_DATABASE_URL` | `postgresql:///vorch` / `…/vorch_test` | state |
| `VORCH_WORKFLOW_DIR` / `VORCH_USER_WORKFLOW_DIR` | `workflows/` / `~/vorch/workflows` | registry |
| `VORCH_BACKENDS_FILE` | `~/vorch/backends.yaml` | remote ComfyUI backends (e.g. Modal); see `docs/modal-backend.md` |
| `VORCH_ENV_FILE` | `~/vorch/vorch.env` | `KEY=VALUE` secrets (Modal proxy token, `HF_TOKEN`) loaded into every vorch process at start; existing variables win |
| `VORCH_COMFY_REQUEST_TIMEOUT` / `VORCH_COMFY_POLL_INTERVAL` / `VORCH_COMFY_COLD_START_GRACE` | 30 s / 2 s / 240 s | ComfyUI client: per-call timeout, history polling, cold-start tolerance for remote backends |
| `VORCH_PROJECTS_ROOT` / `VORCH_EXPORT_ROOT` | `~/vorch/projects` / `~/vorch/export` | files |
| `VORCH_RUN_TIMEOUT` / `VORCH_ASSEMBLE_TIMEOUT` | 600 s | per ComfyUI workflow / ffmpeg assembly |
| `VORCH_VRAM_GUARD_MB` / `VORCH_VRAM_SETTLE_SEC` | 10000 / 3 | VRAM guard |
| `VORCH_UI_URL` / `VORCH_CORS_ORIGINS` | `http://localhost:3000` | UI origin (CORS only needed without the Next proxy) |

Frontend: `API_ORIGIN` (server-side rewrite target, default `http://127.0.0.1:8000`) and
`NEXT_PUBLIC_API_ORIGIN` (make the browser talk to FastAPI directly).

## Reproducibility & provenance

Every generated image, clip and assembly gets a sidecar `<file>.json` recording the workflow,
template, exact prompts, the **materialised seed** (a job seed of 0 picks a random one, which is
then recorded), sizes and all parameters, plus the job, profile, scene and variant it belongs to.
Seeds derive deterministically per scene, variant, attempt and regeneration from the job seed, so
a fixed seed reproduces a whole job. The decision log explains why each frame was accepted,
retried, flagged or regenerated.

## Development

```bash
.venv/bin/python -m pytest -q          # ~300 tests; needs PostgreSQL (schema per test), ffmpeg for media tests
cd frontend && npm test                # Vitest + Testing Library
make api-types                         # regenerate frontend/openapi.json + TS types after API changes (a test enforces it)
make frontend && make start            # production UI build
```

Documentation map: [`NOTES.md`](NOTES.md) (the original spec: endpoints, node ids, VRAM
strategy, decisions), [`DESIGN.md`](DESIGN.md) (architecture, stack, build order),
[`docs/workflows.md`](docs/workflows.md) (registering workflows),
[`docs/modal-backend.md`](docs/modal-backend.md) (optional Modal cloud backend),
[`docs/sequential-continuity-plan.md`](docs/sequential-continuity-plan.md),
[`SKILL.md`](SKILL.md) + `skills/` (operational how-tos), [`CLAUDE.md`](CLAUDE.md) (working
rules for coding sessions).

## License

Not yet specified.
