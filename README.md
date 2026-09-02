# Video Orchestrator

Local, self-contained AI video-production pipeline. It turns a text idea into a series of
consistent video clips — autonomously and entirely on your own machine — by directing two
local systems in a self-evaluating loop:

- **Gemma 4** (`gemma-4-26B-A4B`, multimodal) — the reasoning + vision "brain": writes
  storyboards/prompts and **evaluates** every generated result, deciding accept or retry.
- **ComfyUI** — the "hands": FLUX.2 Klein keyframes, Wan 2.2 image-to-video, ReActor face swap.

```
idea → storyboard → [per scene: keyframe → EVALUATE → animate → EVALUATE → face swap → EVALUATE] → clips
```

After each generation Gemma scores the result against the task and either accepts it or corrects
the prompt and retries (up to N attempts).

## Architecture

A **deterministic Python core** drives all control flow; Gemma is called as a *tool*, never as an
autonomous planner. This choice (over an LLM-agent framework) was validated by a multi-source
research pass — see [`DESIGN.md`](DESIGN.md) §11–12.

```
┌─ Next.js UI :3000 ──────────────┐    rewrites /api, /media
│  setup (env · workflows · profiles ·│ ─────────────────────────┐
│  projects) · new-job wizard · job   │                          ▼
│  page (SSE timeline, gallery, gates)│   ┌─ FastAPI :8000 (vorch serve) ─────────────────────┐
└─────────────────────────────────────┘   │  /api + SSE · /media (ComfyUI output)               │
                                          │  JobManager (one GPU lock) → Orchestrator:         │
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

The pipeline is phased (all keyframes → `/free` → all videos → `/free` → all swaps) to fit a
**16GB VRAM** budget, with semi-autonomous human-approval gates after the storyboard and after
keyframe evaluation.

**Hardware target:** RTX 5070 Ti (16GB VRAM), 64GB RAM, Linux.

## Status

**Phase 1 — walking skeleton** ✅ — a deterministic single-scene driver that proves the ComfyUI
chain end-to-end (keyframe → face swap → animate).

**Phase 2 — in progress.** The **evaluator** is built: Gemma vision scores a keyframe into a
validated `Verdict` and the pipeline retries with the fix suggestion (`run-scene --evaluate`,
or the standalone `evaluate` command). **Sequential continuity** is built too: `run-sequence`
chains scenes so the last frame of one clip seeds the next (`continuous: true`), re-locking
identity each step. **State** is persisted: `state.py` stores each `Job`/`Scene` in PostgreSQL
(JSONB, resumable) with an append-only **decision log** recording why every frame was accepted or
retried. And the **orchestrator** ties it together: from a one-line idea it generates a
storyboard (Gemma), then drives storyboard → keyframes (+eval/retry) → videos as a phased,
resumable state machine with **semi-autonomous approval gates** (`run` / `approve` / `jobs`).
A **VRAM guard** (`vram.py`) reclaims GPU memory at the Wan boundary. Phase 2 was validated
end-to-end against live Gemma + ComfyUI.

**Phase 3 — multipurpose raw material + setup phase** ✅ (Sept 2026). The app now produces
raw material for cinematic *and* marketing use: **output profiles** (16:9 / 9:16 / 1:1 video,
stills-only jobs with N variants, longer clips, style vocabularies) drive every size, length and
prompt; keyframes can be **reference-locked edits** (identity / KV edit); registered
**post-process** workflows run after each clip. The **setup phase** adds a structured
**environment check**, **projects** with shared assets (master face, references), and a
per-job **handoff export folder** for Shotcut (`sNN_clip.mp4`, `sNN_keyframe.png`, manifest,
README). State moved to **PostgreSQL**; the UI is a **Next.js** app on the FastAPI API. Any
ComfyUI workflow can be plugged in through a YAML sidecar without touching Python.

### Pluggable workflows

Each ComfyUI template is paired with a registry YAML (`workflows/*.yaml`) that names the
node fields the orchestrator may replace. New workflows drop into `~/.vorch/workflows/`
without Python changes: `python -m vorch registry inspect <api.json>` → `registry add` →
`registry validate --online`. See [`docs/workflows.md`](docs/workflows.md).

## Repository layout

| Path | What |
|------|------|
| [`NOTES.md`](NOTES.md) | The spec — endpoints, per-workflow node IDs, VRAM strategy, decisions. **Read first.** |
| [`DESIGN.md`](DESIGN.md) | Orchestrator architecture, the validated stack, build order. |
| [`SKILL.md`](SKILL.md) + `skills/` | Operational how-tos (ComfyUI API, Gemma eval/prompt, VRAM). |
| `workflows/` | ComfyUI API-format workflow templates (Klein / Wan / ReActor). |
| `vorch/` | The orchestrator package (clients, workflow registry, steps, pipeline, CLI). |
| `tests/` | GPU-free checks for the workflow registry. |

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The core needs `requests`/`pydantic`/`psycopg`; the `[app]` extra adds the FastAPI server (`vorch serve`).

### Database (PostgreSQL)

Job state and the decision log live in a local **PostgreSQL** database (psycopg 3, JSONB,
plain-SQL migrations in `vorch/migrations/`). The default DSN `postgresql:///vorch` uses the
local socket with peer auth, i.e. your OS user needs a PostgreSQL role with `CREATEDB`
(`sudo -u postgres createuser -s $USER` once). Then:

```bash
python -m vorch db init                      # create the database + apply migrations
python -m vorch db status                    # DSN, schema version, job count
python -m vorch db import-sqlite runs/vorch.db   # one-off: bring over pre-PostgreSQL runs
```

Override with `VORCH_DATABASE_URL=postgresql://user:pass@host/db`. Tests use
`VORCH_TEST_DATABASE_URL` (default `postgresql:///vorch_test`) and isolate every test in its
own throwaway schema; they stop with a clear message if PostgreSQL is not running.

## Usage

Requires the local services running: **ComfyUI** on `:8188` (with the FLUX.2 / Wan / ReActor
models) and, for prompt generation, **Gemma** (llama-server) on `:8080`. Endpoints and paths are
configurable via `VORCH_*` environment variables (see `vorch/config.py`).

```bash
# Individual steps (each isolates one ComfyUI endpoint)
python -m vorch keyframe "oil painting, woman kneading bread, impasto, visible brushstrokes"
python -m vorch animate keyframe.png "slow camera push-in, hands kneading dough"
python -m vorch faceswap scene.png master_face.jpeg

# Score an image against an intent (Gemma vision → accept/retry verdict)
python -m vorch evaluate keyframe.png "woman kneading bread, oil painting"

# Show GPU memory (and optionally reclaim ComfyUI VRAM)
python -m vorch vram
python -m vorch vram --free

# Full single scene: keyframe → swap → animate
python -m vorch run-scene \
  --keyframe-prompt "..." \
  --animation-prompt "..." \
  --master master_face.jpeg

# ...with self-evaluation: judge the keyframe and retry with the fix (max N)
python -m vorch run-scene --evaluate --max-attempts 2 \
  --keyframe-prompt "..." --animation-prompt "..." --master master_face.jpeg

# A sequence of scenes — continuous scenes start on the previous clip's last frame
python -m vorch run-sequence scenes.yaml --master master_face.jpeg
```

`scenes.yaml` is a list of scenes; flag `continuous: true` to seed a scene from the previous
clip's last frame (otherwise it gets a fresh identity-locked keyframe):

```yaml
- keyframe_prompt: "a castle at dawn, oil painting, impasto"
  animation_prompt: "slow camera pan left"
- keyframe_prompt: "(ignored when continuous)"
  animation_prompt: "push in on the gate"
  continuous: true
```

Carried frames are re-locked to the master face by default (ReActor swap, countering Wan drift);
pass `--no-relock` to disable. See [`docs/sequential-continuity-plan.md`](docs/sequential-continuity-plan.md).

### Full job from an idea (the orchestrator)

Drive the whole phased pipeline from a one-line idea via a YAML job spec
(see [`examples/job.example.yaml`](examples/job.example.yaml)):

```bash
python -m vorch run job.yaml          # storyboard -> keyframes (+eval) -> videos
python -m vorch jobs                  # list persisted jobs + status
python -m vorch job <job_id>          # scenes + full decision log
```

In `autonomy: semi` the run pauses at approval gates (after the storyboard, and again before the
expensive Wan phase); inspect, then continue:

```bash
python -m vorch approve <job_id> storyboard   # release the storyboard gate
python -m vorch approve <job_id> keyframes    # release the pre-Wan gate
```

`autonomy: full` runs straight through. State + the decision log persist to PostgreSQL
(`VORCH_DATABASE_URL`, default `postgresql:///vorch`), so a job is **resumable** and every
accept/retry is auditable.

**Face swap is optional.** Set `face_swap: false` in the job spec (or the *Face swap* switch in the
dashboard, or `--no-face-swap` on `run-scene`/`run-sequence`) to animate the keyframe directly
without imposing a master face — useful for landscapes, objects, or generic characters. `master_face`
is then not required.

### Web UI (Next.js)

The control panel is a separate **Next.js** app in `frontend/` (App Router, TypeScript,
Tailwind, TanStack Query) that proxies `/api` and `/media` to FastAPI, so the browser stays
same-origin. Node 22 (`frontend/.nvmrc`) is required for the UI only — `vorch serve` never
needs it.

```bash
# terminal 1 — API + media
.venv/bin/python -m vorch serve                 # http://127.0.0.1:8000/api, /docs

# terminal 2 — UI (first time: npm ci)
cd frontend && npm run dev                      # http://localhost:3000
```

`make frontend` builds it (`npm ci && next build`), `make start` serves the build, and
`make api-types` regenerates `frontend/openapi.json` + `src/api/schema.d.ts` after any API
change (a pytest snapshot test fails until you do). Frontend tests: `cd frontend && npm test`.
If proxied SSE ever buffers, set `NEXT_PUBLIC_API_ORIGIN=http://127.0.0.1:8000` to have the
browser talk to FastAPI directly (CORS is enabled for the Next origin).

### Reproducibility

Every generated image/video gets a sidecar manifest next to it — `<output>.json` — recording the
exact prompt(s), the seed, the template, and all params that produced it. A `seed` of `0` means
"pick a random one": a concrete seed is materialized, used, and recorded (so the saved value
actually reproduces the result — re-running with that seed + prompt regenerates the same frame).

Evaluation requires Gemma started with `--mmproj` (vision projector); without it the model is
blind and verdicts are meaningless.

Run the GPU-free registry checks with:

```bash
.venv/bin/python -m pytest -q
```

## License

Not yet specified.
