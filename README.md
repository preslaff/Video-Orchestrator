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
┌─ NiceGUI app (on FastAPI) ── single local process ──────────────────────┐   [Phase 2]
│  UI: job form · live progress (SSE) · keyframe gallery + approval gate    │
│  Deterministic state machine (the orchestrator core):                     │
│    STORYBOARD ─gate─ KEYFRAMES ─/free─ VIDEOS ─/free─ SWAPS ─ FINAL        │
│    job queue = asyncio + SQLite (resumable state + decision log)           │
└───────────────┬──────────────────────────────────┬───────────────────────┘
                │ HTTP (called as tools)            │ HTTP
                ▼                                    ▼
   Gemma · llama-server :8080             ComfyUI :8188
   text: storyboard/prompts              /prompt · /history · /free · /interrupt
   vision: eval → JSON verdict           = deterministic generator
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
identity each step. **State** is persisted: `state.py` stores each `Job`/`Scene` in SQLite
(resumable) with an append-only **decision log** recording why every frame was accepted or
retried. And the **orchestrator** ties it together: from a one-line idea it generates a
storyboard (Gemma), then drives storyboard → keyframes (+eval/retry) → videos as a phased,
resumable state machine with **semi-autonomous approval gates** (`run` / `approve` / `jobs`).
Still to come: the VRAM manager and the async queue + NiceGUI dashboard (designed in `DESIGN.md`).

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

Phase 1 needs only `requests`; the `[app]` extra (NiceGUI/FastAPI/Instructor) is for Phase 2.

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

`autonomy: full` runs straight through. State + the decision log persist to SQLite
(`VORCH_STATE_DB`, default `runs/vorch.db`), so a job is **resumable** and every accept/retry is
auditable.

Evaluation requires Gemma started with `--mmproj` (vision projector); without it the model is
blind and verdicts are meaningless.

Run the GPU-free registry checks with:

```bash
.venv/bin/python -m pytest -q
```

## License

Not yet specified.
