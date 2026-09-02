# DESIGN.md — Video Orchestrator pipeline

> Architecture for the local, self-contained workflow-automation pipeline.
> Read `NOTES.md` first (the spec) and `SKILL.md` + `skills/` (operational how-tos).
> This document records *how we build the orchestrator* and the decisions behind it.

## 1. Goal

Turn a text idea into a series of consistent video clips, fully locally, via a
self-evaluating pipeline that directs **Gemma 4** (reasoning + vision) and
**ComfyUI** (generation). See `NOTES.md` §1 for the product behaviour.

## 2. Principles (non-negotiable)

1. **Fully local, offline at runtime, self-contained.** No cloud or external service in the run loop. Claude Code is build-time only.
2. **Deterministic core, Gemma as a called tool.** Plain Python drives all control flow (loops, phases, retries). The *only* model-driven steps are prompt-writing and vision evaluation — Gemma is invoked as a subroutine over its HTTP endpoint, it does not drive dispatch.
3. **No agent framework.** The pipeline is deterministic plumbing; an LLM planner would add fragility, latency, and VRAM cost for no benefit. (OpenClaw / CrewAI / LangGraph evaluated and rejected for the executor role. OpenClaw remains a possible *optional outer trigger layer* later — not in scope here.)
4. **Templates not finals.** Node-ID surgery is centralized in one registry; the rest of the code uses semantic names, never raw node IDs like `"75:73"`.
5. **Assets are runtime inputs.** Master face and references come from the job spec and are injected by node ID at execution time. Nothing hardcoded.
6. **Phase-by-phase, not scene-by-scene** (`NOTES.md` §5) — load each heavy model once, `/free` between phases.
7. **Walking skeleton first** (`NOTES.md` §9) — single scene, no evaluation, end-to-end before anything clever.

## 3. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | **Python** | Ecosystem (requests, ffmpeg bindings), matches skill pseudocode. |
| Orchestration | **Deterministic Python core; Gemma as a called tool; no agent framework** | Dispatch is deterministic; LLM planning is fragile and VRAM-costly. |
| VRAM strategy | **Option B — Gemma pinned to CPU** (`--n-gpu-layers 0`) | ComfyUI owns the GPU; zero swap conflict; simplest. (`NOTES.md` §4) |
| Pipeline phasing | **By phase** (`NOTES.md` §5) | Minimizes model load/unload churn on 16GB. |
| Autonomy | **Semi-autonomous** | Human approval gates after storyboard and after keyframe eval; cheap to remove later. |
| Trigger / job spec | **YAML job file + CLI** | Declarative, version-controllable, keeps assets dynamic. |
| Hermes / other LLMs | **Rejected** | Text-only (no vision — the load-bearing feature) and extra VRAM. |

## 4. Module layout

Plain Python package (working name `vorch` — rename freely):

```
vorch/
├── config.py        # endpoints, comfy input/output dirs, model picks, retry N, eval frame count
├── clients/
│   ├── comfy.py     # load template, set inputs, POST /prompt, poll /history, extract, upload, /free
│   └── gemma.py     # text + vision completions, lenient JSON parse
├── workflows.py     # REGISTRY: logical step → (template file, semantic→node-ID map). All node surgery here.
├── steps/
│   ├── keyframe.py  # Klein text-to-image
│   ├── animate.py   # Wan image-to-video
│   └── faceswap.py  # ReActor
├── evaluator.py     # Gemma vision → Verdict (Instructor/Pydantic); retry-with-fix helper   [Phase 2]
├── storyboard.py    # Gemma text → scene briefs + keyframe/animation prompts                [Phase 2]
├── pipeline.py      # per-scene chains: run_scene / run_sequence (continuity)
├── orchestrator.py  # job-level phase machine + semi-auto gates over state.Store            [Phase 2]
├── state.py         # Job/Scene model, PostgreSQL-persisted (JSONB; resumable + decision log) [Phase 2]
├── media.py         # last_frame() — ffprobe+ffmpeg frame extraction for continuity         [Phase 2]
├── vram.py          # gpu_free_mb(), free_comfyui(), guards (thin under Option B)            [Phase 2]
├── queue.py         # asyncio job queue over PostgreSQL (sequence long jobs, cancel, resume) [Phase 2]
├── server.py        # FastAPI routes + SSE progress stream (lives inside the NiceGUI app)    [Phase 2]
├── app/             # NiceGUI dashboard: job form, live gallery, HITL approval gates, retries [Phase 2]
└── __main__.py      # CLI entry (python -m vorch ...)
```

Phase 1 (the walking skeleton) builds only `config.py`, `clients/`, `workflows.py`,
`steps/`, `pipeline.py`, and `__main__.py`. The `[Phase 2]` modules are designed (§11)
but deferred until the skeleton runs end-to-end.

## 5. The workflow registry (key design)

`workflows.py` is the **single source of truth** for the node IDs in `NOTES.md` §3.
It maps each logical operation to its template file and a semantic→(node-ID, field) map,
so the rest of the code never touches raw node IDs and templates can change in one place.

```python
REGISTRY = {
    "keyframe": {
        "template": "image_flux2_klein_text_to_image.json",
        "params": {
            "positive": ("76",    "value"),
            "seed":     ("75:73", "noise_seed"),
            "width":    ("75:68", "value"),
            "height":   ("75:69", "value"),
            "steps":    ("75:62", "steps"),
            "cfg":      ("75:63", "cfg"),
            "unet":     ("75:70", "unet_name"),
        },
        "outputs": ("images",),
    },
    "animate": {
        "template": "wan22_i2v_gguf.json",
        "params": {
            "start_image": ("5",  "image"),   # filename in ComfyUI input/
            "positive":    ("6",  "text"),
            "negative":    ("7",  "text"),
            "width":       ("8",  "width"),
            "height":      ("8",  "height"),
            "length":      ("8",  "length"),  # MUST be 4n+1 (49, 81, 121…)
            "seed_high":   ("9",  "noise_seed"),
            "seed_low":    ("10", "noise_seed"),
            "fps":         ("12", "fps"),
        },
        "outputs": ("gifs", "videos"),
    },
    "faceswap": {
        "template": "ReActor_Face_Swap.json",
        "params": {
            "input_image":  ("1", "image"),   # scene/frame to swap
            "source_image": ("2", "image"),   # master face
            "restore_model":("3", "face_restore_model"),
        },
        "outputs": ("images",),
    },
}
```

The `comfy.py` client takes a logical step + a dict of semantic params, applies the map
onto a fresh copy of the template, validates (e.g. `length % 4 == 1` for Wan), and dispatches.

## 6. Data model (`state.py`)

- **Job** → **Storyboard** → `list[Scene]`
- **Scene**: `description, character, setting, mood` →
  `keyframe_prompt, keyframe_path, keyframe_verdict` →
  `animation_prompt, video_path, video_verdict` →
  `swapped_path, swap_verdict` → `status`.
  Keeps the original prompt **and** each `fix_suggestion` (needed for retry).
- **Verdict** (`NOTES.md` §6 as a dataclass):
  `action_correct, identity_match, style_match, artifacts, composition_ok, score, verdict, fix_suggestion`.

Persisting Job to JSON gives a **resumable run** and the **decision log** (why each frame
was accepted/retried — Georgi's debugging requirement).

## 7. Job spec (YAML)

```yaml
idea: "Жена меси хляб в селска кухня на разсъмване"
n_scenes: 5
style: "oil painting, impasto, visible brushstrokes"
master_face: assets/jena-cropped.jpeg   # ReActor source — injected at runtime, not hardcoded
references: []                            # optional identity_edit refs (staged into ComfyUI input/)
autonomy: semi                            # semi | full
retry:
  max_attempts: 2
video:
  length: 49                              # 4n+1
  eval_frames: 5
output_dir: runs/                         # job state + outputs land here
```

Run: `python -m vorch run job.yaml`. The skeleton also gets step-level commands
(e.g. `python -m vorch keyframe "<prompt>"`) for isolated testing.

## 8. Control flow (full pipeline)

Phased per `NOTES.md` §5; **⏸ = semi-autonomous human gate**:

1. **Storyboard** (Gemma text) → scenes → ⏸ approve
2. **Prompt gen** (Gemma text) → all keyframe + animation prompts
3. **Klein phase** → generate all keyframes (model loaded once)
4. **Eval keyframes** (Gemma vision) → retry failures with `fix_suggestion`, max N → ⏸ approve before expensive Wan phase
5. `/free` → **Wan phase** → animate approved keyframes
6. **Eval videos** (frame sampling, `NOTES.md` §6) → mark retries
7. `/free` → **ReActor phase** → impose master face
8. **Final eval** → clips ready for edit

Retry policy: feed the original prompt + `fix_suggestion` back to the prompt generator,
regenerate, re-evaluate, up to `max_attempts`. Log every attempt to state.

The **ReActor swap is optional** — a per-job `face_swap` flag (default on, persisted as run-config
like `evaluate`). When off, the keyframe is animated directly and `master_face` is not required
(landscapes/objects/generic characters). In the implemented skeleton ordering the swap runs on the
keyframe *before* Wan (DESIGN §9), so "no swap" simply animates the raw keyframe.

## 9. Build order (walking skeleton → full)

Per `NOTES.md` §9 — narrow and end-to-end first:

**Phase 1 — skeleton (no evaluation):**
1. `clients/comfy.py` — run one workflow, poll, extract output
2. `clients/gemma.py` — text + vision request
3. `workflows.py` — registry for the 3 core templates
4. Single-scene driver: prompt → Klein → ReActor(keyframe) → Wan → output clip, staging each output into `input/` between steps
5. CLI to run it → **confirm end-to-end before continuing**

> Skeleton swap ordering: the provided `ReActor_Face_Swap.json` is **image-based**, so the
> skeleton swaps the face on the *keyframe* then animates (Klein → ReActor → Wan). This differs
> from §8's "swap after Wan" (which needs a video-capable ReActor workflow not yet in `workflows/`).
> Independent step CLI commands make reordering trivial.

**Phase 2 — intelligence & scale:**
6. ✅ `evaluator.py` + retry loop (vision feedback) — `Verdict` (Pydantic), `evaluate_image`, `generate_keyframe_evaluated`; wired into `run_scene(evaluate=True)` and the `evaluate` CLI command.
7. ✅ Sequential continuity — `media.last_frame` + `run_sequence`/`Scene`; the `run-sequence` CLI (YAML scene list). See §13.
8. ✅ `state.py` — PostgreSQL-persisted `Job`/`Scene` (resumable; moved from SQLite 2026-09, see `db.py` + `migrations/`) + append-only decision log (`Store`, `Decision`, `record_verdict`). Structured logging still pending.
9. ✅ Multi-scene phased orchestration + semi-auto gates — `storyboard.py` (Gemma narrative) + `orchestrator.py` (`Orchestrator`: storyboard ⏸ → keyframes+eval ⏸ → videos, persisted to `Store`); CLI `run`/`approve`/`jobs`/`job`. Resumable from reloaded state; gates cleared via logged approvals.
10. ✅ `vram.py` — `gpu_free_mb`/`snapshot`/`log_state` (nvidia-smi, best-effort) + `reclaim()` guard at the Klein/ReActor→Wan boundary; `vram` CLI. Thin under Option B (Gemma on CPU).
11. ✅ `queue.py` + `api/` — async `JobManager` (Orchestrator in a thread executor, one-GPU lock, event Broker with ids + replay) → FastAPI `/api` + named SSE (`vorch serve`), `/media` from ComfyUI's output dir. *(The NiceGUI dashboard that first hosted this was replaced by the Next.js app in Sept 2026.)*

**Phase 3 — multipurpose raw material + setup phase (Sept 2026):**
12. ✅ `registry.py` + `workflows/*.yaml` — declarative workflow registry (kinds, params → node/field, contracts, `~/.vorch/workflows` overrides); `vorch registry inspect/add/validate` scaffolder. `docs/workflows.md`.
13. ✅ `formats.py` + `genparams.py` — aspect presets → Klein/Wan sizes; `GenParams` threaded through steps/pipeline/orchestrator; `derive_seed`; `media.fit_image`; `steps/edit.py` (reference-locked keyframes); `steps/postprocess.py` (post-step chains); `storyboard.PromptContext`; `Verdict.framing_ok` + `evaluate_video`.
14. ✅ `profiles.py` (+ `profiles_builtin/*.yaml`), `projects.py`, `envcheck.py`, `export.py`, `submission.py` — the setup phase; orchestrator plans by profile kind (video: 2 gates, stills: 1 gate + variants); PostgreSQL migrations 0001/0002.
15. ✅ `api/` routers: jobs, events, media, setup, workflows, profiles, projects (+assets), export; CLI `setup check`, `profiles`, `project …`, `export`, `db …`.
16. ✅ `frontend/` — Next.js control panel (jobs list, new-job wizard with env gate, job page with SSE timeline / variants / gate review / export, setup pages).

## 10. Open / deferred

- **Video evaluation reliability** — Gemma scores sampled frames; objective checks (motion happened, no morph between frames) are reliable, temporal smoothness is not. Flag subjective scores for human review, don't hard-block (`skills/gemma-evaluate.md`).
- **Option A (VRAM swapping)** — only if CPU Gemma eval proves intolerably slow. Start with Option B.
- **OpenClaw outer layer** — optional later, for chat-triggered / scheduled runs wrapping `vorch` as a single skill. Not in scope.

## 11. Application layer & stack (research-validated)

A deep-research pass (25 sources, adversarially verified) confirmed the deterministic-core
decision and selected a Python-native stack. The whole app runs as **one local process**.

- **Orchestration — deterministic Python core, Gemma as a called tool. No agent framework.**
  Anthropic's "Building Effective Agents": use *workflows* (predefined code paths) for
  well-defined tasks, reserve *autonomous agents* for problems whose steps can't be drawn up
  front. Our phase graph is fully drawable. Autonomous-agent orchestration runs ~10× the
  cost/latency and trades away the determinism we want. (Even LangGraph is really a deterministic
  state-machine graph — a legitimate *later* option if the hand-written machine gets unwieldy,
  used as a graph engine, not LLM planning.)

- **Structured output — constrain at the source + validate externally.** `llama-server`
  supports `response_format` with a JSON schema (or raw GBNF grammar on `/completion`); native,
  no extra decoding engine. **Function/tool calling requires the `--jinja` flag.** Wrap responses
  with **Instructor (Pydantic)** for typed `Verdict` objects. ⚠️ llama.cpp grammar can *fail open*
  (issue #19051) — never trust the grammar alone, always Pydantic-validate. XGrammar is
  vLLM/SGLang/MLC, **not** llama.cpp. (A benchmark claiming multimodal LLMs systematically fail
  at JSON-over-vision was *refuted* in verification — structured verdicts from Gemma-vision are
  reliable in practice.)

- **Backend — FastAPI + SSE + asyncio + PostgreSQL.** **SSE** (not WebSocket) for live progress —
  one-directional server→client, native in FastAPI (`EventSourceResponse`), handles cancellation.
  **asyncio** for the job queue — a broker (Celery/Redis/Dramatiq) is over-engineering for
  single-user local. State + decision log: originally SQLite; **superseded 2026-09 by the local
  PostgreSQL 16** already on the machine (psycopg 3 sync pool, JSONB job blobs, versioned SQL
  migrations, one throwaway schema per test). Same resumability, real concurrency, queryable blobs.

- **Frontend — Next.js** (`frontend/`; App Router, TypeScript, Tailwind, TanStack Query), the
  same stack as Georgi's other projects. It runs as its own Node server and *rewrites* `/api`
  and `/media` to FastAPI, so the browser stays same-origin and the API is a pure service with a
  typed client generated from its OpenAPI. **Supersedes (Sept 2026)** the original NiceGUI
  choice, which unified backend + frontend in one process; production is now two local
  processes (`vorch serve` + `next start`) plus PostgreSQL.

- **ComfyUI is itself a deterministic orchestrator.** `/prompt` (FIFO, returns uuid4 `prompt_id`),
  `/history` (poll), `/free` (reclaim VRAM), `/interrupt` (→ the cancel button), `--lowvram`
  (partial loading for 16GB). Accept/retry logic stays in Python, not LLM-planned.

### Recommended architecture (localhost; Next.js UI + FastAPI service + PostgreSQL)

```
┌─ Next.js :3000 ─────────────────┐ rewrites /api,/media
│ setup · wizard · job page (SSE) │ ──────────┐
└─────────────────────────────────┘           ▼
┌─ FastAPI :8000 (vorch serve) ────────────────────────────────────────────┐
│  /api + named SSE (ids, replay) · /media · env check · export             │
│  Deterministic state machine (the orchestrator core, profile-driven):     │
│    STORYBOARD ─⏸gate─ KEYFRAMES(+variants) ─⏸gate─ VIDEOS ─ POST ─ EXPORT │
│    JobManager: asyncio + one GPU lock; PostgreSQL (jobs JSONB, decisions, │
│    profiles, projects); registry = workflows/*.yaml (+ user dir)          │
└───────────────┬───────────────────────────────────┬──────────────────────┘
                │ HTTP (called as TOOLS)             │ HTTP
                ▼                                     ▼
   Gemma · llama-server :8080              ComfyUI :8188
   text: storyboard/prompts               /prompt (FIFO, prompt_id) · /history
   vision: eval → JSON Verdict            /free (VRAM) · /interrupt (cancel)
   --jinja · json_schema + Instructor     --lowvram
   = reasoning tool, NOT the driver       = deterministic generator
```

## 12. Sources (stack research)

- Anthropic — Building Effective Agents: https://www.anthropic.com/research/building-effective-agents
- LangGraph: Controlled Workflows, Not Autonomous Agents: https://medium.com/@egrois/langgraph-controlled-workflows-not-autonomous-agents-37332efa753f
- Agents vs Workflows (Fika Ventures): https://medium.com/fika-ventures/agents-vs-workflows-the-framework-founders-actually-need-519b5da8bd34
- llama.cpp server README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp GBNF grammars README: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- Instructor (structured output): https://python.useinstructor.com/integrations/ollama/
- Structured output with local models: https://simmering.dev/blog/structured_output/
- FastAPI — Server-Sent Events: https://fastapi.tiangolo.com/tutorial/server-sent-events/
- FastAPI background tasks: arq vs built-in: https://davidmuraya.com/blog/fastapi-background-tasks-arq-vs-built-in/
- NiceGUI — actions & events: https://nicegui.io/documentation/section_action_events
- Streamlit vs NiceGUI: https://www.bitdoze.com/streamlit-vs-nicegui/
- ComfyUI server.py: https://github.com/comfyanonymous/ComfyUI/blob/master/server.py
- ComfyUI memory/device management: https://deepwiki.com/Comfy-Org/ComfyUI/2.6-memory-and-device-management
- ComfyUI API guide: https://www.runflow.io/blog/comfyui-api-developer-guide

## 13. Sequential continuity (last-frame → next keyframe)

`run_sequence(scenes, master_face, …)` chains scenes so consecutive clips flow into each other.
Full design rationale: [`docs/sequential-continuity-plan.md`](docs/sequential-continuity-plan.md).

- **Two modes, opt-in.** Each `Scene` is a *cut* by default — a fresh Klein keyframe, identity-locked
  to the master via ReActor. Flag `continuous: true` and the scene is instead seeded by the **previous
  clip's last frame** (`media.last_frame`, ffprobe+ffmpeg). Scene 0 is always a cut.
- **Drift control.** Wan drifts identity across a clip; chaining compounds it. Carried frames are
  re-locked to the master (ReActor swap) before animating — default on, `--no-relock` to disable.
- **Phasing tension.** Continuity is inherently sequential (scene N depends on N-1's video), so a
  chained run **cannot** batch the Wan phase the way NOTES §5 prescribes. Klein loads once for the cut
  scenes; the Wan+ReActor chain then runs scene-by-scene. Resolution aligns for free — a carried
  832×480 frame is already Wan-native.
