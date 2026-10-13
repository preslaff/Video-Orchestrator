# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **local agentic orchestrator for AI video production**. It turns a text idea into a series of consistent video clips, autonomously, by directing two local systems in a self-evaluating loop:

- **Gemma 4** (`gemma-4-26B-A4B-it`, MoE 26B/~4B active, multimodal) — the "brain + eyes": writes storyboards/prompts and **evaluates** every generated result via vision, returning structured JSON verdicts (accept/retry).
- **ComfyUI** — the "hands": runs image/video/face-swap workflows.

Pipeline per scene: `keyframe prompt → image → EVALUATE → animation prompt → video → EVALUATE → face swap → EVALUATE`. After each generation Gemma scores it and decides accept or retry-with-corrected-prompt (max N attempts).

Hardware target: **RTX 5070 Ti, 16GB VRAM**, 64GB RAM, Xubuntu. The 16GB ceiling drives most architecture decisions.

The project documentation and the user's working language is **Bulgarian** (user: Георги / Georgi). Keep that in mind when reading/writing notes; code and identifiers are English.

## Current status: implemented (Phases 1–3), Python package `vorch/` + Next.js `frontend/`

The orchestrator is built and validated live: `vorch/` (~9k LOC, `pytest -q` ≈ 260 GPU-free
tests) and a Next.js control panel. Read `README.md` for how to run it and `DESIGN.md` §9 for
the build order actually followed. Key facts a new session must not re-derive:

- **Product scope (Sept 2026):** a multipurpose *raw-material* generator for cinematic and
  marketing use — video 16:9 / 9:16 / 1:1, stills-only jobs with variants, longer clips,
  reference-locked edits, post-process chains. Final editing happens in Shotcut, out of scope;
  the app ends with a per-job **export folder** (`vorch/export.py`), optionally with `assembly.mp4`
  (selected clips joined with ffmpeg `xfade` transitions: `vorch/assembly.py`, `media.assemble`).
- **Setup phase:** output **profiles** (`vorch/profiles.py`, built-ins in `profiles_builtin/`),
  **projects** with assets (`projects.py`), **environment check** (`envcheck.py`, blocks submit on
  fail), **submission** resolution (`submission.py`). Jobs snapshot their profile (`Job.profile`).
- **Workflows are pluggable by YAML** (`vorch/registry.py`, `workflows/*.yaml`, user dir
  `~/vorch/workflows`; `vorch registry inspect|add|validate`). Node-id surgery lives only there.
- **State is PostgreSQL** (`vorch/db.py`, `migrations/*.sql`, DSN `postgresql:///vorch`, tests
  use one throwaway schema each). SQLite is gone.
- **UI is Next.js** (`frontend/`, proxies `/api` + `/media` to `vorch serve`). NiceGUI is gone.
- **Formats:** `formats.py` (aspect → sizes), `genparams.py` (flat params threaded through
  steps/pipeline/orchestrator). Stills jobs have one gate; video jobs two.

Working rules: keep changes surgical, keep `pytest -q` green, regenerate `frontend/openapi.json`
(`make api-types`) whenever routes change (a test enforces it), and record decisions in the docs.

## Documentation map (read these, don't duplicate them)

- **`NOTES.md`** — the spec. Endpoints, start commands, exact node-IDs to replace per workflow (§3), VRAM strategy (§4), pipeline phasing (§5), evaluation criteria (§6), finalized decisions (§7), dev phasing (§9). **Read first; everything assumes it as context.**
- **`DESIGN.md`** — the orchestrator architecture: module layout, the workflow registry, data model, YAML job spec, control flow with semi-auto gates, and build order (§9, incl. Phase 3). Also §11–12: the stack — **deterministic Python core (no agent framework); Gemma via llama.cpp json_schema + Pydantic; FastAPI + SSE + asyncio on PostgreSQL; Next.js frontend (proxying /api); two local processes.** Don't re-litigate these.
- **`docs/services.md`** — the five external services vorch drives, their **measured** VRAM cost (total 16303 MiB; gemma 1838 even with `--n-gpu-layers 0`; comfy 222 idle / ~22000 with Wan; tts **11268**), why OmniVoice and a Wan render cannot coexist, and what vorch will deliberately never manage. Read before touching anything that starts or stops a service.
- **`docs/workflows.md`** — how to register a newly installed ComfyUI workflow (YAML sidecar, kinds, contracts, post-process chains).
- **`docs/modal-backend.md`** — the optional per-workflow cloud backend (Modal): `backends.yaml`, the download seam in the ComfyUI client, deployment, model comparison, cost and risks. Local stays the default.
- **`docs/model-tuning-options.md`** — whether to fine-tune or stay frozen (LoRA, IC-LoRA control adapters, video LoRA on Modal), compared against the failures this project actually has. Read before anyone proposes training anything: it records that **text fidelity is a VAE problem, not a weights problem**, that two LTX motion-control IC-LoRAs are already on disk unused, and where a LoRA would plug into the registry.
- **`docs/layered-compositing-plan.md`** — the staged plan for turning `locked_regions` from a tracked rectangle into a tracked **shape** (per-pixel SAM2 matte), and the answer to "can we build the layered architecture tool?". Records that **`tracking.paste` is already alpha-over** so the compositor exists, that phase 1 deliberately never persists an alpha channel, and the verified SAM2 install for ComfyUI's venv. Phases 2+ (an RGBA `layer` kind) are costed, not designed.
- **`SKILL.md`** — router skill + architectural invariants. Points to:
  - `skills/comfyui-api.md` — load template, replace params, POST `/prompt`, poll `/history`, extract outputs
  - `skills/gemma-evaluate.md` — vision evaluation, image + video (frame extraction), JSON verdict parsing
  - `skills/gemma-prompt.md` — storyboard + keyframe/animation prompt generation; learned FLUX.2 prompting rules
  - `skills/vram-manager.md` — avoiding OOM between models
  - `skills/fixing-distortions.md` — repairing a bad frame: which instrument for which defect (kv_edit vs identity_edit vs t2i vs locked regions), how to write a corrective prompt, how to pick a reference. The `/fix-frame` skill (`.claude/skills/`) drives it and the read-only `frame-doctor` agent does the looking.

## Architecture invariants (from SKILL.md — do not violate)

1. **VRAM is the binding constraint (16303 MiB measured).** Never assume two heavy models coexist. **Measured 2026-09-24, not estimated:** Gemma **1838 MiB — yes, even with `--n-gpu-layers 0`** ("on CPU" is true of compute, not memory); ComfyUI 222 idle and ~22000 with the Wan dual-model (only fits with RAM offload); Klein ~8GB; OmniVoice narration **5726 MiB fresh, 11268 after a long session** (the manifest reserves the worst case). So `1838 + 11268 + 222 = 13328` leaves 2975 MiB — **narration and a render do not fit together**, which makes phase-scoping the narration server an invariant, not an optimisation. Free VRAM between heavy steps via ComfyUI `POST /free {"unload_models": true, "free_memory": true}`. Default strategy is **Option B: Gemma pinned to CPU** (`--n-gpu-layers 0`) so ComfyUI owns the GPU. See `docs/services.md`, `NOTES.md` §4, `skills/vram-manager.md`.
2. **Workflows are templates, not finals.** Load the JSON, replace **only** the fields listed in `NOTES.md` §3 by node ID, leave the rest untouched. Never rewrite graph structure.
3. **Phase by stage, not by scene** (`NOTES.md` §5): generate all keyframes (Klein loaded once) → `/free` → all videos (Wan) → `/free` → all swaps (ReActor). Minimizes model load/unload churn.
4. **Wan video length = 4n+1.** `length` in `WanImageToVideo` accepts only 49, 81, 121… not arbitrary values.
5. **Gemma vision requires `--mmproj`.** If a verdict is nonsense or "I can't see the image", the llama-server was started without the mmproj projector.
6. **Identity via post-hoc ReActor swap, not per-scene reference editing.** Reference editing (`identity_edit`) **locks composition** — use it only for "same composition, light edit". For new scenes use text-to-image, then impose the face with ReActor. The face is chosen **per scene** (`Scene.identity`, a project `master_face` asset assigned by the storyboard from the assets' labels or by the user at a gate; empty = no swap); `Job.master_face` is only the default. (`NOTES.md` §3, `skills/gemma-prompt.md`.)
7. **Evaluation returns JSON, not prose.** Always prompt Gemma for structured output (`NOTES.md` §6) and parse it leniently (Gemma sometimes wraps in ```json fences). Temperature **0.2 for evaluation** (consistent), **0.7 for generation** (creative) — don't mix them up.

## Services & endpoints

- **Gemma**: OpenAI-compatible `http://localhost:8080/v1/chat/completions`. Image input as base64 in an `image_url` content block. Start command + model paths in `NOTES.md` §2.
- **ComfyUI**: `http://localhost:8188`. Run = `POST /prompt`; result = poll `GET /history/{prompt_id}`. Outputs land in `~/comfy/ComfyUI/output/` (video in `output/video/` subfolder); `LoadImage` nodes read from `~/comfy/ComfyUI/input/` (use `POST /upload/image` to stage a generated file for the next step).

> Note: the `gemma4-local` / `llama-mcp` MCP servers exposed to Claude Code (`llm_chat`, `llm_review`, etc.) wrap the same local model but are **Claude's own tooling** — distinct from the orchestrator's direct HTTP calls to port 8080, which is what the application code will use.

## Workflow files

The five ComfyUI workflow templates in `workflows/` (names in `NOTES.md` §3 and `skills/comfyui-api.md` match disk):

| File in `workflows/`                          | Purpose                              |
|-----------------------------------------------|--------------------------------------|
| `image_flux2_klein_text_to_image.json`        | keyframe from text                   |
| `klein_identity_edit.json`                    | keyframe with locked composition     |
| `wan22_i2v_gguf.json`                         | animate keyframe → video             |
| `ReActor_Face_Swap.json`                      | impose master face                   |
| `image_flux2_klein_9b_kv_image_edit.json`     | reserve KV-cache edit variant        |

**Asset/reference filenames are runtime inputs, never hardcoded.** The `.inputs.image` values currently in the workflow JSONs (`Joro.jpg`, `maj.jpeg`, `FaceSwap_*.png`, etc.) and the names in `NOTES.md` are inert leftover defaults — at runtime the orchestrator overwrites them by node ID, exactly like prompts and seeds (invariant #2). The user (or the pipeline) supplies the master face, identity references, and chained intermediate files dynamically; treat `assets/jena-cropped.jpeg` as the current sample master, not fixed config.

All workflow JSONs are **ComfyUI API format** (flat dict keyed by node ID), not UI export (which has a `nodes` array and won't work with `/prompt`). Node IDs in the templates can be compound (e.g. `75:73`, `75:68`). Each template has a **sidecar YAML** (`workflows/<name>.yaml`) that is the executable source of truth for which `(node, field)` pairs are replaced — `NOTES.md` §3 is the original prose version. Five entries ship: `keyframe`, `animate`, `faceswap`, `identity_edit`, `kv_edit`.

## Working style (Georgi's preference)

- **Surgical, well-understood changes — not quick fixes.**
- Before any heavy operation (generation, loading a large model), confirm VRAM state.
- Log the evaluator's decisions (why a frame was accepted/retried) for later debugging.
- Objective checks (action_correct / identity_match / artifacts) are reliable → automate them. Subjective aesthetics (mood/score) are weaker → flag for human review, don't hard-block.
