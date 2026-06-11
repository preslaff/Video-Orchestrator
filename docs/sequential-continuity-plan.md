# Sequential Video Continuity — Design & Build Plan

> **Status:** Designed, not yet built. Two design decisions locked (see [Decisions](#decisions-locked)).
> Belongs to **Phase 2** of the roadmap, on top of the committed single-scene driver and vision evaluator.

## Goal

Generate a *sequence* of scenes where the **last frame of scene N's video** becomes the
**start image for scene N+1's animation**, so consecutive clips flow into each other for visual
consistency. This ships as a focused continuity primitive (`last_frame` extraction) plus a sequence
runner (`run_sequence`) built on the existing `run_scene` building blocks.

```
scene 0:  text → Klein keyframe → ReActor(master) → Wan → clip₀
scene 1:  clip₀.last_frame ──────► ReActor(master) → Wan → clip₁   (continuous)
scene 2:  text → Klein keyframe → ReActor(master) → Wan → clip₂   (cut)
```

## Design rationale

### Two continuity modes — both needed
Feeding the exact last frame forward produces a **continuous shot**: the next clip literally
continues the image. Ideal for flowing motion within the same context — but it also **carries over
the old composition** (the same "reference locks composition" caveat as `identity_edit`,
NOTES §3). A hard scene change to a new location must therefore **not** chain the frame:

- **continuous** → start image = previous clip's last frame.
- **cut** → start image = a fresh Klein keyframe from text (identity locked via ReActor to master).

### Drift accumulates when chaining
Wan drifts the face/objects across a clip; seeding the next clip from the drifted last frame
compounds the error. **Mitigation built in:** re-apply ReActor (swap master face) onto each carried
last frame before animating, re-locking identity every scene.

> *Optional future gate:* evaluate the carried frame and fall back to a fresh keyframe if it has
> degraded past threshold.

### Phasing tension
NOTES §5 batches "all keyframes → all videos → all swaps" to minimize model loads. Continuity is
**inherently sequential** — scene N's start depends on N−1's video — so a chained run cannot batch
the video phase. For a sequence: Klein loads once (scene 0 + any cut scenes), then the Wan+ReActor
chain runs scene-by-scene.

> *Future hybrid for the orchestrator:* pre-batch all *cut* keyframes (Klein) up front, then run the
> sequential Wan chain. Noted, not built now.

### Resolution aligns for free
Wan outputs 832×480; a carried last frame is already 832×480 — directly Wan-compatible (no resize),
unlike the scene-0 Klein keyframe (1024×576).

### Last-frame extraction
Reuse the ffprobe + ffmpeg pattern from `skills/gemma-evaluate.md`:

```bash
ffprobe -count_frames … nb_read_frames          # → TOTAL
ffmpeg -i clip -vf "select=eq(n\,TOTAL-1)" -frames:v 1 last.png
```

Exact; clips are ~49 frames so cost is trivial. Fallback `-sseof -0.2` if probing fails.

## Decisions (locked)

- **Opt-in chaining.** Every scene gets its own fresh Klein keyframe (identity locked to master)
  **unless** flagged `continuous: true`, in which case its start image is the previous clip's last
  frame. Scene 0 is always fresh — no previous frame to chain from.
- **Re-lock identity each scene.** Carried last frames get a ReActor swap to master before
  animating (default on; `--no-relock` to disable).

## Build steps

1. **`vorch/media.py`** *(new)* — `last_frame(video_path, dest=None) -> Path` via ffprobe+ffmpeg.
   Pure I/O, GPU-free, unit-testable by monkeypatching `subprocess`.

2. **`vorch/pipeline.py`** — add:
   - `@dataclass Scene(keyframe_prompt, animation_prompt, continuous: bool = False)`.
   - `run_sequence(scenes, master_face, *, seed, evaluate, relock_identity=True, max_attempts) -> list[SceneResult]`.
     For each scene: if `i == 0` or **not** `scene.continuous`, build a fresh Klein keyframe (reuse
     `keyframe` / `generate_keyframe_evaluated`) and ReActor-swap to master; else (continuous)
     extract `last_frame(prev_video)` and — if `relock_identity` — ReActor-swap it to master. Stage
     via `comfy.upload_image`, `comfy.free()` before Wan, `animate(...)`, record the result, carry
     the video forward.
   - Extend `SceneResult` with `start_image: Path` and `continued_from_previous: bool`.

3. **`vorch/__main__.py`** — `run-sequence scenes.yaml --master … [--evaluate] [--seed] [--no-relock]`.
   Scenes file is YAML (aligns with the locked YAML job-spec decision): a list of
   `{keyframe_prompt, animation_prompt, continuous?}`. Adds `pyyaml` to core deps.

4. **`tests/test_sequence.py`** *(new, GPU-free)* — monkeypatch `keyframe`, `faceswap`, `animate`,
   `last_frame`, `comfy.upload_image/free`. Assert: scene 0 uses a fresh keyframe; a `continuous`
   scene pulls `last_frame` of the previous clip; a non-continuous scene builds a fresh keyframe;
   identity re-lock is applied to carried frames when enabled; the video carries forward correctly.
   Plus a `media.last_frame` test with a faked `subprocess`.

5. **Docs** — `DESIGN.md`: add a "Sequential continuity" subsection (two modes, drift/re-lock,
   phasing tension) and mark it in the build order. `NOTES.md`: short note that continuity chaining
   is sequential (diverges from §5 batching). `README.md`: `run-sequence` usage.

## Reuse (do not re-implement)

- `vorch/pipeline.py`: `run_scene`, `generate_keyframe_evaluated`, `SceneResult`.
- `vorch/steps/{keyframe,animate,faceswap}.py`; `vorch/clients/comfy.py` `upload_image` / `free` / `output_path`.
- `vorch/evaluator.py` `evaluate_image`; the ffmpeg pattern from `skills/gemma-evaluate.md`.

## Verification

- **GPU-free:** `.venv/bin/python -m pytest -q` — existing 20 + new sequence/media tests pass.
- **CLI parses:** `python -m vorch run-sequence --help`.
- **Live** (needs ComfyUI; manual, local): a 2-scene YAML where scene 2 is `continuous` → confirm
  scene 2's clip visibly starts on scene 1's final frame; a `cut` scene → confirm a fresh keyframe.
