# Motion-transfer scenes (v1, Wan-Animate-2) — implementation plan

> Status: planned 2026-09-05, not started. Design decisions agreed with Georgi; see `modal/workflows/animate_wan_animate2.yaml` for the deployed post-step this builds on.

## Context

Georgi wants a workflow where a **reference video supplies only the action and movement**, and the new clip shows a different character, background and props. Wan-Animate-2 does exactly this (driving video → motion; reference image → character; prompt → background) and is already deployed on the Modal L40S as a post-step that reuses the job's own clip (`modal/workflows/animate_wan_animate2.yaml`, measured 2.2–2.6 min per 81 frames). Version one makes it a first-class way to produce a scene's clip: the driving video is a **project asset**, the scene's Klein keyframe (after the optional face swap) is the character reference, and the clip comes from a `motion` workflow instead of image-to-video. Everything downstream (post-steps, assembly, export, evaluation, gates) stays as it is.

**v1 limits:** one Animate-2 segment per scene (the driving video is resampled to the profile's fps and capped at the workflow's max length, 81 frames ≈ 5 s @ 16 fps; longer references are trimmed with a warning); the background follows the prompt (a "keep my keyframe background" variant via Wan 2.2 Fun Control and segment chaining are later steps).

## Design decisions

- **`mode` stays the keyframe method (`t2i` | `edit`).** Motion is orthogonal: a new per-scene field `driving_video` (local path). When set, the video phase runs motion transfer; when null, image-to-video as today. This avoids touching the five places that enumerate `mode` and lets a motion scene still use an edit-mode keyframe.
- **New registry kind `motion`** with its own contract, separate from `postprocess` (which stages the job's *own* clip). The Wan-Animate-2 template is shared; two YAMLs point at it: the existing post-step `animate_wan_animate2` and the new `motion_wan_animate2`.
- **Reference image = the scene's keyframe (swapped if face swap is on).** The keyframe is the character design the evaluator already approved, in the intended outfit — Animate-2 takes clothing from the reference.
- **Prompts:** `positive` = the scene's keyframe prompt (character + setting → background), `motion` = the scene's animation prompt. For motion scenes the storyboard step lets Gemma **look at the reference video** (3 sampled frames via `media.sample_frames` + `gemma.complete_vision`) and write the motion sentence from what it sees, instead of inventing camera moves.

## Plan

### 1. Registry + workflow (`vorch/registry.py`, `modal/workflows/`)

- `KINDS += "motion"`; `KIND_CONTRACTS["motion"] = Contract(required={driving_video, reference_image, positive}, optional={negative, motion, width, height, length, fps, seed, steps, reference_strength, filename_prefix}, min_image_inputs=1, min_inputs=2)` + a check that one input is `video` (extend `_contract_problems`).
- `modal/workflows/motion_wan_animate2.yaml`: kind `motion`, backend `modal-l40s`, same `wan_animate2_replace.json`, params renamed to the contract (`driving_video` → node 1 `file`, `reference_image` → node 3 `image`, `positive`/`motion`/`negative`, `width`/`height` (multiple_of_16, ≤1280), `length` (4n+1, 5–81), `seed`, `steps`, `reference_strength`, `filename_prefix`). Keep `animate_wan_animate2.yaml` unchanged. Add to `tests/test_modal_assets.py` EXPECTED; docs/workflows.md kind table.
- `formats.workflow_rules` already reads `length`/`width` constraints by name — reuse it for the motion workflow (`size_max`, `length_constraint`, and add `length_max` from the param's `max`).

### 2. Media helper (`vorch/media.py`)

- `prepare_driving_video(src, dest, *, fps, max_frames) -> MediaInfo`: ffmpeg `-i src -vf fps=<fps> -frames:v <max_frames> -an -c:v libx264 -pix_fmt yuv420p dest` (`.part` → rename, `MediaError` on failure), then `probe(dest)`. Caller computes `length = largest 4n+1 ≤ min(max_frames, nb_frames)` (≥ 5). Test with a synthesized 30 fps clip (`_clip_with_audio` pattern in `tests/test_media_assemble.py`): 3 s → 16 fps → 48 frames → length 45; 10 s → capped at 81.

### 3. Data model + API (`vorch/state.py`, `vorch/api/schemas.py`, `vorch/api/public.py`, `vorch/queue.py`, `vorch/submission.py`, `vorch/projects.py`)

- `Scene.driving_video: str | None` and `Job.driving_video: str | None` (default for new scenes, copied at storyboard like `references`); add to `_scene_to_dict/_scene_from_dict`, `job_to_dict/job_from_dict`, `new_job`. Orchestrator `SCENE_EDIT_FIELDS` and the `queue.update_scenes` allow-list gain `driving_video`.
- Projects: `ASSET_KINDS += "video"` (`AssetKind` literal); `api/projects.upload_asset` unchanged (bytes only) — add a light check: kind `video` requires a video extension (`media.VIDEO_EXTS`).
- `JobSpec`: `driving_video_asset_id: str | None = None`, `driving_video: str = ""` (path). `submission.resolve_submission` resolves the asset id to its path (KeyError → 404 like references); `Submission.driving_video`. `SceneEdit` / `SceneRegenerate`: `driving_video: str | None` (`""` clears). `ScenePublic`/`JobPublic`: `driving_video`, plus `driving_video_url` when the path is under `COMFY_OUTPUT` (the prepared copy) — the asset itself is served by `/api/projects/{id}/assets/{aid}/file`.
- `orchestrator.create_job(..., driving_video=None)`: when set, `media.probe` it (reject non-video with a clear message) and warn (decision-log entry) if longer than the workflow's max frames at the profile fps.

### 4. Pipeline + orchestrator (`vorch/pipeline.py`, `vorch/orchestrator.py`, `vorch/genparams.py`, `vorch/profiles.py`, `vorch/storyboard.py`)

- `GenParams.motion_workflow: str = "motion_wan_animate2"`; `profiles.VideoSettings.motion_workflow` (optional, same default); `GenParams.from_profile` maps it; `profile.workflows_used()` includes it **only when a job actually uses motion** (env check must not warn for every profile) — simplest: the orchestrator validates the motion workflow's presence in `create_job` when `driving_video` is set (`registry.load_registry().get(...)` → clear error).
- `pipeline.animate_from_video(driving_video, reference_image, positive, motion, *, seed, params, negative=None, filename_prefix=None, prepared_dir) -> Path`:
  1. `backend = backends.for_workflow(params.motion_workflow)`; rules = `formats.workflow_rules(params.motion_workflow)`.
  2. `media.prepare_driving_video(src, prepared_dir / f"{prefix}_driving.mp4", fps=params.fps, max_frames=rules.length_max)` → `length`.
  3. upload both files to `backend`; `vram.reclaim(backend=backend)`; `run_step(params.motion_workflow, driving_video=..., reference_image=..., positive=..., motion=..., negative=..., width/height=params.wan_size, length=length, fps?=only if declared, seed=..., filename_prefix=...)`.
- `orchestrator._phase_videos`: per scene, after the (optional) swap: `if scene.driving_video: video = pipeline.animate_from_video(scene.driving_video, frame, scene.keyframe_prompt, scene.animation_prompt, ..., filename_prefix=output_prefix(job, i, "clip"))` with a `step="motion"` event `{index, length, fps, seconds, source: basename}`; else the existing `animate_frame`. Post-steps unchanged (they receive the motion clip). `regenerate_scene(what="video")` works as is (clears `video_path`).
- `_phase_storyboard`: copy `job.driving_video` onto scenes; for motion scenes call `storyboard.animation_prompt(brief, context=..., reference_frames=media.sample_frames(driving_video, n=3))` — new optional argument that switches the system prompt to "describe the performer's action seen in these frames in one sentence; the camera is static; no new actions" and uses `gemma.complete_vision`. Falls back to the text-only prompt if vision fails (log a decision).
- `PromptContext.motion_source: str | None` (the asset name) so `storyboard_lines()` can tell the storyboard model that the action is given by a reference video (scenes describe character/setting/mood, not choreography).

### 5. Frontend

- Project page (`frontend/src/app/projects/[id]/page.tsx`): `KINDS += 'video'`; file input `accept` becomes `video/*` when kind is `video`; asset grid renders `<video controls preload="metadata">` for `kind === 'video'` (reuse `MediaThumb` with `video`). Empty-state copy mentions motion reference videos.
- Wizard (`frontend/src/components/NewJobWizard.tsx`): Brief step gains **Motion source** (`data-testid="motion"`): "none (Wan image-to-video)" or one of the project's `video` assets, with a `<video>` preview and the note "≤ 5 s at 16 fps is used; the keyframe is the character, the background follows the prompt; keep the framing (full body / bust) consistent with the reference". `refAssets` excludes `video` assets. JobSpec gets `driving_video_asset_id`. Review shows "motion: <asset>". Test: select the asset → posted body has `driving_video_asset_id: 'a1'`.
- GatePanel: per-scene `driving video` input (path; shown for video jobs; editable while the clip is missing), saved via `SceneEdit.driving_video`, sent with regenerate `what: 'video'`; button label "Animate (motion)" when set.
- SceneGrid: chip `motion · <basename>`; `events.ts` `describeStep` case `'motion'` → "Transferring the motion of <source> onto scene N with Wan-Animate-2 — L frames @ fps (~s)". `make api-types`.

### 6. Docs + install

- README (features, Jobs section: motion mode; UI table rows), `docs/workflows.md` (kind `motion` contract), `docs/modal-backend.md` §5 row, `modal/README.md` (copy `motion_wan_animate2.yaml`), memory note.
- Copy the new YAML into `~/vorch/workflows/` (Georgi's install) and upload a short reference video as a `video` asset of the perfume project for the live test.

## Commits

1. `registry: kind 'motion' + motion_wan_animate2 workflow; media.prepare_driving_video` (+ tests).
2. `motion scenes: driving_video on jobs/scenes, video asset kind, submission, pipeline.animate_from_video, orchestrator motion step, storyboard sees the reference frames` (+ tests: state round-trip, submission, orchestrator with fakes, storyboard fallback).
3. `ui: motion source in the wizard, driving video in the gate panel, video assets on the project page, motion step label` (+ wizard test, openapi).
4. `docs: motion-transfer scenes` + live run.

## Verification

- `pytest -q` (new: contract, media helper with real ffmpeg, state/submission/orchestrator/storyboard fakes, API schema) and `cd frontend && npm run lint && npx tsc --noEmit && npm test && npm run build`.
- Live: upload a 3–5 s reference video (kind `video`) to Perfume Echoes; new job on the fp8 9:16 profile with motion source = that asset, one scene, autonomy full. Expect: storyboard with a motion sentence written from the frames, Klein keyframe, swap, `step: motion` event, prepared `s00_clip_driving.mp4` next to the outputs, the clip downloaded as `s00_clip_00001_.modal-l40s.mp4` (~2.5 min on the L40S), manifest `workflow: motion_wan_animate2`, export folder complete. Then regenerate the clip from the gate panel with an edited motion prompt.
