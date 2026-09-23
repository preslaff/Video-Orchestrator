---
name: fix-frame
description: >-
  Diagnose and repair a distorted keyframe or clip in a vorch job — malformed hands, product
  parts that do not connect (bag handles, straps), garbled brand text, wrong face, broken scene
  logic. Works out which of vorch's repair instruments fits the defect, drafts the corrective
  prompt, and asks before spending anything on a render. Use whenever a generated frame is
  wrong and the user wants it fixed. Triggers — "fix this frame", "the handles are distorted",
  "bad hands", "the label is garbled", "this keyframe is wrong", "repair scene",
  "fix the distortion", "поправи кадъра", "дефект в кадъра".
---

# Fix a distorted frame

Repairing a bad generation in vorch is a diagnosis, not a prompt tweak. The instrument depends
on the defect, and choosing wrong wastes a paid render (≈ $0.40–$1.15 per regenerated scene on
Modal) while leaving the frame exactly as broken.

**Read `skills/fixing-distortions.md` first** — the decision table, the prompt rules and the
pitfalls live there and are not repeated here. This file is only the procedure.

Repair is **HTTP-only**: the CLI has no subcommand that edits a scene. Everything below goes to
`vorch serve` on `localhost:8000`.

## Procedure

### 1. Gather

```bash
JOB=<job id>; I=<scene index>
curl -s localhost:8000/api/jobs/$JOB            # scenes, verdicts, variants, clips, edit_reference_slots
curl -s localhost:8000/api/jobs/$JOB/decisions  # why the evaluator retried, warnings
```

`GET /api/jobs/{job}/scenes/{i}/versions/{version_id}` is the single best "why is this frame
bad" call: seed, prompt, negative, params, workflow, and `matches_scene_prompt` (which catches
the frame that was rendered before the prompt was edited).

### 2. Look

Extract the frame **and every candidate reference** and view them at native resolution. Crop into
the defect — a 576×1024 frame shown whole hides exactly the detail in question.

```bash
ffmpeg -v error -y -i <frame> -vf "crop=400:460:140:340,scale=iw*1.6:-1" /tmp/defect.png
```

### 3. Diagnose — delegate to `frame-doctor`

Launch the `frame-doctor` agent with the job id and scene index. It is read-only, it looks at
the frame and the references, and it returns: the defect class, **whether the reference already
carries the defect**, the recommended instrument with exact field values, a draft prompt, and —
if nothing in the project fits — the photo to ask for. Delegating keeps the images out of this
conversation's context.

Do not skip the reference check. It is the most common cause and the cheapest to rule out.

### 4. Rehearse — free of job state

Before mutating anything, render one candidate outside the job. This touches no scene, clears no
outputs and reopens no gate:

```bash
python -m vorch edit "<corrective prompt>" \
  --ref <the scene's current keyframe> --ref <corrective reference> --workflow kv_edit
```

Look at the result. Iterate on the prompt here, not in the job.

### 5. Propose, then ask

State the instrument, every field you will set, the prompt verbatim, and why that instrument and
not the others. Then **ask before rendering**. Never regenerate a scene without a yes.

### 6. Apply

```bash
# Surgical repair of the existing frame
curl -s -X POST localhost:8000/api/jobs/$JOB/scenes/$I/regenerate \
  -H 'content-type: application/json' -d '{
    "what": "keyframe", "mode": "edit", "edit_base": "keyframe", "edit_workflow": "kv_edit",
    "keyframe_prompt": "<end-state prompt>", "reference_asset_ids": ["<asset>"]}'

# Take the evaluator's own advice          # Substitute a known-good image
-d '{"what":"keyframe","apply_suggestion":true}'   -d '{"what":"keyframe","keyframe_asset_id":"<asset>"}'

# Only the motion is wrong                 # Protect small type the animator smears
-d '{"what":"video","animation_prompt":"..."}'     # PUT /scenes/$I/locks {"regions":[{...}]}

# Free: an earlier take was better
curl -s -X PUT localhost:8000/api/jobs/$JOB/scenes -H 'content-type: application/json' \
  -d '{"scenes":[{"index":'$I',"chosen_variant_id":"<vid>"}]}'
```

Mutating calls return **409 while a segment is in flight** — `POST /api/jobs/$JOB/stop`, poll
`GET /api/jobs/$JOB` until `active:false`, then retry.

### 7. Verify

Re-read the scene's verdict and judge by the **objective gate**, not `score`:
`action_correct ∧ identity_match ∧ composition_ok ∧ ¬artifacts ∧ framing_ok is not False`
(`artifacts: true` means bad). Then compare the same crop before and after, at native size.

If the repair worked and the frame is worth reusing, promote it:
`POST /api/jobs/$JOB/scenes/$I/save-asset {"kind":"reference","label":"..."}`.

## When to ask for a photo instead

If no project asset shows the object correctly, stop and say exactly what is needed — subject,
angle, what must be visible, minimum resolution. A repair driven by a reference that carries the
same defect cannot succeed, and rendering it anyway spends money to reproduce the fault.

Register what you are given before using it:
`python -m vorch project add-asset <slug> <path> --kind product --label "<what it shows>"`.

## Pitfalls

- **`kv_edit` accepts no `negative`, no `width`, no `height`.** Its size follows the first
  reference. Negatives only work on `t2i` and `identity_edit`.
- **`identity_edit` locks composition** — it conditions a *fresh render* on the references. For
  "keep this frame, change one thing" the answer is always `kv_edit` + `edit_base=keyframe`.
- **Never use kv_edit to restore brand text.** It produced `BEOWER BHORMNCE`
  (`docs/label-fidelity.md`). Text means locked regions on a clip, compositing on a still.
- **Regenerating a keyframe drops `locked_regions`** when the marks no longer fit. Re-mark after.
- Both edit workflows take exactly **2 references**; with `edit_base=keyframe` that is the frame
  plus one. A third is silently ignored.
