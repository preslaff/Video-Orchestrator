# Fine-tuning and frozen-weight options — a comparison

> Status: research note, 2026-10-12. **No decision taken.** Written to settle what is worth
> spending time on before any of it is attempted. Machine facts below were checked on the box
> that day; anything estimated is marked as such.

## Why this note exists

The question was whether to fine-tune the image and video models, or to keep the weights frozen
and add LoRA or similar. Both are possible. Neither is worth doing unless it fixes a failure this
project actually has, so that is where the comparison starts.

## What this project actually gets wrong

| failure | current mitigation | can weights or adapters fix it? |
|---|---|---|
| Brand text mangled by the video model | `locked_regions`, tracked and pasted back per frame | **No.** The video VAE compresses 8× spatially; the glyphs are gone before any weight matters. Settled in [`label-fidelity.md`](label-fidelity.md). |
| Identity drifts between scenes | ReActor post-hoc swap, `identity_edit`, Wan-Animate-2 | Yes, but it is already solved twice. Marginal gain. |
| The look varies scene to scene | a style prefix in the prompt | **Yes — this is what a LoRA is for.** |
| No control over camera or character motion | prompt the animate step and hope | **Not a weight problem.** It is a conditioning problem. |
| Distortions (hands, product parts that do not connect) | `/fix-frame`, regenerate | Partly. A LoRA biases toward your subject matter; it does not fix structure. |

Two conclusions fall out before comparing anything:

- **Do not fine-tune for text fidelity.** This is the most common way to waste weeks on this
  problem. The compression happens in the VAE, downstream of the weights.
- **Motion wants conditioning, not training.** Which is what makes route A interesting.

## What is on the machine (verified 2026-10-12)

```
models/diffusion_models/
  flux-2-klein-base-4b-fp8.safetensors      4.1 GB   base, 20+ steps, cfg 5
  flux-2-klein-4b-fp8.safetensors           4.1 GB   distilled, 4 steps, cfg 1
  Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf    10.8 GB   two-expert MoE, one half
  Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf     10.8 GB   the other half
  wan2.2_s2v_14B_fp8_scaled.safetensors    16.4 GB
  ltx-video-2b-v0.9.5.safetensors           6.3 GB   old 2B line, superseded
  flux1-dev.safetensors                    23.8 GB

models/loras/
  ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors   313 MB   ** unused **
  ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors          625 MB   ** unused **
  wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors   1.2 GB   in animate_wan_animate2
  Wan2.2-I2V-High-MysticXXX.safetensors                         293 MB
```

Card: RTX 5070 Ti, 16303 MiB. Disk: 693 GB free. Remote: `modal-l40s` at **$1.95/h**,
`modal-h100` at **$3.95/h** (declared in `~/vorch/backends.yaml`).

Dataset material: **137** `reference` project assets, **4** `product`, **1** `master_face`,
**6** `video`, and **343** generated keyframes under `COMFY_OUTPUT/vorch/`.

## Route A — control adapters (IC-LoRA), weights frozen

Two LTX control adapters are already downloaded and **referenced by no workflow**. An IC-LoRA is
not a style LoRA: the base weights stay frozen and the adapter adds a *conditioning channel*.
"Motion-track control" is the mechanism for driving animation from authored trajectories rather
than copying them from reference footage — which is the open question from
[`motion-transfer-plan.md`](motion-transfer-plan.md) approached from the other side.

| | |
|---|---|
| Training cost | **None.** Already on disk. |
| Buys you | A conditioning channel for motion; the direct route to driving animation from a diagram |
| Main risk | **The adapters are for LTX 2.3; the registered workflow is LTX 2.5** (`animate_ltx25`, 22B distilled, Modal H100). Adapters are tied to the architecture they were trained against. If they do not load, this route needs the 2.5 equivalent or it collapses into route C. |
| Unresearched | What `motion-track-control` expects as input — point tracks, dense optical flow, a sparse trajectory set — and therefore what a sketch would have to be converted into. None of this has been checked. |
| Effort if it works | One workflow YAML plus a sketch→tracks converter. Days. |

The asymmetry is what recommends it: hours of research against a possible answer to the motion
question at zero training cost, and a negative result (2.3 ≠ 2.5) is still worth knowing early.

## Route B — style LoRA on Klein, trained locally

| | |
|---|---|
| Training cost | Local, free. 16 GB is enough for a 4B at rank 16–32 with an fp8 base, an 8-bit optimiser and gradient checkpointing — *estimate, not yet confirmed against a specific trainer* |
| Buys you | One consistent look across every scene, instead of a style prefix competing with the scene prompt. Plugs in cleanly: the registry already carries a `unet` enum param, so a LoRA becomes another declared param on the keyframe workflow |
| Why it is unusually viable here | **Both the base and the distilled Klein are present.** Training a guidance-distilled checkpoint tends to wreck its few-step behaviour; having the base means training it and applying the result, which is the safe path. Most setups do not have that choice. |
| Main risk | **The 343 keyframes are model output.** Training on them feeds Klein's own artifacts back in and drifts toward them. The 137 reference assets are the real material. A style LoRA wants 20–40 genuinely varied images. |
| Second risk | A base-trained LoRA applied to the distilled model at 4 steps usually works, but is not guaranteed. Verify on both. |
| Effort | Dataset and captioning is the real work; the training run is hours. |

## Route C — video LoRA on Modal

| | |
|---|---|
| Training cost | H100 only, at **$3.95/h**. Tens of dollars per attempt, and the first attempt is rarely right |
| Buys you | Motion style, or a character that holds across frames instead of being swapped in |
| Risk specific to this model | Wan 2.2 I2V A14B is a **two-expert MoE** — `HighNoise` and `LowNoise` load as separate `UnetLoaderGGUF` nodes. Training means choosing the high-noise expert (composition and motion), the low-noise one (detail), or both at double cost. There is no obvious default. |
| Second risk | Video datasets are far harder than image ones: clips, consistent framing, captions describing motion. There are **6** video assets. That is not a dataset. |
| Also | The local card cannot train this at all, so every iteration is a paid remote round-trip with no quick local check |
| Effort | Weeks, realistically |

## Recommendation

1. **Route A, now.** Free, already on disk, and the only one that addresses the motion question.
   Test adapter compatibility against the 2.5 workflow before anything else.
2. **Route B, if visual consistency is the goal.** Genuinely feasible locally, and the
   base/distilled pair makes it safer here than it usually is. Gate it on assembling ~30 **real**
   images — not generated ones.
3. **Route C, last and probably not at all.** The MoE split, the $3.95/h iteration loop and six
   video assets all point the same way. Exhaust the frozen-weight routes first.

And the standing one: **no amount of training will fix brand text in video.** That is VAE
compression, and `locked_regions` is already the right answer.

## If a LoRA is adopted, where it plugs in

Nothing structural has to change. A LoRA is a model file plus a loader node, and the registry
already models exactly that shape:

- the template gains a `LoraLoaderModelOnly` (as `wan_animate2_replace.json` already has);
- the YAML declares `lora` as `type: enum` with `enum_from: {class_type: LoraLoaderModelOnly,
  input: lora_name}`, the same way `unet` is declared today, so the UI offers the installed files
  and `registry validate --online` catches a missing one;
- `strength` becomes a float param;
- a profile picks it per job, and `Scene.keyframe_workflow` can override it per scene.

See [`workflows.md`](workflows.md) for the YAML reference and
[`modal-backend.md`](modal-backend.md) for how a remote backend is declared.
