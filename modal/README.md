# Modal backend — deployment runbook (Phase B)

Design, cost model and risks: [`../docs/modal-backend.md`](../docs/modal-backend.md). Local stays the
default; a workflow runs on Modal only when its registry YAML says `backend: modal-…`.

| file | purpose |
|---|---|
| `vorch_comfy.py` | the Modal app: ComfyUI v0.34.0 (pinned to the local version) as two proxy-authenticated web servers, `comfy_l40s` and `comfy_h100`, models from the `vorch-models` Volume |
| `download_models.py` | fills the Volume from Hugging Face (`wan22_fp16`, `wan22_fp8`) |
| `workflows/` | registry entries for the cloud models: `animate_wan22_fp16` (H100) / `animate_wan22_fp8` (L40S) — the built-in `animate` graph with full-precision loaders; `animate_ltx25` (H100; LTX-2.5 two-stage with audio, sizes /32, 8n+1 frames, 24 fps); `keyframe_flux2_4b` (L40S, Apache 2.0) / `keyframe_flux2_9b` (L40S, **non-commercial**); `animate_wan_animate2` (L40S post-step: Wan-Animate-2 puts the master face/body into the clip) |
| `upload_local_models.sh` | copies ReActor's models (inswapper, GFPGAN, CodeFormer, buffalo_l) from the local ComfyUI into the Volume for the remote `faceswap_video` variant |
| `profiles/` | importable profiles: Wan fp8/fp16 (480p, 720p) with Klein + ReActor local; `cloud_ltx_16x9` (LTX-2.5 with audio); `cloud_social_9x16_all_remote` (Klein 4B + Wan fp8 + Wan-Animate-2, nothing on the local GPU) |
| `backends.example.yaml` | template for `~/vorch/backends.yaml` |

## One-time setup (your account)

```bash
.venv/bin/pip install -e ".[cloud]"          # the modal client
.venv/bin/modal setup                        # browser login
# Dashboard -> Settings -> Proxy Auth Tokens -> New token, then store it once:
cat > ~/vorch/vorch.env <<'EOF'
MODAL_PROXY_KEY=wk-…
MODAL_PROXY_SECRET=ws-…
EOF
chmod 600 ~/vorch/vorch.env        # loaded by `vorch serve` and the CLI at start; no shell exports needed
```

## Deploy

```bash
.venv/bin/modal run modal/download_models.py --only wan22_fp8     # 36 GB, L40S class
.venv/bin/modal run modal/download_models.py --only wan22_fp16    # 69 GB, H100 class
# Phase C sets (see download_models.py): klein4b 16 GB · klein9b 35 GB (non-commercial) ·
# wan_animate2 25 GB · ltx25 39 GB (gated: accept the license on huggingface.co/Lightricks/LTX-2.5,
# then export HF_TOKEN=$(cat ~/.cache/huggingface/token))
.venv/bin/modal run modal/download_models.py --only klein4b,wan_animate2,ltx25
bash modal/upload_local_models.sh                                  # ReActor models from the local install
.venv/bin/modal deploy modal/vorch_comfy.py                       # prints the two URLs
cp modal/backends.example.yaml ~/vorch/backends.yaml              # paste the URLs
cp modal/workflows/* ~/vorch/workflows/
.venv/bin/python -m vorch profiles --import modal/profiles/cloud_social_9x16_fp8.yaml
.venv/bin/python -m vorch profiles --import modal/profiles/cloud_cinematic_16x9.yaml
```

The Volume download runs on Modal's side (no local bandwidth). The first `deploy` builds the image
(ComfyUI + torch, several minutes); later deploys reuse it.

## Verify

```bash
.venv/bin/python -m vorch setup check                 # backends: local + modal-l40s, modal-h100 (not probed)
.venv/bin/python -m vorch setup check --deep          # probes both (wakes the GPUs!), validates the cloud workflows against them
.venv/bin/python -m vorch registry validate animate_wan22_fp8 --online   # against modal-l40s
```

Then run a one-scene job on the imported profile (UI or `vorch run`). Expect: the keyframe and
face swap locally, then `comfy modal-l40s: ready after …s` in the log, the clip downloaded as
`COMFY_OUTPUT/vorch/<job8>/s00_clip_00001_.modal-l40s.mp4`, and `backend`/`gpu` in its manifest.

## Verified (2026-09-05)

Workspace `preslaff`: `modal deploy` 125 s (image build) then 5 s; `wan22_fp8` download ≈ 6 min on
Modal's side; `vorch setup check --profile <cloud profile>` probed `comfy:modal-l40s` and validated
`animate_wan22_fp8` online; job `86c7ff2e` (social 9:16, 81 frames) rendered the Wan phase on the
L40S in 6 min 9 s from face swap to downloaded `s00_clip_00001_.modal-l40s.mp4` (≈ $0.20 of GPU
time + the 10-minute idle tail). Gotcha fixed on the way: a Volume cannot be mounted over
ComfyUI's stock `models/` folder, so the image removes it.

### Phase C (2026-09-05)

| job | path | measured |
|---|---|---|
| a45ff3f8 | Wan 2.2 **fp16 1280×720 × 81**, H100 | 12 min 27 s render (the H100 log's own figure); impasto style preserved at 720p |
| 74dddbb7 | **LTX-2.5** 1280×704 × 97 @ 24 fps, H100 | 90 s first clip (cold), ~28 s per clip after; clips carry 48 kHz AAC; `vorch assemble --spec {audio: true}` kept the track |
| d8ae9721 | all remote on the L40S: Klein 4B bf16 keyframe → Wan fp8 81 f → Wan-Animate-2 | keyframe ~10 s, Wan 6 min, Animate-2 2.2–2.6 min; with the scene prompts auto-filled the character lands in the scene's setting |
| 829ee354 | Wan fp8 49 f → remote ReActor over the clip (L40S) | ReActor step 2 min 10 s incl. transfers (51 s locally) |

Gotchas met: the LTX repo is gated (accept the license on HF, export `HF_TOKEN`); Wan fp16 720p at 20 steps is a
long render even on the H100 — expect 12–13 min per 81-frame clip; Wan-Animate-2 takes clothing from the
reference image and the background from the prompt, so use a reference asset in the intended outfit.

## Cost hygiene

- `scaledown_window` is 10 minutes: a container costs its GPU rate for 10 idle minutes after the
  last request. `vorch setup check --deep` wakes containers; the plain check does not.
- Never commit `min_containers`. For a long session, `modal app update`-style keep-warm is a
  manual, temporary choice.
- `modal app list` / `modal app stop vorch-comfy` stops everything; the Volume persists
  ($0.09/GiB-month beyond the free tier).
