# Modal backend — deployment runbook (Phase B)

Design, cost model and risks: [`../docs/modal-backend.md`](../docs/modal-backend.md). Local stays the
default; a workflow runs on Modal only when its registry YAML says `backend: modal-…`.

| file | purpose |
|---|---|
| `vorch_comfy.py` | the Modal app: ComfyUI v0.34.0 (pinned to the local version) as two proxy-authenticated web servers, `comfy_l40s` and `comfy_h100`, models from the `vorch-models` Volume |
| `download_models.py` | fills the Volume from Hugging Face (`wan22_fp16`, `wan22_fp8`) |
| `workflows/` | registry entries for the cloud Wan variants (`animate_wan22_fp16` → modal-h100, `animate_wan22_fp8` → modal-l40s); same node ids and params as the built-in `animate` |
| `profiles/` | importable profiles that keep Klein + ReActor local and send only the Wan phase to Modal |
| `backends.example.yaml` | template for `~/vorch/backends.yaml` |

## One-time setup (your account)

```bash
.venv/bin/pip install -e ".[cloud]"          # the modal client
.venv/bin/modal setup                        # browser login
# Dashboard -> Settings -> Proxy Auth Tokens -> New token, then in the shell that runs vorch:
export MODAL_PROXY_KEY=wk-…  MODAL_PROXY_SECRET=ws-…
```

## Deploy

```bash
.venv/bin/modal run modal/download_models.py --only wan22_fp8     # 36 GB, L40S class
.venv/bin/modal run modal/download_models.py --only wan22_fp16    # 69 GB, H100 class
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

## Cost hygiene

- `scaledown_window` is 10 minutes: a container costs its GPU rate for 10 idle minutes after the
  last request. `vorch setup check --deep` wakes containers; the plain check does not.
- Never commit `min_containers`. For a long session, `modal app update`-style keep-warm is a
  manual, temporary choice.
- `modal app list` / `modal app stop vorch-comfy` stops everything; the Volume persists
  ($0.09/GiB-month beyond the free tier).
