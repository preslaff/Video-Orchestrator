# Modal cloud backend — design

> **Status (2026-09-05):** designed; Phase A (the code seam) is implemented in the repo, Phases B–E
> are planned. Facts about Modal and the models were researched in September 2026 and carry
> dates; re-verify prices and model availability before Phase B.

## 1. Why and what

The local RTX 5070 Ti (16 GB) fixes what vorch can run: Wan 2.2 only as a Q5 GGUF at 480p and
49–81 frames, FLUX.2 Klein only as the fp8 4B, no room for LTX-2.5. Modal offers serverless
GPUs (L40S 48 GB, A100 80 GB, H100 …) billed per second. Running **some** generation steps there
removes the ceiling for those steps: unquantized Wan 2.2 at 720p, FLUX.2 at full precision, LTX-2.5
with native audio, and whatever comes next.

Decisions:

- The backend is chosen **per workflow**. A registry entry (`workflows/*.yaml`) may declare
  `backend: modal-h100`; profiles keep referencing workflows by name, so one profile can keep
  Klein and ReActor local and send Wan to Modal, and a "cloud" profile can name workflows that
  only exist on Modal.
- **Local stays the default** and keeps working unchanged. Nothing changes for a machine without
  a `backends.yaml`.
- **Gemma stays local** on the CPU (storyboard + evaluation are cheap). "Gemma on Modal" is an
  optional later phase (§7).
- The orchestrator, gates, evaluator, export and assembly stay backend-agnostic: a remote
  ComfyUI is just another URL whose outputs are downloaded into the local layout.

## 2. Architecture

```
┌─ this machine ───────────────────────────────────────────────────────────────────────┐
│  Next.js UI :3000 ─→ FastAPI :8000 (vorch serve) ─→ Orchestrator ─→ steps.run_step     │
│                                                          │                            │
│   ~/vorch/backends.yaml ──→ backends.for_spec(workflow) ─┤                            │
│                                                          ▼                            │
│   ComfyUI :8188 (local, GGUF, 16 GB) ◄── comfy client ──► https://…-comfy-h100.modal.run│
│   ~/comfy/ComfyUI/output/vorch/<job8>/  ◄── /view download (remote outputs land here)  │
│   Gemma llama-server :8080 (CPU)          PostgreSQL          ffmpeg (assembly, export)  │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                                              ▲ Modal-Key / Modal-Secret
┌─ Modal app "vorch-comfy" ────────────────────────────────────┴───────────────────────┐
│  comfy_l40s  (L40S 48 GB)  ── ComfyUI 0.34 + custom nodes ── Volume vorch-models (~150 GB)│
│  comfy_h100  (H100 80 GB)  ── same image ─────────────────────── (Wan fp16, LTX bf16)    │
│  @modal.web_server(8188, requires_proxy_auth=True), max_containers=1, scaledown 10 min  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

Data flow for one remote step: stage inputs with `POST /upload/image` on **that** backend →
`POST /prompt` → poll `GET /history/{id}` → for each output descriptor `GET /view?filename=…`
streamed into the **local** `COMFY_OUTPUT/<subfolder>/` → the rest of the pipeline (provenance
sidecar, evaluation, next step's upload, media URL, assembly, export) sees a local file exactly as
it does today.

**Why the same local layout and not a scratch dir.** `media_url` (`relative_to(COMFY_OUTPUT)`),
the `/media` mount, `assembly.output_dir`, `export._place` (hardlinks), the provenance sidecar,
`last_frame` and `sample_frames` all take the returned path as-is. Downloading into
`COMFY_OUTPUT/<subfolder>/` keeps all of them untouched. A cloud-only machine still needs a
writable `COMFY_OUTPUT`; it is just a directory (`VORCH_COMFY_ROOT=~/vorch/comfy`) and the env
check creates it.

**Naming rule for downloaded files.** ComfyUI numbers outputs by scanning its own output dir, and
a fresh Modal container starts empty, so a regenerated scene can come back with the same name as
the previous one. Downloads are therefore named `<stem>.<backend>[-n]<ext>`
(`s00_clip_00001_.modal-h100.mp4`, then `…modal-h100-2.mp4`); a download never overwrites.
The suffix is unchanged, so every extension-based check keeps working. `orchestrator.output_prefix`
(`vorch/<job8>/sNN_kind`) is backend-agnostic and still defines the layout on both sides.

## 3. The seam (Phase A, implemented)

### Backends — `vorch/backends.py`, `~/vorch/backends.yaml`

```yaml
backends:
  modal-l40s:
    url: https://<workspace>--vorch-comfy-comfy-l40s.modal.run
    auth: modal-proxy              # none | modal-proxy | bearer
    key_env: MODAL_PROXY_KEY       # -> header Modal-Key
    secret_env: MODAL_PROXY_SECRET # -> header Modal-Secret   (bearer: token_env -> Authorization: Bearer)
    request_timeout: 60            # per HTTP call
    poll_interval: 3
    cold_start_grace: 300          # seconds of connection errors tolerated while the container boots
    gpu: L40S                      # labels only
    cost_per_hour: 1.95
  modal-h100:
    url: https://<workspace>--vorch-comfy-comfy-h100.modal.run
    auth: modal-proxy
    key_env: MODAL_PROXY_KEY
    secret_env: MODAL_PROXY_SECRET
    cold_start_grace: 300
    gpu: H100
    cost_per_hour: 3.95
```

Rules: the file holds **no secrets**, only env-var names; a missing env var is a `BackendError`
reported by the env check, not an import-time crash. Entries default to `remote: true`
(`remote: false` is allowed for a second local ComfyUI sharing `COMFY_ROOT`). `local` is always
present (from `VORCH_COMFY_URL`) and cannot be redefined. Names match `^[a-z0-9][a-z0-9_-]{0,31}$`
because they appear in filenames. `config.BACKENDS_FILE` (`VORCH_BACKENDS_FILE`) points elsewhere
if needed. Global client defaults: `VORCH_COMFY_REQUEST_TIMEOUT` (30 s), `VORCH_COMFY_POLL_INTERVAL`
(2 s), `VORCH_COMFY_COLD_START_GRACE` (240 s).

### Client — `vorch/clients/comfy.py`

- One `requests.Session` per backend with its auth headers (never logged or serialised).
- Every call has a timeout and a **retry budget** for connection errors, timeouts and 502/503/504
  with backoff (`_request(backend, …, retry_for=…)`).
- `run(wf, timeout, backend=)`: remote backends first `wait_ready` (poll `/system_stats` up to
  `cold_start_grace`), and the workflow deadline starts **after** the container is ready, so a
  cold start never eats a workflow's `timeout`. History entries whose `status_str` is `error` raise
  `ComfyError` with ComfyUI's messages instead of timing out. On deadline the client sends
  `/interrupt`.
- `fetch_output(descriptor, backend)`: streams `/view` to a `.part` file, renames atomically,
  applies the naming rule. `output_path(descriptor, backend)` returns path arithmetic for local
  backends and `fetch_output` for remote ones.
- `upload_image`, `object_info` (cached per backend), `free`, `system_stats`, `interrupt` all
  take `backend=`; `backend=None` means local, so every existing caller keeps working.

### Routing

- `registry.WorkflowSpec.backend` (default `local`); `steps.run_step` resolves
  `backends.for_spec(spec)` for the run, the output resolution and the provenance manifest
  (`backend`, `gpu`).
- Uploads go to **the consuming workflow's backend**: the master face to the face-swap workflow's
  backend, references to the edit workflow's, the frame to animate to the animate workflow's, each
  post-step's input to its own. The reference upload cache is keyed by backend.
- `vram.reclaim(backend=)` is a no-op for remote backends (dedicated GPU; no `/free`, no sleep,
  no nvidia-smi).

### Environment check

- `backends`: the YAML loads and its env vars exist.
- `comfy:<name>` per backend. **A probe wakes a billed GPU container**, so remote backends are
  probed only when the selected profile uses a workflow on them (the job follows immediately, so
  the warm container is not wasted) or with `--deep`; otherwise the row says "not probed".
- Each workflow is validated against **its own** backend's `/object_info` (node classes and model
  files), and the row shows the backend.
- `comfy_dirs` fails only when a local workflow is in use; otherwise it creates `COMFY_OUTPUT`
  (downloads land there) and merely warns about a missing input dir.
- `/api/setup/config` lists the backends (no secrets); the Workflows page shows a backend column.

### Dry run without Modal (verified 2026-09-05)

The whole seam can be exercised against the local ComfyUI by declaring it a second, "remote"
backend — no cloud account needed:

```yaml
# a scratch backends file (VORCH_BACKENDS_FILE=…/backends.yaml)
backends:
  loopback:
    url: http://localhost:8188
    remote: true          # readiness wait, /view download, no local VRAM reclaim
    cold_start_grace: 10
    gpu: RTX 5070 Ti (loopback)
```

With a copy of `workflows/keyframe.yaml` as `keyframe_remote` (`backend: loopback`) in a
scratch `VORCH_USER_WORKFLOW_DIR`: `vorch setup check` lists `comfy:loopback` as *not probed*
and the workflow row as `@loopback (offline check)`; `--deep` probes it and validates the
workflow against its `/object_info`; `vorch keyframe … --workflow keyframe_remote` waits for
readiness, submits, polls, and downloads the result as
`COMFY_OUTPUT/Flux2-Klein_00057_.loopback.png` (8 s end to end at 512², 8 steps) with a manifest
recording `backend: loopback` and the GPU label. Nothing in the local default path changes.

## 4. Modal deployment (Phase B)

Install once: `pip install modal && modal setup` (browser login). Create a proxy auth token in
the Modal dashboard and export `MODAL_PROXY_KEY` / `MODAL_PROXY_SECRET` locally. Store the
Hugging Face token as a Modal secret: `modal secret create huggingface HF_TOKEN=…`.

`modal/vorch_comfy.py` (to be added to the repo):

```python
import subprocess, time, urllib.request
import modal

COMFY_VERSION = "0.34.0"          # keep equal to the local ComfyUI (see risk "version drift")
NODES = ["ComfyUI-GGUF", "ComfyUI-VideoHelperSuite", "comfyui-reactor-node", "ComfyUI-GIMM-VFI",
         "ComfyUI-KJNodes", "ComfyUI-LTXVideo"]

image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("comfy-cli", "huggingface_hub[hf_transfer]")
    .run_commands(f"comfy --skip-prompt install --nvidia --version {COMFY_VERSION}",
                  "comfy node install " + " ".join(NODES))
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
models = modal.Volume.from_name("vorch-models", create_if_missing=True)
hf = modal.Secret.from_name("huggingface")
app = modal.App("vorch-comfy")

def _serve(port=8188):
    subprocess.Popen(f"comfy launch -- --listen 0.0.0.0 --port {port} --max-upload-size 1024", shell=True)
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=1); return
        except Exception:
            time.sleep(1)
    raise RuntimeError("ComfyUI did not come up")

COMMON = dict(image=image, timeout=24 * 3600, scaledown_window=600, max_containers=1,
              volumes={"/root/comfy/ComfyUI/models": models}, secrets=[hf])

@app.function(gpu="L40S", memory=65536, **COMMON)
@modal.concurrent(max_inputs=8)
@modal.web_server(8188, startup_timeout=300, requires_proxy_auth=True)
def comfy_l40s(): _serve()

@app.function(gpu="H100", memory=131072, **COMMON)
@modal.concurrent(max_inputs=8)
@modal.web_server(8188, startup_timeout=300, requires_proxy_auth=True)
def comfy_h100(): _serve()
```

Two functions give two URLs and two backend names: `modal-l40s` for Klein 4B/9B, ReActor, GIMM,
LTX-2.5 fp8; `modal-h100` for Wan 2.2 fp16 at 720p and LTX-2.5 bf16. `max_containers=1` with
`@modal.concurrent(8)` lets one ComfyUI serve one job's uploads and polls; ComfyUI queues prompts
itself. The Volume is mounted directly at `models/` so ReActor finds
`models/insightface/inswapper_128.onnx` and `models/facerestore_models/GFPGANv1.4.pth` without
symlinks. `requires_proxy_auth=True` is mandatory: without it the endpoint is a public GPU.

`modal/download_models.py` (`modal run modal/download_models.py --only wan22`): a list of
`(repo, file, models/<subdir>)` downloaded with `hf_hub_download` into the Volume, then
`models.commit()`. Set: Wan 2.2 I2V A14B fp16 high + low (28.6 GB each), `umt5_xxl` encoder, Wan
2.1 VAE; FLUX.2 Klein 4B bf16 (13 GB) + Qwen3 text encoder + FLUX.2 VAE (+ Klein 9B, 29 GB,
non-commercial); LTX-2.5 dev + its Gemma-4-12B text encoder; `inswapper_128.onnx`, `GFPGANv1.4.pth`;
GIMM-VFI weights. About 150 GB, within the 1 TiB free Volume tier.

Cold starts, in order of effort: keep `scaledown_window=600` (a running job keeps the container
warm through its polling); `min_containers=1` only during a work session and never in the
committed file (about $2–4 per idle hour); memory snapshots (`enable_memory_snapshot=True` with
ComfyUI started in `@modal.enter(snap=True)`) for the import phase; experimental GPU snapshots
(community: 53–77 s down to 8–10 s) as a Phase C experiment. Loading 57 GB of Wan fp16 weights
from the Volume at ~2.5 GB/s is 25–40 s regardless.

Deploy with `modal deploy modal/vorch_comfy.py`; the URLs go into `~/vorch/backends.yaml`.

## 5. Cloud workflows and profiles (Phase C)

New registry entries, exported as API JSON from ComfyUI's own templates on the Modal instance:

| name | kind | backend | notes |
|---|---|---|---|
| `animate_wan22_fp16` | animate | modal-h100 | same params as `animate`; `UNETLoader` fp16 instead of the GGUF loader; 832×480, 49/81 frames |
| `animate_wan22_fp16_720p` | animate | modal-h100 | same template; profile sizes 1280×720 / 720×1280 / 960×960, 81 frames |
| `keyframe_flux2_9b` | keyframe | modal-l40s | Klein 9B; tag `license:non-commercial` |
| `animate_ltx25` | animate | modal-h100 (bf16) or modal-l40s (fp8) | `length` constraint `8n+1`, sizes `multiple_of_32`; outputs mp4 **with audio** |
| `faceswap_video_reactor` | postprocess | modal-l40s | ReActor over a whole clip (VHS load → ReActor → combine); needs the master face staged on that backend |
| `animate_wan_animate2` | postprocess | modal-h100 | Wan-Animate-2 character replacement: model-level identity, an alternative to frame-wise ReActor |

Code changes for Phase C: `registry.CONSTRAINTS["8n+1"]` and `["multiple_of_32"]`; profile and
`GenParams` validation read the length/size constraints from the chosen workflow instead of
assuming `4n+1` / `multiple_of_16`; `formats.WAN_PRESETS_720` (1280×720, 720×1280, 960×960,
864×1080, 960×720, 720×960) as UI defaults; `formats.vram_warnings` skipped when the animate
workflow is remote; `postprocess.AUTO_FILL` gains `source_image` (the master face staged on the
step's backend) for video face swap; `media.assemble` gains an `audio` option for LTX clips
(today it strips audio).

Built-in profiles: `cloud_cinematic_16x9_720p` (Klein local 1280×720 → `animate_wan22_fp16_720p`
81 frames @ 16 fps, ReActor local), `cloud_social_9x16_720p`, `cloud_ltx_16x9` (`animate_ltx25`,
1280×704, 97 frames, 24 fps).

## 6. Model comparison (ComfyUI-native, state of Sept 2026)

| Model | Weights / VRAM | GPU class | Strengths | License | Unlocks vs local |
|---|---|---|---|---|---|
| Wan 2.2 A14B I2V fp16 | 28.6 GB × 2 + umt5 11 GB; 480p ≈ 40 GB, 720p 60–80 GB | A100-80 / H100 | same family as today at full precision, 720p, 81+ frames; LightX2V 4-step LoRAs | Apache 2.0 | quality + resolution + length |
| Wan 2.2 5B TI2V | 10 GB fp16, 1280×704 @ 24 fps, 121 frames | L4 / L40S | cheap, fast, higher fps | Apache 2.0 | a fast draft tier |
| Wan-Animate-2 (Aug 2026) | Wan-based | H100 | character replacement with identity across a clip | Apache 2.0 | video-level identity instead of frame-wise ReActor |
| FLUX.2 Klein 4B bf16 | 13 GB | L4 / L40S | today's keyframe model at full precision | Apache 2.0 | small quality gain; cheap |
| FLUX.2 Klein 9B | 29 GB fp16 | L40S | better detail and text | **non-commercial** | not for client work |
| FLUX.2-dev 32B | 64 GB bf16 / 32 GB fp8 | H100 | top FLUX quality | non-commercial | same caveat |
| LTX-2.5 (Aug 2026, 22B) | bf16 ≥ 40 GB, fp8 ≈ 20 GB | L40S (fp8) / H100 | native synchronized audio, multishot, up to 4K/50 fps, fast | Community: free < $10M revenue | audio + long shots; different look |
| HunyuanVideo 1.5 (8.3B) | 14–24 GB | L40S | strong motion, 720p + SR | Tencent community, territory clauses | check the license first |
| SkyReels-V3 (Jan 2026) | Wan-based, fp8 | L40S / H100 | audio-to-video, reference-to-video | open incl. commercial | talking-head style material |
| Wan 2.5 / 2.6 | API only | — | — | no open weights | not usable here |

Commercial-safe defaults for the cloud profiles: Wan 2.2 (Apache) and Klein 4B (Apache); LTX-2.5
under its revenue threshold; Klein 9B and dev only for non-commercial use.

## 7. Gemma on Modal (Phase D, optional)

A second Modal app serving llama.cpp (with `--mmproj`) or vLLM with the Gemma 4 26B behind
`@modal.web_server` on an L40S: evaluation drops from tens of seconds on the CPU to a few. The
`gemma` client would gain the same backend/auth/timeout treatment as the ComfyUI client. Worth it
only when evaluation time dominates (many variants, video evaluation); otherwise the local CPU is
free and simpler.

## 8. Cost model

Assumptions: Modal prices of Sept 2026, cold start ≈ 90 s billed, `scaledown_window` 600 s, and
**per-clip timings that must be benchmarked** (no clean public A100/H100 numbers for Wan fp16).

| Scenario | GPU time | compute | cold start | idle tail | total |
|---|---|---|---|---|---|
| 5 scenes, Wan 2.2 fp16 480p × 49, A100-80 ($2.50/h), 3 min per clip | 15 min | $0.63 | $0.06 | $0.42 | **≈ $1.10** |
| 5 scenes, Wan fp16 720p × 81, H100 ($3.95/h), 6 min per clip | 30 min | $1.98 | $0.10 | $0.66 | **≈ $2.75** |
| 10 Klein 4B keyframes on L4 ($0.80/h), ~6 s each | 1 min | $0.01 | $0.02 | $0.13 | ≈ $0.16, dominated by the idle tail: keep Klein local |
| one regenerated scene on H100 | 6 min | $0.40 | $0.10 | $0.66 | ≈ $1.15 (the tail dominates; use `scaledown_window=120` when regenerating a lot) |
| `min_containers=1` H100 for a 4 h session | 240 min | $15.80 | — | — | only during active work |
| Volume with 150 GB of weights | — | — | — | — | $0 within the 1 TiB free tier ($13.5/month at list) |

The Starter plan's $30 monthly credit covers roughly 25 five-scene 480p jobs or 10 at 720p.
Media transfer is 5–20 MB per clip; Modal does not itemise egress today (verify).

## 9. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Cold start counted against the workflow timeout | spurious timeouts | `wait_ready` before `/prompt`; the deadline starts after ready; `cold_start_grace` per backend |
| Modal's 150 s request cap (303 redirect after) | long blocking calls | submit-and-poll only; `/view` streams in seconds; the client follows redirects |
| Endpoint public without proxy auth | anyone can run your GPU | `requires_proxy_auth=True` required in the deploy; headers never logged |
| The env check wakes a GPU container | $0.33–0.66 per idle tail | probe remote backends only when the profile uses them or with `--deep` |
| Model licenses | Klein 9B / dev non-commercial; LTX-2.5 revenue threshold; Hunyuan territory clauses | license tags on workflows; commercial-safe defaults |
| Version drift between local and cloud ComfyUI | template rejected at `/prompt` | pin `COMFY_VERSION` equal to local; env check reports both versions; workflows validated against their own backend |
| Output name collisions / counter reset | a download overwrites a previous clip | `<stem>.<backend>[-n]<ext>`, never overwrite |
| Mid-job failure (container died, network) | one scene fails | retry budgets per request; the job fails as today; `regenerate` reruns one scene |
| Cost runaway | | `max_containers=1`, `scaledown_window`, a Modal budget alert, `min_containers` never committed |
| Provenance | which GPU made a clip? | manifests record `backend` and `gpu` |
| Tests cannot reach Modal | seam regressions | a fake remote backend in the test suite; a live checklist (§11) |
| Cloud-only machine without a ComfyUI install | env check fails | `comfy_dirs` is local-only; `VORCH_COMFY_ROOT=~/vorch/comfy` |
| Remote upload limit | video post-steps fail | `--max-upload-size 1024` in the deploy |
| Container RAM while swapping Wan fp16 models | OOM kill | `memory=131072` on the H100 function |

## 10. Effort (one experienced developer)

| Phase | Scope | Days |
|---|---|---|
| A — seam (done) | backends, client, routing, VRAM skip, env check, tests, docs | 5 |
| B — Modal deployment | image, custom nodes, model downloader + Volume, proxy auth, `backends.yaml`, first cloud workflow (Wan fp16 480p) + profile, live checklist, benchmarks | 5 |
| C — 720p, FLUX.2 9B, LTX-2.5, video identity | new constraints, workflow-driven validation, size tier, audio in assembly, `source_image` autofill, templates + 3 profiles | 6.5 |
| D — Gemma on Modal (optional) | model server, client backend, env check | 3 |
| E — UI polish | backend badges, grouped env page, profile chips with cost hints, "on modal (H100)" in the step line | 2 |
| **Total** | | **21.5 (18.5 without D)** |

## 11. Live checklist (Phase B)

1. `pip install modal && modal setup`; create a proxy token; `modal secret create huggingface HF_TOKEN=…`.
2. `modal run modal/download_models.py --only wan22` (watch the Volume size), then `modal deploy modal/vorch_comfy.py`.
3. Write `~/vorch/backends.yaml` with the two URLs; export `MODAL_PROXY_KEY` / `MODAL_PROXY_SECRET`.
4. `vorch setup check --deep`: `backends` ok, `comfy:modal-h100` ok with the reported GPU, every `animate_wan22_fp16*` workflow ok against that backend.
5. Run a 1-scene job on `cloud_cinematic_16x9` (480p); confirm the download name
   `s00_clip_00001_.modal-h100.mp4`, the manifest's `backend`/`gpu`, the media URL, export and
   assembly.
6. Regenerate the scene: a same-named remote file gets the `-1` suffix (`<stem>.<backend>-1<ext>`), nothing is overwritten.
7. Benchmark per-clip time at 480p and 720p; update §8; set `scaledown_window` for your rhythm.
8. Read the Modal usage page after the day and compare with §8.
