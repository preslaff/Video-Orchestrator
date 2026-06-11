# Modal-Hosted Backend — Design Note (optional, future)

> **Status:** Exploratory design note. **Not implemented.** Captures *how* the orchestrator could
> run its heavy generation on [Modal](https://modal.com) (serverless cloud GPUs) instead of — or
> alongside — the local RTX 5070 Ti, so the option is well-understood before any code is written.
> Further research into Modal's current capabilities is pending; treat the Modal-side specifics here
> as a sketch to validate, not settled fact.

## Why this is even feasible

The whole orchestrator is a **deterministic Python core that calls ComfyUI and Gemma as HTTP
tools**. Nothing in the control flow assumes those services are local — it assumes they answer at a
URL. Both URLs are already environment-overridable (`vorch/config.py`):

```python
GEMMA_URL = os.environ.get("VORCH_GEMMA_URL", "http://localhost:8080")
COMFY_URL = os.environ.get("VORCH_COMFY_URL", "http://localhost:8188")
```

Point those at a Modal web endpoint and most of the client works untouched:

| Client call | Mechanism | Works against remote? |
|---|---|---|
| `comfy.run()` | `POST /prompt`, poll `GET /history/{id}` | ✅ pure HTTP |
| `comfy.upload_image()` | `POST /upload/image` | ✅ pure HTTP |
| `comfy.free()` | `POST /free` | ✅ pure HTTP (and largely moot on Modal — see below) |
| `gemma.complete_text/vision()` | OpenAI-compatible `/v1/chat/completions` | ✅ pure HTTP |

## The one seam that breaks on remote

`comfy.output_path()` resolves a result by reading the **local filesystem**:

```python
def output_path(descriptor: dict) -> Path:
    sub = descriptor.get("subfolder", "")
    return config.COMFY_OUTPUT / sub / descriptor["filename"]   # ← assumes ComfyUI is on THIS machine
```

On Modal the output file lives on the **remote container's ephemeral disk**, not the orchestrator's.
Every downstream step that opens that path locally then fails:

- `pipeline.run_scene` — staging the keyframe/swapped frame into the next step
- the planned `media.last_frame` — ffmpeg over the local video file
- `evaluator.evaluate_image` — reading the image for Gemma vision

### Fix: fetch over HTTP instead of reading the path
ComfyUI exposes outputs at `GET /view?filename=…&subfolder=…&type=output`. The seam is to make
output resolution **mode-aware**:

- **Local** `COMFY_URL` → return the local path (current behaviour, zero overhead).
- **Remote** `COMFY_URL` → `GET /view`, download to a local scratch dir (e.g. `runs/<job>/`),
  return that local path.

Everything downstream then operates on a real local file exactly as today — the rest of the
pipeline, registry, evaluator, and continuity work stay **backend-agnostic and unchanged**. This is
the *only* code change required to make remote ComfyUI usable.

## What changes conceptually

| Concern | Local (RTX 5070 Ti, 16GB) | Modal |
|---|---|---|
| **16GB VRAM ceiling** | The binding constraint driving the entire architecture | **Dissolves** — choose A100 / H100 / L40S; Wan dual-model (~22GB) fits natively, no RAM offload |
| `comfy.free()` + phase-by-stage batching (NOTES §5) | Essential — one shared GPU | Largely **moot** — each Modal function gets its own GPU; per-step isolation is free |
| Deployment | ComfyUI + llama-server already running locally | ComfyUI-on-Modal: ASGI/web-server endpoint, model weights in a `modal.Volume`, cold-start latency on first hit |
| Cost / latency | Sunk hardware cost, fast local disk I/O | Pay-per-GPU-second + image/video upload/download round-trips |
| Failure modes | OOM, model swap churn | Cold starts, container lifecycle, network transfer of large media |

## Recommended shape: per-step backend, not all-or-nothing

The cleanest design treats backend as a **per-step choice**, selected by which URL each client points
at. For this hardware the natural split:

- **Gemma stays local.** It's already pinned to CPU under VRAM Option B, so it isn't competing for
  the GPU. Moving it to Modal buys little and adds a network round-trip to *every* evaluation.
- **Heavy ComfyUI steps (Wan especially) optionally go to Modal** when you want headroom the local
  card can't give: Wan native dual-model, higher resolutions, or parallel scenes.

With the `/view` fetch seam in place, the **same pipeline runs local or cloud by flipping one env
var** — `VORCH_COMFY_URL`. No changes to the workflow registry, evaluator, or sequence runner.

## ComfyUI-on-Modal — minimal sketch (to validate, not final)

```python
# modal_comfy.py  (illustrative — verify against current Modal docs before use)
import modal

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI /root/ComfyUI",
        "pip install -r /root/ComfyUI/requirements.txt",
        # + install custom nodes: ReActor, Wan, GGUF loaders, etc.
    )
)
# Model weights persist across cold starts in a Volume (don't bake 20GB into the image).
weights = modal.Volume.from_name("comfy-weights", create_if_missing=True)

app = modal.App("vorch-comfy")

@app.function(image=image, gpu="A100", volumes={"/root/ComfyUI/models": weights}, timeout=3600)
@modal.web_server(8188, startup_timeout=120)
def comfyui():
    import subprocess
    subprocess.Popen(
        "python /root/ComfyUI/main.py --listen 0.0.0.0 --port 8188",
        shell=True,
    )
```

Then locally:

```bash
export VORCH_COMFY_URL="https://<your-workspace>--vorch-comfy-comfyui.modal.run"
python -m vorch run-scene --keyframe-prompt "..." --animation-prompt "..." --master face.jpeg
```

**Open questions for the research pass:**
- Exact current Modal API for long-lived web servers vs. `@modal.asgi_app` / `@modal.web_server`,
  and how cold-start / scale-down interacts with a multi-step job that polls `/history`.
- Best way to keep large model weights warm (Volume vs. baked image vs. `modal.Cls` with
  container reuse) and the cold-start cost for the Wan/Klein/ReActor model set.
- Whether to keep ComfyUI's input/output on the container disk (current assumption) or back it with
  a Volume so `/view` round-trips can be skipped.
- Cost envelope per scene (GPU-seconds × media transfer) vs. local generation.

## Decision

Defer. The local path is the Phase-2 focus. This stays an **optional cloud backend** we can enable
later for a fully cloud-based setup; the only prerequisite is the mode-aware `output_path` fetch
seam, which is small and isolated. Revisit after the local pipeline (state persistence, multi-scene
orchestration, UI) is solid.
