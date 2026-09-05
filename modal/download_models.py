"""Fill the ``vorch-models`` Volume from Hugging Face.

    modal run modal/download_models.py                     # default set: wan22_fp16
    modal run modal/download_models.py --only wan22_fp8    # one set
    modal run modal/download_models.py --only all
    modal run modal/download_models.py --list-sets
    modal run modal/download_models.py --only klein4b,klein9b,ltx25,wan_animate2   # Phase C sets

Files land in the Volume under ComfyUI's ``models/<subdir>/`` layout, so the
registry YAMLs in ``modal/workflows/`` find them by the same names the local
install would use. Existing files with the right size are skipped. Export
``HF_TOKEN`` locally for gated repositories (the Comfy-Org repackaged Wan
weights are public).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

WAN22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
KLEIN4B = "Comfy-Org/flux2-klein-4B"
KLEIN9B_ENC = "Comfy-Org/vae-text-encorder-for-flux-klein-9b"
BFL9B = "black-forest-labs/FLUX.2-klein-9B"          # non-commercial license; accept it on HF first
LTX25 = "Lightricks/LTX-2.5"                         # gated: accept the LTX-2.x Community License on HF
ANIMATE2 = "Comfy-Org/Wan-Animate-2"

# name -> [(repo, path in repo, models/<subdir>)]
SETS: dict[str, list[tuple[str, str, str]]] = {
    "wan22_fp16": [  # 68.8 GB — H100 class
        (WAN22, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors", "diffusion_models"),
        (WAN22, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors", "diffusion_models"),
        (WAN22, "split_files/text_encoders/umt5_xxl_fp16.safetensors", "text_encoders"),
        (WAN22, "split_files/vae/wan_2.1_vae.safetensors", "vae"),
    ],
    "wan22_fp8": [  # 35.6 GB — L40S class (fp8_scaled, still un-GGUF'd)
        (WAN22, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
        (WAN22, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
        (WAN22, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
        (WAN22, "split_files/vae/wan_2.1_vae.safetensors", "vae"),
    ],
    "klein4b": [  # 16.1 GB — FLUX.2 Klein 4B bf16 (Apache 2.0) + Qwen3 4B encoder + FLUX.2 VAE (L40S)
        (KLEIN4B, "split_files/diffusion_models/flux-2-klein-4b.safetensors", "diffusion_models"),
        (KLEIN4B, "split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
        (KLEIN4B, "split_files/vae/flux2-vae.safetensors", "vae"),
    ],
    "klein9b": [  # 34.9 GB — FLUX.2 Klein 9B (NON-COMMERCIAL) + Qwen3 8B encoder (+ VAE if klein4b not fetched)
        (BFL9B, "flux-2-klein-9b.safetensors", "diffusion_models"),
        (KLEIN9B_ENC, "split_files/text_encoders/qwen_3_8b.safetensors", "text_encoders"),
        (KLEIN9B_ENC, "split_files/vae/flux2-vae.safetensors", "vae"),
    ],
    "ltx25": [  # 38.6 GB — LTX-2.5 22B distilled int8 + Gemma-4-12B encoder int8 + video/audio VAE + spatial upscaler (gated)
        (LTX25, "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "diffusion_models"),
        (LTX25, "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "text_encoders"),
        (LTX25, "vae/ltx-2.5-video-vae-bf16.safetensors", "vae"),
        (LTX25, "vae/ltx-2.5-audio-vae-bf16.safetensors", "vae"),
        (LTX25, "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "latent_upscale_models"),
    ],
    "wan_animate2": [  # 24.9 GB — Wan-Animate-2 int8 + distill LoRA + umt5 fp8 + clip_vision_h + Wan2.1 VAE bf16 (L40S)
        (ANIMATE2, "diffusion_models/wan_animate_2_int8_convrot.safetensors", "diffusion_models"),
        (ANIMATE2, "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras"),
        (ANIMATE2, "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
        (ANIMATE2, "clip_vision/clip_vision_h.safetensors", "clip_vision"),
        (ANIMATE2, "vae/Wan2_1_VAE_bf16.safetensors", "vae"),
    ],
    # ReActor's models (inswapper_128.onnx, GFPGANv1.4.pth, insightface buffalo_l) are not on HF
    # in one place: copy them from the local install with modal/upload_local_models.sh.
}

MODELS = "/models"
volume = modal.Volume.from_name("vorch-models", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet]>=0.34")
app = modal.App("vorch-models-download", image=image)


@app.function(
    volumes={MODELS: volume},
    timeout=4 * 3600,
    cpu=4,
    memory=8 * 1024,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def download(items: list[tuple[str, str, str]]) -> list[str]:
    import shutil
    from pathlib import Path

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    done: list[str] = []
    for repo, path, subdir in items:
        dest = Path(MODELS) / subdir / Path(path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        want = next((f.size for f in api.list_repo_tree(repo, path_in_repo=str(Path(path).parent))
                     if f.path == path), None)
        if dest.exists() and (want is None or dest.stat().st_size == want):
            print(f"skip  {dest} (present)")
            done.append(str(dest))
            continue
        print(f"fetch {repo}/{path} -> {dest} ({(want or 0) / 2**30:.1f} GB)")
        got = hf_hub_download(repo, path, local_dir=f"{MODELS}/_hf")
        shutil.move(got, dest)
        done.append(str(dest))
    shutil.rmtree(f"{MODELS}/_hf", ignore_errors=True)
    volume.commit()
    return done


@app.local_entrypoint()
def main(only: str = "wan22_fp16", list_sets: bool = False) -> None:
    if list_sets:
        for name, items in SETS.items():
            print(f"{name}: " + ", ".join(Path(p).name for _, p, _ in items))
        return
    names = list(SETS) if only == "all" else [s.strip() for s in only.split(",")]
    for name in names:
        if name not in SETS:
            raise SystemExit(f"unknown set {name!r}; known: {sorted(SETS)}")
    items = [it for n in names for it in SETS[n]]
    for dest in download.remote(items):
        print("ok   ", dest)

