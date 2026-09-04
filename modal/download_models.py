"""Fill the ``vorch-models`` Volume from Hugging Face.

    modal run modal/download_models.py                     # default set: wan22_fp16
    modal run modal/download_models.py --only wan22_fp8    # one set
    modal run modal/download_models.py --only all
    modal run modal/download_models.py --list-sets

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
    # Phase C candidates (docs/modal-backend.md §5-6): FLUX.2 Klein 9B, LTX-2.5,
    # ReActor (inswapper_128.onnx, GFPGANv1.4.pth), GIMM-VFI. Add sets here.
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

