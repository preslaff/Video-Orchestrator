#!/bin/bash
# Copy ReActor's model files from the local ComfyUI into the vorch-models Volume
# (they are not published on Hugging Face as one set). Idempotent-ish: `modal volume put`
# fails on existing files, so pass --force to overwrite.
set -euo pipefail
COMFY="${VORCH_COMFY_ROOT:-$HOME/comfy/ComfyUI}"
MODAL="${MODAL_BIN:-$(dirname "$0")/../.venv/bin/modal}"
FORCE="${1:-}"
for rel in insightface/inswapper_128.onnx insightface/models/buffalo_l facerestore_models/GFPGANv1.4.pth facerestore_models/codeformer.pth; do
  src="$COMFY/models/$rel"
  if [ ! -e "$src" ]; then echo "skip (missing locally): $src"; continue; fi
  echo "put $rel"
  "$MODAL" volume put vorch-models "$src" "/$rel" $FORCE
done
echo "done: $("$MODAL" volume ls vorch-models /insightface 2>/dev/null | tail -n +2 | wc -l) entries under /insightface"
