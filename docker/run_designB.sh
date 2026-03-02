#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Run Design B — GPU (PyTorch 2.5 + CUDA 12.4)
# Usage:  bash docker/run_designB.sh [optional extra docker args...]
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="${P2MPP_IMAGE:-p2mpp:latest}"

echo "══════════════════════════════════════════════════"
echo "  Design B · GPU  |  Image: ${IMAGE_NAME}"
echo "══════════════════════════════════════════════════"

docker run --rm -it \
    --gpus all \
    --name p2mpp-designB \
    -e DESIGN=B \
    -v "${PROJECT_ROOT}":/workspace \
    -w /workspace \
    "${@}" \
    "${IMAGE_NAME}" \
    bash -c '
        echo "[DesignB] Copying pre-built PyTorch Chamfer extension..."
        if ls /tmp/torch_chamfer/*.so 1>/dev/null 2>&1; then
            cp -f /tmp/torch_chamfer/*.so external/torch_chamfer/ 2>/dev/null || true
        fi
        echo "[DesignB] Verifying GPU..."
        python -c "import torch; print(f\"PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}  Devices: {torch.cuda.device_count()}\"); assert torch.cuda.is_available(), \"No GPU found\""
        echo "[DesignB] Ready. Dropping to shell."
        exec bash
    '
