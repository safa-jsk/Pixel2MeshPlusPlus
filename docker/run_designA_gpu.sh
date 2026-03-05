#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Run Design A — GPU (TensorFlow tf.compat.v1, CUDA 12.4)
# Usage:  bash docker/run_designA_gpu.sh [optional extra docker args...]
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="${P2MPP_IMAGE:-p2mpp:latest}"

echo "══════════════════════════════════════════════════"
echo "  Design A · GPU  |  Image: ${IMAGE_NAME}"
echo "══════════════════════════════════════════════════"

docker rm -f p2mpp-designA-gpu 2>/dev/null || true

docker run --rm -it \
    --gpus all \
    --name p2mpp-designA-gpu \
    -e DESIGN=A_GPU \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -v "${PROJECT_ROOT}":/workspace \
    -w /workspace \
    "${@}" \
    "${IMAGE_NAME}" \
    bash -c '
        echo "[DesignA-GPU] Copying pre-built TF ops..."
        mkdir -p external/tf_ops/prebuilt
        cp -f /tmp/tf_ops_prebuilt/*.so external/tf_ops/prebuilt/ 2>/dev/null || true
        echo "[DesignA-GPU] Verifying GPU..."
        python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices(\"GPU\"); print(f\"TF GPUs: {len(gpus)}\"); assert len(gpus)>0, \"No GPU found\""
        echo "[DesignA-GPU] Ready. Dropping to shell."
        exec bash
    '
