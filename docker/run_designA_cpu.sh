#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Run Design A — CPU only (TensorFlow tf.compat.v1, no GPU)
# Usage:  bash docker/run_designA_cpu.sh [optional extra docker args...]
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="${P2MPP_IMAGE:-p2mpp:latest}"

echo "══════════════════════════════════════════════════"
echo "  Design A · CPU  |  Image: ${IMAGE_NAME}"
echo "══════════════════════════════════════════════════"

docker rm -f p2mpp-designA-cpu 2>/dev/null || true

docker run --rm -it \
    --name p2mpp-designA-cpu \
    -e CUDA_VISIBLE_DEVICES="" \
    -e DESIGN=A_CPU \
    -v "${PROJECT_ROOT}":/workspace \
    -w /workspace \
    "${@}" \
    "${IMAGE_NAME}" \
    bash -c '
        echo "[DesignA-CPU] Copying pre-built TF ops..."
        mkdir -p external/tf_ops/prebuilt
        cp -f /tmp/tf_ops_prebuilt/*.so external/tf_ops/prebuilt/ 2>/dev/null || true
        echo "[DesignA-CPU] Ready. Dropping to shell."
        exec bash
    '
