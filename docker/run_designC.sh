#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Run Design C — GPU (FaceScape domain adaptation, future)
# Usage:  bash docker/run_designC.sh [optional extra docker args...]
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="${P2MPP_IMAGE:-p2mpp:latest}"

echo "══════════════════════════════════════════════════"
echo "  Design C · GPU  |  Image: ${IMAGE_NAME}"
echo "══════════════════════════════════════════════════"

docker run --rm -it \
    --gpus all \
    --name p2mpp-designC \
    -e DESIGN=C \
    -v "${PROJECT_ROOT}":/workspace \
    -w /workspace \
    "${@}" \
    "${IMAGE_NAME}" \
    bash -c '
        echo "[DesignC] Copying pre-built extensions..."
        mkdir -p external/tf_ops/prebuilt
        cp -f /tmp/tf_ops_prebuilt/*.so external/tf_ops/prebuilt/ 2>/dev/null || true
        if ls /tmp/torch_chamfer/*.so 1>/dev/null 2>&1; then
            cp -f /tmp/torch_chamfer/*.so external/torch_chamfer/ 2>/dev/null || true
        fi
        echo "[DesignC] Ready (stub). Dropping to shell."
        exec bash
    '
