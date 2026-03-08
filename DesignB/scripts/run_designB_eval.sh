#!/bin/bash
# =============================================================================
# Design B — Evaluation Runner (runs INSIDE the Docker container)
# Usage (from /workspace or any dir):
#   bash DesignB/scripts/run_designB_eval.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DESIGN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS="$PROJECT_ROOT/artifacts"

echo "========================================================================"
echo "DESIGN B — PyTorch GPU EVALUATION"
echo "========================================================================"
echo "Project Root : $PROJECT_ROOT"
echo "Script Dir   : $SCRIPT_DIR"
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────────────
echo "=> Checking prerequisites..."

if [ ! -f "$DESIGN_DIR/designB_eval_list.txt" ]; then
    echo "ERROR: designB_eval_list.txt not found at $DESIGN_DIR/designB_eval_list.txt"
    exit 1
fi

if [ ! -f "$ARTIFACTS/checkpoints/torch/mvp2m_converted.npz" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $ARTIFACTS/checkpoints/torch/mvp2m_converted.npz"
    exit 1
fi

if [ ! -f "$ARTIFACTS/checkpoints/torch/meshnet_converted.npz" ]; then
    echo "ERROR: Stage 2 checkpoint not found at $ARTIFACTS/checkpoints/torch/meshnet_converted.npz"
    exit 1
fi

if [ ! -f "$ARTIFACTS/data_templates/iccv_p2mpp.dat" ] && [ ! -f "$PROJECT_ROOT/assets/data_templates/iccv_p2mpp.dat" ]; then
    echo "ERROR: Mesh data template not found (checked assets/data_templates/iccv_p2mpp.dat)"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/data/ShapeNetRendering" ]; then
    echo "ERROR: $PROJECT_ROOT/data/ShapeNetRendering/ not found"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/data/p2mppdata/test" ]; then
    echo "ERROR: $PROJECT_ROOT/data/p2mppdata/test/ not found"
    exit 1
fi

echo "  All prerequisites found."
echo ""

# ── GPU check ─────────────────────────────────────────────────────────────────
echo "=> Checking GPU availability..."
python3 -c "
import torch
print('  PyTorch version: {}'.format(torch.__version__))
print('  CUDA available: {}'.format(torch.cuda.is_available()))
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print('    GPU {}: {}'.format(i, torch.cuda.get_device_name(i)))
else:
    print('  WARNING: No CUDA GPU found — inference will be very slow on CPU.')
"
echo ""

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR="$ARTIFACTS/outputs/designB/eval_meshes"
mkdir -p "$OUTPUT_DIR"

TOTAL_SAMPLES=$(wc -l < "$DESIGN_DIR/designB_eval_list.txt")
MAX_ATTEMPTS=10
attempt=1

echo "========================================================================"
echo "Running infer_with_metrics.py  ($TOTAL_SAMPLES samples, GPU enabled)"
echo "  Eval list  : $DESIGN_DIR/designB_eval_list.txt"
echo "  Output dir : $OUTPUT_DIR"
echo "========================================================================"
echo ""

while true; do
    DONE=$(ls "$OUTPUT_DIR/"*_predict.xyz 2>/dev/null | wc -l)
    echo "[Attempt $attempt] $DONE / $TOTAL_SAMPLES samples already done"

    if [ "$DONE" -ge "$TOTAL_SAMPLES" ]; then
        echo "All $TOTAL_SAMPLES samples complete."
        break
    fi

    python3 "$SCRIPT_DIR/infer_with_metrics.py" \
        --stage1_checkpoint "$ARTIFACTS/checkpoints/torch/mvp2m_converted.npz" \
        --stage2_checkpoint "$ARTIFACTS/checkpoints/torch/meshnet_converted.npz" \
        --mesh_data         "$PROJECT_ROOT/assets/data_templates/iccv_p2mpp.dat" \
        --test_file         "$DESIGN_DIR/designB_eval_list.txt" \
        --image_root        "$PROJECT_ROOT/data/ShapeNetRendering" \
        --gt_root           "$PROJECT_ROOT/data/p2mppdata/test" \
        --output_dir        "$OUTPUT_DIR"
    EXIT_CODE=$?

    DONE=$(ls "$OUTPUT_DIR/"*_predict.xyz 2>/dev/null | wc -l)
    if [ "$DONE" -ge "$TOTAL_SAMPLES" ]; then
        echo "Evaluation complete ($DONE/$TOTAL_SAMPLES samples)."
        break
    fi

    if [ $attempt -ge $MAX_ATTEMPTS ]; then
        echo "ERROR: Evaluation failed after $MAX_ATTEMPTS attempts ($DONE/$TOTAL_SAMPLES done)"
        exit 1
    fi

    echo "WARNING: Crashed at sample $DONE/$TOTAL_SAMPLES (exit code $EXIT_CODE) — resuming..."
    attempt=$((attempt + 1))
    sleep 2
done

echo ""
echo "========================================================================"
echo "EVALUATION COMPLETE"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
BENCHMARK_DIR="$ARTIFACTS/outputs/designB/benchmark"
if [ -f "$BENCHMARK_DIR/summary_stats.txt" ]; then
    echo ""
    cat "$BENCHMARK_DIR/summary_stats.txt"
fi
echo "========================================================================"
