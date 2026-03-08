#!/bin/bash
# =============================================================================
# Design A GPU — Evaluation Runner (runs INSIDE the Docker container)
# Usage (from /workspace or any dir):
#   bash DesignA_GPU/scripts/run_eval_gpu.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DESIGN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS="$PROJECT_ROOT/artifacts"

echo "========================================================================"
echo "DESIGN A - GPU EVALUATION"
echo "========================================================================"
echo "Project Root : $PROJECT_ROOT"
echo "Script Dir   : $SCRIPT_DIR"
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────────────
echo "=> Checking prerequisites..."

if [ ! -f "$DESIGN_DIR/designA_eval_list.txt" ]; then
    echo "ERROR: designA_eval_list.txt not found at $DESIGN_DIR/designA_eval_list.txt"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/data/p2mppdata/test" ]; then
    echo "ERROR: $PROJECT_ROOT/data/p2mppdata/test/ not found"
    exit 1
fi

if [ ! -d "$ARTIFACTS/checkpoints/tf/coarse_mvp2m/models" ]; then
    echo "ERROR: Stage 1 model checkpoint not found at $ARTIFACTS/checkpoints/tf/coarse_mvp2m/models"
    exit 1
fi

if [ ! -d "$ARTIFACTS/checkpoints/tf/refine_p2mpp/models" ]; then
    echo "ERROR: Stage 2 model checkpoint not found at $ARTIFACTS/checkpoints/tf/refine_p2mpp/models"
    exit 1
fi

echo "  All prerequisites found."
echo ""

# ── GPU check ─────────────────────────────────────────────────────────────────
echo "=> Checking GPU availability..."
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('  TF GPUs found: {}'.format(len(gpus)))
for g in gpus:
    print('    {}'.format(g))
if not gpus:
    print('  WARNING: No GPU found — will fall back to CPU.')
"

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR="$ARTIFACTS/outputs/designA_GPU/eval_meshes"
mkdir -p "$OUTPUT_DIR"

TOTAL_SAMPLES=$(wc -l < "$DESIGN_DIR/designA_eval_list.txt")
MAX_ATTEMPTS=10
attempt=1

echo ""
echo "========================================================================"
echo "Running eval_designA_gpu_complete.py  (1000 samples, GPU enabled)"
echo "  Eval list  : $DESIGN_DIR/designA_eval_list.txt"
echo "  Output dir : $OUTPUT_DIR"
echo "========================================================================"
echo ""

cd "$SCRIPT_DIR"

while true; do
    DONE=$(ls "$OUTPUT_DIR/"*_predict.xyz 2>/dev/null | wc -l)
    echo "[Attempt $attempt] $DONE / $TOTAL_SAMPLES samples already done"

    if [ "$DONE" -ge "$TOTAL_SAMPLES" ]; then
        echo "All $TOTAL_SAMPLES samples complete."
        break
    fi

    python3 eval_designA_gpu_complete.py \
        --eval_list  "$DESIGN_DIR/designA_eval_list.txt" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_id 0
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

    echo "WARNING: Crashed at sample $DONE/$TOTAL_SAMPLES (exit $EXIT_CODE) — resuming..."
    attempt=$((attempt + 1))
    sleep 2
done

echo ""
echo "========================================================================"
echo "EVALUATION COMPLETE"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
if [ -f "$ARTIFACTS/outputs/designA_GPU/benchmark/summary_stats.txt" ]; then
    echo ""
    cat "$ARTIFACTS/outputs/designA_GPU/benchmark/summary_stats.txt"
fi
echo "========================================================================"

