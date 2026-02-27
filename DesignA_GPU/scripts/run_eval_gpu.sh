#!/bin/bash
# =============================================================================
# Design A GPU Evaluation Script
# Run Design A (TensorFlow) with GPU enabled on designA_eval_1000.txt
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================================================"
echo "DESIGN A - GPU EVALUATION"
echo "========================================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Script Dir: $SCRIPT_DIR"
echo ""

# Check for TensorFlow GPU Docker image
echo "=> Checking for TensorFlow GPU Docker image..."

# TensorFlow 1.15 requires CUDA 10.0, which is incompatible with modern GPUs
# We'll try TensorFlow 2.x with compatibility mode

# Option 1: Try with TensorFlow 1.15 GPU (requires CUDA 10.0)
TF_IMAGE="tensorflow/tensorflow:1.15.5-gpu"

# Option 2: Try with TensorFlow 2.x and compatibility mode
# TF_IMAGE="tensorflow/tensorflow:2.10.0-gpu"

echo "=> Using Docker image: $TF_IMAGE"
echo ""

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/artifacts/outputs/designA_GPU/eval_1000"
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "=> Starting evaluation..."
echo "=> Eval list: $PROJECT_ROOT/data/designA_eval_1000.txt"  # NOTE: place eval list in data/
echo "=> Output dir: $OUTPUT_DIR"
echo ""

cd "$SCRIPT_DIR"

docker run --rm --gpus all \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace/DesignA_GPU/scripts \
    $TF_IMAGE \
    python eval_designA_gpu.py \
        --eval_list ../../data/designA_eval_1000.txt \
        --output_dir ../../artifacts/outputs/designA_GPU/eval_1000 \
        --gpu_id 0 \
    2>&1 | tee "$OUTPUT_DIR/run_log.txt"

echo ""
echo "========================================================================"
echo "EVALUATION COMPLETE"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $OUTPUT_DIR/run_log.txt"
echo "========================================================================"
