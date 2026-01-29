#!/bin/bash
# =============================================================================
# Design B: PyTorch GPU-Accelerated Mesh Generation
# =============================================================================
# This script runs the complete 2-stage mesh generation pipeline:
#   Stage 1: MVP2M coarse mesh generation (TensorFlow CPU)
#   Stage 2: P2MPP mesh refinement (PyTorch GPU)
#
# Usage:
#   ./run_designB_inference.sh
#
# Requirements:
#   - Docker with NVIDIA GPU support
#   - p2mpp:cpu Docker image (TensorFlow)
#   - p2mpp-pytorch:gpu Docker image (PyTorch)
# =============================================================================

set -e  # Exit on error

# Configuration
WORKSPACE="/home/crystal/Documents/Thesis/Pixel2MeshPlusPlus"
TEST_FILE="data/designB_eval_test_full.txt"
DATA_ROOT="data/designA_subset/p2mppdata/test"
IMAGE_ROOT="data/designA_subset/ShapeNetRendering/rendering_only"
COARSE_MESH_DIR="outputs/designB/coarse_meshes"
OUTPUT_DIR="outputs/designB/eval_meshes"

# Docker images
TF_IMAGE="p2mpp:cpu"
PYTORCH_IMAGE="p2mpp-pytorch:gpu"

cd "$WORKSPACE"

echo "============================================================"
echo "Design B: PyTorch GPU-Accelerated Mesh Generation"
echo "============================================================"
echo ""

# Generate test list from available samples
echo "Generating test list from available samples..."
ls "$DATA_ROOT"/*.dat | xargs -n1 basename > "$TEST_FILE"
NUM_SAMPLES=$(wc -l < "$TEST_FILE")
echo "Found $NUM_SAMPLES test samples"
echo ""

# Create output directories
mkdir -p "$COARSE_MESH_DIR"
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Stage 1: Generate Coarse Meshes (TensorFlow CPU)
# =============================================================================
echo "============================================================"
echo "Stage 1: Generating Coarse Meshes (TensorFlow CPU)"
echo "============================================================"
echo ""

STAGE1_START=$(date +%s.%N)

docker run --rm \
    -v "$WORKSPACE":/workspace \
    -w /workspace \
    "$TF_IMAGE" \
    bash -c "pip install networkx -q 2>/dev/null && python designA/generate_coarse_meshes.py \
        --test_file $TEST_FILE \
        --data_root $DATA_ROOT \
        --image_root $IMAGE_ROOT \
        --output_dir $COARSE_MESH_DIR"

STAGE1_END=$(date +%s.%N)
STAGE1_TIME=$(echo "$STAGE1_END - $STAGE1_START" | bc)

echo ""
echo "Stage 1 completed in ${STAGE1_TIME}s"
echo ""

# =============================================================================
# Stage 2: Refine Meshes (PyTorch GPU)
# =============================================================================
echo "============================================================"
echo "Stage 2: Refining Meshes (PyTorch GPU)"
echo "============================================================"
echo ""

STAGE2_START=$(date +%s.%N)

docker run --rm --gpus all \
    -v "$WORKSPACE":/workspace \
    -w /workspace \
    "$PYTORCH_IMAGE" \
    python pytorch_impl/test_p2mpp_inference.py \
        --checkpoint pytorch_impl/checkpoints/meshnet_converted.npz \
        --test_file "$TEST_FILE" \
        --data_root "$DATA_ROOT" \
        --image_root "$IMAGE_ROOT" \
        --coarse_mesh_dir "$COARSE_MESH_DIR" \
        --output_dir "$OUTPUT_DIR"

STAGE2_END=$(date +%s.%N)
STAGE2_TIME=$(echo "$STAGE2_END - $STAGE2_START" | bc)

echo ""
echo "Stage 2 completed in ${STAGE2_TIME}s"
echo ""

# =============================================================================
# Summary
# =============================================================================
TOTAL_TIME=$(echo "$STAGE1_TIME + $STAGE2_TIME" | bc)
TIME_PER_SAMPLE=$(echo "scale=3; $TOTAL_TIME / $NUM_SAMPLES" | bc)

echo "============================================================"
echo "Design B Inference Complete"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Total samples: $NUM_SAMPLES"
echo "  - Stage 1 time (CPU): ${STAGE1_TIME}s"
echo "  - Stage 2 time (GPU): ${STAGE2_TIME}s"
echo "  - Total time: ${TOTAL_TIME}s"
echo "  - Time per sample: ${TIME_PER_SAMPLE}s"
echo ""
echo "Output files:"
echo "  - Coarse meshes: $COARSE_MESH_DIR/"
echo "  - Refined meshes: $OUTPUT_DIR/"
echo ""
echo "Generated $(ls -1 $OUTPUT_DIR/*_predict.obj 2>/dev/null | wc -l) mesh files"
echo "============================================================"
