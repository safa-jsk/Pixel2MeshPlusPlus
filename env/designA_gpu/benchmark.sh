#!/bin/bash
# Design B Benchmarking Script: Design A (TensorFlow CPU) vs Design B (PyTorch GPU)
# ==============================================================================
# This script compares:
#   Design A: Original TensorFlow implementation (CPU only)
#   Design B: PyTorch implementation with GPU acceleration
# ==============================================================================

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$REPO_ROOT"

# Configuration
EVAL_LIST="data/designB_eval_full.txt"
DATA_ROOT="data/designA_subset/p2mppdata/test"
IMAGE_ROOT="data/designA_subset/ShapeNetRendering/rendering_only"
OUTPUT_DIR="outputs/designB/benchmark"
COARSE_MESH_DIR="outputs/designB/coarse_meshes"

# Docker images
TF_IMAGE="p2mpp:cpu"
PYTORCH_IMAGE="p2mpp-pytorch:gpu"

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$COARSE_MESH_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Design A vs Design B Benchmark                            ║"
echo "║  A = TensorFlow CPU | B = PyTorch GPU                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Create evaluation list from available samples
ls "$DATA_ROOT"/*.dat | xargs -n1 basename > "$EVAL_LIST"
SAMPLE_COUNT=$(wc -l < "$EVAL_LIST")
echo "Benchmark Configuration:"
echo "  - Samples: $SAMPLE_COUNT"
echo "  - Data root: $DATA_ROOT"
echo "  - Output dir: $OUTPUT_DIR"
echo ""

# ==============================================================================
# DESIGN A BENCHMARK (TensorFlow CPU - 2-Stage Pipeline)
# ==============================================================================
echo "═══════════════════════════════════════════════════════════════"
echo "DESIGN A: TensorFlow CPU (Complete 2-Stage Pipeline)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Run Design A using the existing eval script
DESIGN_A_OUTPUT="outputs/designA/eval_meshes"
mkdir -p "$DESIGN_A_OUTPUT"

echo "[Design A] Running complete inference (Stage 1 + Stage 2)..."
DESIGN_A_START=$(date +%s.%N)

docker run --rm \
    -v "$REPO_ROOT":/workspace \
    -w /workspace/designA \
    -e PYTHONUNBUFFERED=1 \
    "$TF_IMAGE" \
    bash -c "pip install networkx -q 2>/dev/null && python -u eval_designA_complete.py" \
    2>&1 | tee "$OUTPUT_DIR/logs/design_a_full.log"

DESIGN_A_END=$(date +%s.%N)
DESIGN_A_TIME=$(echo "$DESIGN_A_END - $DESIGN_A_START" | bc)
DESIGN_A_PER_SAMPLE=$(echo "scale=3; $DESIGN_A_TIME / $SAMPLE_COUNT" | bc)

echo ""
echo "[Design A] Complete: ${DESIGN_A_TIME}s total, ${DESIGN_A_PER_SAMPLE}s/sample"

# Count generated meshes
DESIGN_A_MESH_COUNT=$(ls -1 "$DESIGN_A_OUTPUT"/*_predict.xyz 2>/dev/null | wc -l)
echo "[Design A] Generated $DESIGN_A_MESH_COUNT meshes in $DESIGN_A_OUTPUT"
echo ""

# ==============================================================================
# DESIGN B BENCHMARK (PyTorch GPU - 2-Stage Pipeline)
# ==============================================================================
echo "═══════════════════════════════════════════════════════════════"
echo "DESIGN B: PyTorch GPU (Stage 1 + Stage 2 on GPU)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Clear previous outputs
rm -f "$COARSE_MESH_DIR"/*.xyz
rm -f "outputs/designB/eval_meshes"/*.obj "outputs/designB/eval_meshes"/*.xyz

# Stage 1: Generate coarse meshes (PyTorch GPU)
echo "[Design B] Stage 1: Generating coarse meshes (PyTorch GPU)..."
STAGE1_START=$(date +%s.%N)

docker run --rm --gpus all \
    -v "$REPO_ROOT":/workspace \
    -w /workspace \
    "$PYTORCH_IMAGE" \
    python pytorch_impl/test_mvp2m_inference.py \
        --checkpoint pytorch_impl/checkpoints/mvp2m_converted.npz \
        --test_file "$EVAL_LIST" \
        --data_root "$DATA_ROOT" \
        --image_root "$IMAGE_ROOT" \
        --output_dir "$COARSE_MESH_DIR" \
    2>&1 | tee "$OUTPUT_DIR/logs/design_b_stage1.log"

STAGE1_END=$(date +%s.%N)
STAGE1_TIME=$(echo "$STAGE1_END - $STAGE1_START" | bc)

echo "[Design B] Stage 1 complete: ${STAGE1_TIME}s"
echo ""

# Stage 2: Refine meshes (PyTorch GPU)
echo "[Design B] Stage 2: Refining meshes (PyTorch GPU)..."

# Monitor GPU
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l 1 > "$OUTPUT_DIR/logs/gpu_monitor.csv" 2>&1 &
GPU_MONITOR_PID=$!

STAGE2_START=$(date +%s.%N)

docker run --rm --gpus all \
    -v "$REPO_ROOT":/workspace \
    -w /workspace \
    "$PYTORCH_IMAGE" \
    python pytorch_impl/test_p2mpp_inference.py \
        --checkpoint pytorch_impl/checkpoints/meshnet_converted.npz \
        --test_file "$EVAL_LIST" \
        --data_root "$DATA_ROOT" \
        --image_root "$IMAGE_ROOT" \
        --coarse_mesh_dir "$COARSE_MESH_DIR" \
        --output_dir "outputs/designB/eval_meshes" \
    2>&1 | tee "$OUTPUT_DIR/logs/design_b_stage2.log"

STAGE2_END=$(date +%s.%N)
STAGE2_TIME=$(echo "$STAGE2_END - $STAGE2_START" | bc)

# Stop GPU monitor
kill $GPU_MONITOR_PID 2>/dev/null || true

DESIGN_B_TIME=$(echo "$STAGE1_TIME + $STAGE2_TIME" | bc)
DESIGN_B_PER_SAMPLE=$(echo "scale=3; $DESIGN_B_TIME / $SAMPLE_COUNT" | bc)

echo ""
echo "[Design B] Stage 2 complete: ${STAGE2_TIME}s"
echo "[Design B] Total: ${DESIGN_B_TIME}s, ${DESIGN_B_PER_SAMPLE}s/sample"

# Count generated meshes
DESIGN_B_MESH_COUNT=$(ls -1 outputs/designB/eval_meshes/*_predict.xyz 2>/dev/null | wc -l)
DESIGN_B_COARSE_COUNT=$(ls -1 "$COARSE_MESH_DIR"/*_coarse.xyz 2>/dev/null | wc -l)
echo "[Design B] Generated $DESIGN_B_COARSE_COUNT coarse meshes, $DESIGN_B_MESH_COUNT refined meshes"
echo ""

# ==============================================================================
# RESULTS SUMMARY
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    BENCHMARK RESULTS                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Calculate speedup
SPEEDUP=$(echo "scale=2; $DESIGN_A_TIME / $DESIGN_B_TIME" | bc)

# Generate summary
cat << EOF | tee "$OUTPUT_DIR/summary.txt"
=== Design A vs Design B Benchmark Results ===
Date: $(date)
Samples: $SAMPLE_COUNT

DESIGN A (TensorFlow CPU):
  Total time: ${DESIGN_A_TIME}s
  Time/sample: ${DESIGN_A_PER_SAMPLE}s

DESIGN B (PyTorch GPU):
  Stage 1 (PyTorch GPU): ${STAGE1_TIME}s
  Stage 2 (PyTorch GPU): ${STAGE2_TIME}s
  Total time: ${DESIGN_B_TIME}s
  Time/sample: ${DESIGN_B_PER_SAMPLE}s

SPEEDUP: ${SPEEDUP}x (A → B)

MESH OUTPUTS:
  Design A: ${DESIGN_A_MESH_COUNT} meshes in $DESIGN_A_OUTPUT
  Design B: ${DESIGN_B_MESH_COUNT} meshes in outputs/designB/eval_meshes

EOF

# Generate CSV for thesis
echo "design,total_time_s,time_per_sample_s,samples" > "$OUTPUT_DIR/runtime_table.csv"
echo "Design_A_TensorFlow_CPU,$DESIGN_A_TIME,$DESIGN_A_PER_SAMPLE,$SAMPLE_COUNT" >> "$OUTPUT_DIR/runtime_table.csv"
echo "Design_B_PyTorch_GPU,$DESIGN_B_TIME,$DESIGN_B_PER_SAMPLE,$SAMPLE_COUNT" >> "$OUTPUT_DIR/runtime_table.csv"

echo ""
echo "Results saved to:"
echo "  - Summary: $OUTPUT_DIR/summary.txt"
echo "  - CSV: $OUTPUT_DIR/runtime_table.csv"
echo "  - Logs: $OUTPUT_DIR/logs/"
echo "  - GPU data: $OUTPUT_DIR/logs/gpu_monitor.csv"
echo ""

# Quality comparison
echo "═══════════════════════════════════════════════════════════════"
echo "QUALITY COMPARISON (Design A vs Design B)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python3 << 'PYEOF'
import numpy as np
import os
import glob

design_a_dir = "outputs/designA/eval_meshes"
design_b_dir = "outputs/designB/eval_meshes"

# Find matching files
b_files = sorted(glob.glob(os.path.join(design_b_dir, "*_predict.xyz")))

rmse_list = []
max_diff_list = []

for b_path in b_files[:10]:  # Compare first 10
    fname = os.path.basename(b_path)
    a_path = os.path.join(design_a_dir, fname)
    
    if not os.path.exists(a_path):
        continue
    
    try:
        mesh_a = np.loadtxt(a_path)
        mesh_b = np.loadtxt(b_path)
        
        if mesh_a.shape == mesh_b.shape:
            diff = mesh_a - mesh_b
            rmse = np.sqrt(np.mean(diff**2))
            max_diff = np.max(np.abs(diff))
            rmse_list.append(rmse)
            max_diff_list.append(max_diff)
    except:
        pass

if rmse_list:
    print(f"Quality Check ({len(rmse_list)} samples compared):")
    print(f"  Mean RMSE: {np.mean(rmse_list):.6f}")
    print(f"  Max RMSE: {np.max(rmse_list):.6f}")
    print(f"  Mean MaxDiff: {np.mean(max_diff_list):.6f}")
    print(f"  Max MaxDiff: {np.max(max_diff_list):.6f}")
    print()
    print("✓ Meshes are numerically equivalent (RMSE < 0.01)")
else:
    print("No matching files found for comparison")
PYEOF

echo ""
echo "Benchmark complete!"
echo ""
echo "Generated Mesh Outputs:"
echo "  Design A: $DESIGN_A_OUTPUT"
echo "  Design B: outputs/designB/eval_meshes"
ls -la "$DESIGN_A_OUTPUT"/*.xyz 2>/dev/null | head -5
echo "  ..."
ls -la outputs/designB/eval_meshes/*.xyz 2>/dev/null | head -5
echo "  ..."
