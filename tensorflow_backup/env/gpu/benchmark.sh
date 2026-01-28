#!/bin/bash
# Design B Benchmarking Script: Design A (CPU) vs Design B (GPU)

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$REPO_ROOT"

EVAL_LIST="data/designB_eval_full.txt"
CHECKPOINT_MVMESH="results/coarse_mvp2m/models/meshnetmvp2m.ckpt-50"
CHECKPOINT_P2M="refine_p2mpp"  # Original training name from p2mpp.yaml
CHECKPOINT_DIR="results"  # Original training save_path from p2mpp.yaml
CONFIG="cfgs/p2mpp.yaml"
OUTPUT_DIR="outputs/designB/benchmark"

mkdir -p "$OUTPUT_DIR/logs"

echo "╔════════════════════════════════════════════╗"
echo "║  Design B Benchmarking (A vs B)            ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Check if checkpoints exist
if [ ! -f "${CHECKPOINT_DIR}/${CHECKPOINT_P2M}/models/meshnet.ckpt-10.meta" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_DIR}/${CHECKPOINT_P2M}/models/meshnet.ckpt-10"
    echo "Available checkpoints:"
    find results -name "*.meta"
    exit 1
fi

echo "Configuration:"
  echo "  Config: $CONFIG"
  echo "  Checkpoint: $CHECKPOINT_DIR/$CHECKPOINT_P2M"
echo "  Eval list: $EVAL_LIST"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Create evaluation list (use all 35 samples from designA_subset)
SMALL_EVAL_LIST="data/designB_eval_full.txt"
ls data/designA_subset/p2mppdata/test/*.dat | xargs -n1 basename > "$SMALL_EVAL_LIST"
SAMPLE_COUNT=$(wc -l < "$SMALL_EVAL_LIST")
echo "Created eval list with $SAMPLE_COUNT samples: $SMALL_EVAL_LIST"
echo "First 3 samples:"
head -3 "$SMALL_EVAL_LIST"
echo "..."
echo ""

# Function to run benchmark
run_benchmark() {
    local design=$1
    local use_gpu=$2
    local run_num=$3
    local output_file="$OUTPUT_DIR/logs/${design}_run_${run_num}.log"
    
    echo "[${design} Run ${run_num}] Starting ($SAMPLE_COUNT samples)..."
    echo "[${design} Run ${run_num}] Monitor: tail -f $output_file"
    
    # Warmup is only on first run
    if [ $run_num -eq 0 ]; then
        echo "  Warmup pass (not counted)..."
        # Set GPU/CPU config via environment
        if [ "$use_gpu" == "true" ]; then
            export CUDA_VISIBLE_DEVICES=0
        else
            export CUDA_VISIBLE_DEVICES=""
        fi
    fi
    
    # Run inference with timing
    start_time=$(date +%s.%N)
    
    python3 test_p2mpp.py \
        -f "$CONFIG" \
        --restore 1 \
        --test_file_path "$SMALL_EVAL_LIST" \
        --test_data_path "data/designA_subset/p2mppdata/test" \
        --test_image_path "data/designA_subset/ShapeNetRendering/rendering_only" \
        --test_mesh_root "outputs/designA/eval_meshes" \
        2>&1 | tee "$output_file"
    
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    echo "$elapsed" >> "$OUTPUT_DIR/${design}_times.txt"
    echo "[${design} Run ${run_num}] Completed in ${elapsed}s"
    echo ""
}

# Design A Benchmark (CPU - warmup + 2 runs)
echo "═══════════════════════════════════════"
echo "DESIGN A BASELINE (CPU)"
echo "═══════════════════════════════════════"
echo ""
export CUDA_VISIBLE_DEVICES=""  # Disable GPU

# Warmup (use 1 sample only)
echo "[Design A] Warmup pass (1 sample - not timed)..."
WARMUP_LIST="data/designB_warmup.txt"
head -1 "$SMALL_EVAL_LIST" > "$WARMUP_LIST"
echo "Warmup sample: $(cat $WARMUP_LIST)"
python3 test_p2mpp.py -f "$CONFIG" --restore 1 --test_file_path "$WARMUP_LIST" --test_data_path "data/designA_subset/p2mppdata/test" --test_image_path "data/designA_subset/ShapeNetRendering/rendering_only" --test_mesh_root "outputs/designA/eval_meshes" \
    2>&1 | tee "$OUTPUT_DIR/logs/design_a_warmup.log"
echo "✓ Warmup complete. Starting benchmarks..."
echo ""

# Benchmark runs
> "$OUTPUT_DIR/design_a_times.txt"  # Clear file
for run in 1 2; do
    run_benchmark "design_a" "false" "$run"
    sleep 2  # Cool-down
done

# Design B Benchmark (GPU - warmup + 2 runs)
echo "═══════════════════════════════════════"
echo "DESIGN B GPU ACCELERATION (RTX4070)"
echo "═══════════════════════════════════════"
echo ""
export CUDA_VISIBLE_DEVICES=0  # Enable GPU 0

# Warmup + GPU memory profiling
echo "[Design B] Warmup pass..."
nvidia-smi > "$OUTPUT_DIR/gpu_before.txt" 2>&1
python3 test_p2mpp.py --config "$CONFIG" --checkpoint "$CHECKPOINT_P2M" --eval_list "$SMALL_EVAL_LIST" \
    2>&1 | tee "$OUTPUT_DIR/logs/design_b_warmup.log" > /dev/null
nvidia-smi > "$OUTPUT_DIR/gpu_after_warmup.txt" 2>&1

# Benchmark runs
> "$OUTPUT_DIR/design_b_times.txt"  # Clear file
for run in 1 2; do
    # Monitor GPU during run
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi -l 1 > "$OUTPUT_DIR/logs/gpu_monitor_b_run${run}.log" 2>&1 &
        MONITOR_PID=$!
    fi
    
    run_benchmark "design_b" "true" "$run"
    
    # Stop monitoring
    if [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    
    sleep 2  # Cool-down
done

echo ""
echo "═══════════════════════════════════════"
echo "BENCHMARK RESULTS SUMMARY"
echo "═══════════════════════════════════════"
echo ""

# Calculate statistics
python3 << 'EOF'
import os
import statistics

output_dir = "outputs/designB/benchmark"

for design in ["design_a", "design_b"]:
    times_file = os.path.join(output_dir, f"{design}_times.txt")
    
    if os.path.exists(times_file):
        with open(times_file, 'r') as f:
            times = [float(line.strip()) for line in f if line.strip()]
        
        if times:
            mean = statistics.mean(times)
            stdev = statistics.stdev(times) if len(times) > 1 else 0
            print(f"{design.upper()}:")
            print(f"  Runs: {len(times)}")
            print(f"  Mean time: {mean:.2f}s")
            print(f"  Std dev: {stdev:.2f}s")
            print(f"  Min: {min(times):.2f}s, Max: {max(times):.2f}s")
            print()

# Calculate speedup
design_a_file = os.path.join(output_dir, "design_a_times.txt")
design_b_file = os.path.join(output_dir, "design_b_times.txt")

if os.path.exists(design_a_file) and os.path.exists(design_b_file):
    with open(design_a_file, 'r') as f:
        times_a = [float(line.strip()) for line in f if line.strip()]
    with open(design_b_file, 'r') as f:
        times_b = [float(line.strip()) for line in f if line.strip()]
    
    if times_a and times_b:
        mean_a = statistics.mean(times_a)
        mean_b = statistics.mean(times_b)
        speedup = mean_a / mean_b
        print(f"SPEEDUP (A→B):")
        print(f"  Design A (CPU): {mean_a:.2f}s")
        print(f"  Design B (GPU): {mean_b:.2f}s")
        print(f"  Speedup factor: {speedup:.2f}x")

EOF

echo ""
echo "Detailed logs saved to: $OUTPUT_DIR/logs/"
echo "GPU monitoring data: $OUTPUT_DIR/gpu_*.txt"
echo ""
