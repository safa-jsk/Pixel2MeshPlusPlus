#!/bin/bash

echo "═══════════════════════════════════════════════════════════"
echo "PYTORCH PIXEL2MESH++ GPU BENCHMARK"
echo "RTX 4070 - Native Ada Lovelace Support"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Verify GPU availability
echo "[1/4] Verifying GPU setup..."
python -c "
import torch
print(f'  PyTorch Version: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA Version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  Compute Capability: {torch.cuda.get_device_capability(0)}')
else:
    print('  ERROR: CUDA not available!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: GPU verification failed!"
    exit 1
fi

echo ""
echo "[2/4] Creating output directories..."
mkdir -p outputs/pytorch_gpu/{run1,run2}
mkdir -p outputs/pytorch_gpu/logs

echo ""
echo "[3/4] Running GPU benchmarks (2 runs, 35 samples each)..."
echo ""

# Configuration
CONFIG="pytorch_impl/cfgs/p2mpp_pytorch.yaml"
CHECKPOINT="results/refine_p2mpp/models/pytorch_model.pth"
TEST_FILE="data/test_list.txt"
DATA_PATH="data/demo/p2mppdata/test"
IMAGE_PATH="data/demo/rendering_only"

# Run 1
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Run 1/2] Starting inference..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

START1=$(date +%s.%N)
python pytorch_impl/test_p2mpp_pytorch.py \
    -c "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --test_file "$TEST_FILE" \
    --data_path "$DATA_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_dir outputs/pytorch_gpu/run1 \
    2>&1 | tee outputs/pytorch_gpu/logs/run1.log

END1=$(date +%s.%N)
TIME1=$(echo "$END1 - $START1" | bc)

echo ""
echo "[Run 1/2] Completed in ${TIME1}s"
echo ""

# Run 2
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Run 2/2] Starting inference..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

START2=$(date +%s.%N)
python pytorch_impl/test_p2mpp_pytorch.py \
    -c "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --test_file "$TEST_FILE" \
    --data_path "$DATA_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_dir outputs/pytorch_gpu/run2 \
    2>&1 | tee outputs/pytorch_gpu/logs/run2.log

END2=$(date +%s.%N)
TIME2=$(echo "$END2 - $START2" | bc)

echo ""
echo "[Run 2/2] Completed in ${TIME2}s"
echo ""

# Calculate statistics
echo "[4/4] Computing benchmark statistics..."

MEAN_TIME=$(echo "scale=3; ($TIME1 + $TIME2) / 2" | bc)
DIFF=$(echo "scale=3; $TIME1 - $TIME2" | bc | tr -d '-')
STD_DEV=$(echo "scale=3; $DIFF / 2" | bc)

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "BENCHMARK RESULTS SUMMARY - PYTORCH GPU (RTX 4070)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Run 1: ${TIME1}s"
echo "Run 2: ${TIME2}s"
echo ""
echo "Mean time: ${MEAN_TIME}s ± ${STD_DEV}s"
echo "Throughput: $(echo "scale=2; 35 / $MEAN_TIME" | bc) samples/sec"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Compare with CPU baseline (if available)
if [ -f "outputs/designA/benchmark/cpu_baseline.txt" ]; then
    CPU_TIME=$(cat outputs/designA/benchmark/cpu_baseline.txt | grep "Mean" | awk '{print $3}' | tr -d 's')
    if [ -n "$CPU_TIME" ]; then
        SPEEDUP=$(echo "scale=2; $CPU_TIME / $MEAN_TIME" | bc)
        echo "CPU Baseline: ${CPU_TIME}s"
        echo "GPU Speedup: ${SPEEDUP}x"
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo ""
    fi
fi

# Save results
echo "{
  \"framework\": \"pytorch\",
  \"device\": \"GPU (RTX 4070)\",
  \"run1_time\": $TIME1,
  \"run2_time\": $TIME2,
  \"mean_time\": $MEAN_TIME,
  \"std_dev\": $STD_DEV,
  \"samples\": 35,
  \"timestamp\": \"$(date -Iseconds)\"
}" > outputs/pytorch_gpu/benchmark_results.json

echo "✓ Benchmark complete!"
echo "  Results saved to: outputs/pytorch_gpu/benchmark_results.json"
echo "  Logs saved to: outputs/pytorch_gpu/logs/"
echo ""
