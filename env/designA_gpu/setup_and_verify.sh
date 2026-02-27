#!/bin/bash
# Design B Quick Setup & Verification for RTX4070

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
echo "Repository root: $REPO_ROOT"
cd "$REPO_ROOT"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  Design B GPU Setup for RTX4070            ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Step 1: Check GPU
echo "[Step 1] Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > outputs/designB/benchmark/system_info.txt 2>&1 || echo "nvidia-smi check (may need docker/env)"
cat outputs/designB/benchmark/system_info.txt 2>/dev/null || echo "  GPU info will be available after env setup"
echo ""

# Step 2: Check CUDA
echo "[Step 2] Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep -i cuda
    echo "  ✓ CUDA toolkit found"
else
    echo "  ⚠ nvcc not in PATH (may be in Docker container)"
fi
echo ""

# Step 3: Verify TensorFlow
echo "[Step 3] Verifying TensorFlow GPU support..."
python3 << 'EOF'
import sys
try:
    import tensorflow as tf
    print("  ✓ TensorFlow {} installed".format(tf.__version__))
    
    # Check GPU (TensorFlow 1.x compatible)
    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    gpus = [x for x in local_devices if x.device_type == 'GPU']
    
    if gpus:
        print("  ✓ {} GPU(s) detected:".format(len(gpus)))
        for gpu in gpus:
            print("      - {}".format(gpu.name))
    else:
        print("  ⚠ No GPU detected - may need proper CUDA/cuDNN setup")
        
except ImportError as e:
    print("  ✗ TensorFlow not installed: {}".format(e))
    sys.exit(1)
EOF
echo ""

# Step 4: Build custom ops
echo "[Step 4] Building custom CUDA ops..."
if [ -f "env/gpu/build_ops.sh" ]; then
    chmod +x env/gpu/build_ops.sh
    bash env/gpu/build_ops.sh 2>&1 | tee outputs/designB/logs/build_log.txt
else
    echo "  ✗ build_ops.sh not found"
    exit 1
fi
echo ""

# Step 5: Verify op loading
echo "[Step 5] Verifying custom op loading..."
cd external
python3 << 'EOF'
import sys
import os

success_count = 0

try:
    import tensorflow as tf
    nn_mod = tf.load_op_library('./tf_nndistance_so.so')
    print("  ✓ NNDistance op loaded successfully")
    success_count += 1
except Exception as e:
    print(f"  ✗ NNDistance op failed: {e}")

try:
    import tensorflow as tf
    am_mod = tf.load_op_library('./tf_approxmatch_so.so')
    print("  ✓ ApproxMatch op loaded successfully")
    success_count += 1
except Exception as e:
    print(f"  ✗ ApproxMatch op failed: {e}")

if success_count == 2:
    print("\n  All ops loaded successfully! Ready for GPU inference.")
else:
    print("\n  Some ops failed to load. Check build logs above.")
    sys.exit(1)
EOF
cd "$REPO_ROOT"
echo ""

echo "╔════════════════════════════════════════════╗"
echo "║  Setup Complete!                           ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Run inference test: python test_p2mpp.py --config cfgs/p2mpp.yaml"
echo "  2. Monitor GPU usage: nvidia-smi -l 1"
echo "  3. Run benchmarks: bash env/gpu/benchmark.sh"
echo ""
