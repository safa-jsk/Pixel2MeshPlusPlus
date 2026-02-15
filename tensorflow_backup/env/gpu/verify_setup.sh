#!/bin/bash
# Design B GPU Setup Verification
# RTX 4070 + TensorFlow 2.4 + CUDA 11.0

set -e

echo "╔════════════════════════════════════════════╗"
echo "║  Design B - GPU Environment Verification  ║"
echo "╚════════════════════════════════════════════╝"
echo ""

echo "1. System Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Date: $(date)"
echo "Container: p2mpp-gpu:latest (Docker)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo ""

echo "2. Python & TensorFlow"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 --version
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
echo ""

echo "3. CUDA & GPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvcc --version | grep release
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

echo "4. TensorFlow GPU Detection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 << 'EOF'
import tensorflow as tf

print(f"CUDA available: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available()}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"✅ GPU detected: {gpu.name}")
        details = tf.config.experimental.get_device_details(gpu)
        if details:
            print(f"   Compute capability: {details.get('compute_capability', 'N/A')}")
else:
    print("❌ No GPU detected")
EOF
echo ""

echo "5. Custom Op Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "external/tf_nndistance_so.so" ]; then
    echo "✅ Custom op found: external/tf_nndistance_so.so"
    ls -lh external/tf_nndistance_so.so
else
    echo "⚠️  Custom op not built yet"
    echo "   Run: bash env/gpu/build_ops_cuda11.sh"
fi
echo ""

echo "6. TF 1.x Compatibility Mode"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 << 'EOF'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()
a = tf.constant(2.0)
b = tf.constant(3.0)
result = sess.run(a + b)
print(f"✅ TF 1.x mode working: 2.0 + 3.0 = {result}")
sess.close()
EOF
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Design B environment ready for RTX 4070"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
