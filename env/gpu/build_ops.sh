#!/bin/bash
# Build script for Design B custom CUDA ops (RTX4070)
# RTX4070 has CUDA Compute Capability 8.9

set -e

echo "==== Design B: Building CUDA Custom Ops ===="
echo "GPU: RTX4070 (Compute Capability 8.9)"
echo "CUDA: 11.2+"
echo ""

# Get paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXTERNAL_DIR="$REPO_ROOT/external"
OUTPUT_DIR="$REPO_ROOT/outputs/designB/logs"

mkdir -p "$OUTPUT_DIR"

cd "$EXTERNAL_DIR"

echo "[1/2] Verifying CUDA/TensorFlow environment..."
python3 << 'PYEOF'
import tensorflow as tf
import os
import sys

print(f"TensorFlow version: {tf.__version__}")
print(f"Python: {sys.version}")

# Get TF compile/link flags
compile_flags = tf.sysconfig.get_compile_flags()
link_flags = tf.sysconfig.get_link_flags()

print(f"\nTF Compile Flags:")
for flag in compile_flags:
    print(f"  {flag}")

print(f"\nTF Link Flags:")
for flag in link_flags:
    print(f"  {flag}")

# Check CUDA
cuda_path = os.environ.get('CUDA_HOME', '/usr/local/cuda')
if os.path.exists(cuda_path):
    print(f"\nCUDA detected at: {cuda_path}")
    lib_path = os.path.join(cuda_path, 'lib64')
    if os.path.exists(lib_path):
        print(f"  libcudart.so found in: {lib_path}")
else:
    print(f"\nWARNING: CUDA_HOME not found at {cuda_path}")

PYEOF

echo ""
echo "[2/2] Building custom ops..."

# Get paths programmatically
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC="$CUDA_HOME/bin/nvcc"
CUDALIB="$CUDA_HOME/lib64"

if [ ! -f "$NVCC" ]; then
    echo "ERROR: nvcc not found at $NVCC"
    exit 1
fi

echo "Using NVCC: $NVCC"
echo "Using CUDA lib: $CUDALIB"

# Get TF flags
TF_CFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

echo "TF_CFLAGS: $TF_CFLAGS"
echo "TF_LFLAGS: $TF_LFLAGS"

# Clean old builds
echo "Cleaning old builds..."
rm -f tf_approxmatch_so.so tf_nndistance_so.so *.cu.o

# Build tf_approxmatch
echo ""
echo "Building tf_approxmatch_g.cu.o..."
$NVCC -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu \
  $TF_CFLAGS -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

echo "Building tf_approxmatch_so.so..."
g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so \
  -shared -fPIC $TF_CFLAGS -lcudart $TF_LFLAGS -L "$CUDALIB" -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# Build tf_nndistance
echo ""
echo "Building tf_nndistance_g.cu.o..."
$NVCC -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu \
  $TF_CFLAGS -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

echo "Building tf_nndistance_so.so..."
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so \
  -shared -fPIC $TF_CFLAGS -lcudart $TF_LFLAGS -L "$CUDALIB" -O2 -D_GLIBCXX_USE_CXX11_ABI=0

echo ""
echo "==== Build Complete ===="
ls -lh tf_*_so.so

echo ""
echo "Testing op loading..."
python3 << 'TESTEOF'
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    import tensorflow as tf
    mod_nn = tf.load_op_library('./tf_nndistance_so.so')
    print("✓ tf_nndistance_so.so loaded successfully")
except Exception as e:
    print(f"✗ Failed to load tf_nndistance_so.so: {e}")

try:
    import tensorflow as tf
    mod_am = tf.load_op_library('./tf_approxmatch_so.so')
    print("✓ tf_approxmatch_so.so loaded successfully")
except Exception as e:
    print(f"✗ Failed to load tf_approxmatch_so.so: {e}")
TESTEOF

echo ""
echo "All done! .so files are ready for GPU execution."
echo "Logs saved to: $OUTPUT_DIR"
