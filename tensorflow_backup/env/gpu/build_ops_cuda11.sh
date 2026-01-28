#!/bin/bash
# Build custom CUDA ops for RTX 4070 (CUDA 11.0+)
# Design B - GPU Acceleration

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
EXTERNAL_DIR="$REPO_ROOT/external"

echo "╔════════════════════════════════════════════╗"
echo "║  Design B - Building CUDA 11 Custom Ops   ║"
echo "╚════════════════════════════════════════════╝"
echo ""

cd "$EXTERNAL_DIR"

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. CUDA toolkit not installed."
    exit 1
fi

echo "CUDA Compiler: $(nvcc --version | grep release)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -f *.so *.o
echo ""

# Build NNDistance op (updated for CUDA 11)
echo "Building NNDistance op for CUDA 11.0+..."

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Compile CUDA kernel
nvcc tf_nndistance_g.cu \
    -o tf_nndistance_g.cu.o \
    -c \
    -O2 \
    -DGOOGLE_CUDA=1 \
    -x cu -Xcompiler -fPIC \
    -arch=sm_89 \
    --expt-relaxed-constexpr

# Compile C++ wrapper and link
g++ -std=c++14 \
    tf_nndistance.cpp \
    tf_nndistance_g.cu.o \
    -o tf_nndistance_so.so \
    -shared -fPIC \
    ${TF_CFLAGS[@]} \
    ${TF_LFLAGS[@]} \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -O2

echo "✓ NNDistance op built successfully"
echo ""

# Verify
if [ -f "tf_nndistance_so.so" ]; then
    echo "✅ Custom op ready: tf_nndistance_so.so"
    ls -lh tf_nndistance_so.so
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "Build complete! GPU ops ready for RTX 4070."
