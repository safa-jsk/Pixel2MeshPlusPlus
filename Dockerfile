# ===========================================================================
# Pixel2Mesh++ — Unified Docker Image
# Ubuntu 24.04.3 LTS | CUDA 12.6 | cuDNN | TensorFlow 2.16 | PyTorch 2.6
#
# Supports ALL designs from a single image:
#   Design A CPU  — TensorFlow (tf.compat.v1) on CPU
#   Design A GPU  — TensorFlow (tf.compat.v1) on GPU
#   Design B      — PyTorch 2.6 + CUDA 12.6
#   Design C      — (future) FaceScape domain adaptation
#
# Build:
#   docker build -t p2mpp:latest .
#
# Run (per design):
#   bash docker/run_designA_cpu.sh
#   bash docker/run_designA_gpu.sh
#   bash docker/run_designB.sh
# ===========================================================================
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

LABEL maintainer="Pixel2Mesh++ Thesis Project"
LABEL description="Unified image — TF 2.16 + PyTorch 2.6, CUDA 12.6, Ubuntu 24.04"

# ── Environment ──────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_ENABLE_ONEDNN_OPTS=0

# ── System packages ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential make cmake \
    git curl wget ca-certificates vim bc \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── Python base ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --break-system-packages --ignore-installed --upgrade pip

# ── PyTorch 2.6 + CUDA 12.6 ─────────────────────────────────────────────
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# ── TensorFlow 2.16 (GPU, with tf.compat.v1) ────────────────────────────
RUN pip install --no-cache-dir --break-system-packages \
    "tensorflow[and-cuda]>=2.16,<2.18"

# ── Shared Python dependencies ───────────────────────────────────────────
RUN pip install --no-cache-dir --break-system-packages \
    tflearn==0.5.0 \
    "numpy>=1.24,<2" \
    "scipy>=1.11" \
    "matplotlib>=3.7" \
    "scikit-image>=0.21" \
    "Pillow>=10.0" \
    "tqdm>=4.65" \
    "pyyaml>=6.0" \
    "opencv-python-headless>=4.8" \
    "trimesh>=3.20" \
    networkx imageio fvcore iopath

# ── Patch tflearn for TF 2.16+ and Pillow 10+ compatibility ──────────────
# 1) tflearn 0.5.0: is_sequence removed in TF 2.16 → renamed to is_nested
# 2) tflearn 0.5.0: PIL.Image.ANTIALIAS removed in Pillow 10 → use LANCZOS
RUN python3 -c "\
import importlib.util, os;\
base = list(importlib.util.find_spec('tflearn').submodule_search_locations)[0];\
\
path1 = os.path.join(base, 'layers', 'recurrent.py');\
src1 = open(path1).read();\
old1 = 'from tensorflow.python.util.nest import is_sequence';\
new1 = 'try:\n    from tensorflow.python.util.nest import is_sequence\nexcept ImportError:\n    from tensorflow.python.util.nest import is_nested as is_sequence';\
open(path1, 'w').write(src1.replace(old1, new1, 1));\
print('patched: is_sequence');\
\
path2 = os.path.join(base, 'data_utils.py');\
src2 = open(path2).read();\
patched = src2.replace('Image.ANTIALIAS', 'Image.LANCZOS');\
open(path2, 'w').write(patched);\
changed = patched != src2;\
print('patched: ANTIALIAS' if changed else 'no-op: ANTIALIAS already gone');\
"

# ── Optional: PyTorch3D (non-fatal if unavailable) ──────────────────────
RUN pip install --no-cache-dir --break-system-packages pytorch3d 2>/dev/null \
    || echo "[INFO] PyTorch3D not available — custom Chamfer impl will be used"

# ── Compile TF custom CUDA ops (Chamfer + EMD) ──────────────────────────
COPY external/tf_ops/src /tmp/tf_ops_src
RUN apt-get update && apt-get install -y --no-install-recommends libeigen3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /tmp/tf_ops_prebuilt && \
    cd /tmp/tf_ops_src && \
    TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && \
    TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') && \
    CUDA_INC=/usr/local/cuda/include && \
    echo "TF_INC=$TF_INC  TF_LIB=$TF_LIB" && \
    TF_LFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') && \
    echo "TF_LFLAGS=$TF_LFLAGS" && \
    # TF 2.16 pip package no longer bundles Eigen — symlink system Eigen to match old TF include path
    mkdir -p "$TF_INC/third_party" && \
    ln -sfn /usr/include/eigen3 "$TF_INC/third_party/eigen3" && \
    nvcc -std=c++17 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu \
    -I"$TF_INC" -I"$CUDA_INC" \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 --expt-relaxed-constexpr && \
    g++ -std=c++17 tf_nndistance.cpp tf_nndistance_g.cu.o \
    -o /tmp/tf_ops_prebuilt/tf_nndistance_so.so \
    -shared -fPIC -I"$TF_INC" -L/usr/local/cuda/lib64 -lcudart \
    $TF_LFLAGS -O2 && \
    nvcc -std=c++17 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu \
    -I"$TF_INC" -I"$CUDA_INC" \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 --expt-relaxed-constexpr && \
    g++ -std=c++17 tf_approxmatch.cpp tf_approxmatch_g.cu.o \
    -o /tmp/tf_ops_prebuilt/tf_approxmatch_so.so \
    -shared -fPIC -I"$TF_INC" -L/usr/local/cuda/lib64 -lcudart \
    $TF_LFLAGS -O2 && \
    ls -lh /tmp/tf_ops_prebuilt/*.so && \
    echo "✓ TF custom CUDA ops compiled"

# ── Compile PyTorch Chamfer CUDA extension ───────────────────────────────
# Set TORCH_CUDA_ARCH_LIST to avoid GPU auto-detect failure during build
# Covers: Volta(7.0), Turing(7.5), Ampere(8.0/8.6), Ada(8.9), Hopper(9.0)
COPY external/torch_chamfer /tmp/torch_chamfer
RUN cd /tmp/torch_chamfer && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    python setup.py build_ext --inplace && \
    ls -lh *.so 2>/dev/null && \
    echo "✓ PyTorch Chamfer extension compiled" \
    || echo "[WARN] PyTorch Chamfer build failed — pure-Python fallback available"

# ── Set up workspace ────────────────────────────────────────────────────
WORKDIR /workspace

# Pre-built ops will be available when repo is mounted at /workspace:
#   /workspace/external/tf_ops/prebuilt/*.so  (copied by run script)
#   /workspace/external/torch_chamfer/*.so    (copied by run script)

# ── Verify installation ────────────────────────────────────────────────
RUN echo "=== Environment Verification ===" && \
    python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}  GPU: {tf.config.list_physical_devices(\"GPU\")}')" && \
    python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}  Devices: {torch.cuda.device_count()}')" && \
    python -c "import numpy, scipy, PIL, cv2, yaml; print('Core deps OK')" && \
    echo "=== All OK ==="

CMD ["/bin/bash"]
