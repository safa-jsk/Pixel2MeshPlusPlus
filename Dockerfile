# ===========================================================================
# Pixel2Mesh++ — Unified Docker Image
# Ubuntu 24.04.3 LTS | CUDA 12.4 | cuDNN | TensorFlow 2.16 | PyTorch 2.5
#
# Supports ALL designs from a single image:
#   Design A CPU  — TensorFlow (tf.compat.v1) on CPU
#   Design A GPU  — TensorFlow (tf.compat.v1) on GPU
#   Design B      — PyTorch 2.5 + CUDA 12.4
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
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04

LABEL maintainer="Pixel2Mesh++ Thesis Project"
LABEL description="Unified image — TF 2.16 + PyTorch 2.5, CUDA 12.4, Ubuntu 24.04"

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
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── Python base ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --break-system-packages --upgrade pip

# ── PyTorch 2.5 + CUDA 12.4 ─────────────────────────────────────────────
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

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

# ── Optional: PyTorch3D (pre-built wheel, non-fatal if unavailable) ─────
RUN pip install --no-cache-dir --break-system-packages \
    --no-index pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py312_cu124_pyt250/download.html 2>/dev/null \
    || echo "[INFO] PyTorch3D wheel not available — custom Chamfer impl will be used"

# ── Compile TF custom CUDA ops (Chamfer + EMD) ──────────────────────────
COPY external/tf_ops/src /tmp/tf_ops_src
RUN mkdir -p /tmp/tf_ops_prebuilt && \
    cd /tmp/tf_ops_src && \
    TF_CFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') && \
    TF_LFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') && \
    # nn_distance (Chamfer)
    nvcc -std=c++17 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu \
    $TF_CFLAGS -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 --expt-relaxed-constexpr && \
    g++ -std=c++17 tf_nndistance.cpp tf_nndistance_g.cu.o -o /tmp/tf_ops_prebuilt/tf_nndistance_so.so \
    -shared -fPIC $TF_CFLAGS -L/usr/local/cuda/lib64 -lcudart $TF_LFLAGS -O2 && \
    # approxmatch (EMD)
    nvcc -std=c++17 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu \
    $TF_CFLAGS -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 --expt-relaxed-constexpr && \
    g++ -std=c++17 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o /tmp/tf_ops_prebuilt/tf_approxmatch_so.so \
    -shared -fPIC $TF_CFLAGS -L/usr/local/cuda/lib64 -lcudart $TF_LFLAGS -O2 && \
    ls -lh /tmp/tf_ops_prebuilt/*.so && \
    echo "✓ TF custom CUDA ops compiled"

# ── Compile PyTorch Chamfer CUDA extension ───────────────────────────────
COPY external/torch_chamfer /tmp/torch_chamfer
RUN cd /tmp/torch_chamfer && \
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
