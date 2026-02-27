FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="Pixel2Mesh++ PyTorch Implementation"
LABEL description="Design B GPU with PyTorch 2.0 - RTX4070 Native Support"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    bc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install \
    scipy>=1.7 \
    matplotlib>=3.5 \
    scikit-image>=0.19 \
    Pillow>=9.0 \
    tqdm>=4.60 \
    pyyaml>=5.4 \
    opencv-python-headless>=4.5 \
    trimesh>=3.9 \
    "numpy>=1.19,<1.24"

# Install PyTorch3D dependencies
RUN pip install fvcore iopath

# Install PyTorch3D from pre-built wheel (compatible with PyTorch 2.0.1 + CUDA 11.7)
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html || \
    echo "[WARN] PyTorch3D pre-built wheel not available, will use custom implementation"

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1

# Verify GPU setup
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

CMD ["/bin/bash"]
