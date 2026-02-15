# Design A GPU Environment
# TensorFlow 2.12 with TF1 compatibility mode for RTX 40xx GPUs (CUDA 11.8)
FROM tensorflow/tensorflow:2.12.0-gpu

LABEL maintainer="Design A GPU Evaluation"
LABEL description="Design A with GPU enabled (TensorFlow 2.12 in TF1 compat mode)"

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV PYTHONUNBUFFERED=1

# Fix NVIDIA repository GPG key issue
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list || true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip && \
    pip install \
    numpy==1.23.5 \
    scipy==1.9.3 \
    matplotlib==3.6.2 \
    scikit-image==0.19.3 \
    Pillow==9.3.0 \
    tqdm==4.64.1 \
    pyyaml==6.0 \
    tflearn==0.5.0 \
    opencv-python-headless==4.6.0.66

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python", "--version"]
