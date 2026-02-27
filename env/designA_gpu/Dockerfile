FROM tensorflow/tensorflow:2.10.1-gpu

LABEL maintainer="Pixel2Mesh++ Design B GPU"
LABEL description="Design B GPU environment for RTX4070 with TensorFlow 2.10.1 (CUDA 11.8)"

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH

# TF 2.x compatibility mode for TF 1.x code
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Fix NVIDIA repository GPG key issue
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list || true

# Install system dependencies for building CUDA ops
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (excluding tensorflow-gpu since it's already in base image)
COPY env/gpu/requirements_gpu_tf210.txt /tmp/requirements_gpu.txt

# Install additional Python packages (skip tensorflow, add tflearn)
RUN grep -v "tensorflow" /tmp/requirements_gpu.txt > /tmp/requirements_extra.txt && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements_extra.txt && \
    pip install opencv-python-headless==4.5.5.64

# Copy project files for building custom ops
COPY external /workspace/external
WORKDIR /workspace/external

# Compile custom CUDA ops for chamfer distance  
RUN TF_CFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') && \
    TF_LFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') && \
    echo "TF_CFLAGS: $TF_CFLAGS" && \
    echo "TF_LFLAGS: $TF_LFLAGS" && \
    nvcc -std=c++14 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu \
    $TF_CFLAGS -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 --expt-relaxed-constexpr && \
    g++ -std=c++14 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so \
    -shared -fPIC $TF_CFLAGS -L/usr/local/cuda/lib64 -lcudart $TF_LFLAGS -O2 && \
    nvcc -std=c++14 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu \
    $TF_CFLAGS -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 --expt-relaxed-constexpr && \
    g++ -std=c++14 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so \
    -shared -fPIC $TF_CFLAGS -L/usr/local/cuda/lib64 -lcudart $TF_LFLAGS -O2 && \
    ls -lh *.so && \
    echo "✓ Custom CUDA ops compiled successfully"

# Create workspace
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1

# Enable TF 1.x compatibility mode globally
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_DISABLE_CUDA_SOLVER=1

# Verify setup
RUN python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('CUDA available:', tf.test.is_built_with_cuda())"

CMD ["/bin/bash"]
