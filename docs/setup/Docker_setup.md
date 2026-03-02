# Pixel2Mesh++ — Docker Setup (Unified Image)

Single Docker image supporting **all designs** — TF 2.16 + PyTorch 2.5, CUDA 12.4, Ubuntu 24.04.

---

## 1) Prerequisites

### Docker Engine (Ubuntu 24.04)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```

### NVIDIA Container Toolkit (GPU designs)

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 2) Build the Unified Image

From the repository root:

```bash
docker build -t p2mpp:latest .
```

The build takes ~15–20 minutes and:
- Installs TensorFlow 2.16 (GPU) + PyTorch 2.5 (CUDA 12.4)
- Compiles TF custom CUDA ops (Chamfer/EMD)
- Compiles PyTorch Chamfer CUDA extension
- Verifies both frameworks detect GPUs

---

## 3) Run a Design

Each design has a dedicated run script in `docker/`:

| Design | Script | GPU Required |
|--------|--------|:------------:|
| A (CPU) | `bash docker/run_designA_cpu.sh` |  No |
| A (GPU) | `bash docker/run_designA_gpu.sh` | Yes |
| B       | `bash docker/run_designB.sh`     | Yes |
| C       | `bash docker/run_designC.sh`     | Yes |

Run scripts mount the repo at `/workspace`, copy pre-built `.so` files from the
build cache, verify GPU connectivity (if applicable), and drop you into a bash
shell.

### Custom Image Name

Set `P2MPP_IMAGE` to use a different tag:

```bash
P2MPP_IMAGE=p2mpp:dev bash docker/run_designB.sh
```

### Passing Extra Docker Args

Any extra arguments are forwarded to `docker run`:

```bash
bash docker/run_designA_gpu.sh --shm-size 8g -v /data/shapenet:/data
```

---

## 4) Sanity Checks (inside the container)

```bash
# TensorFlow
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"

# PyTorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Core deps
python -c "import tflearn, yaml, cv2, scipy, trimesh; print('imports OK')"
```

---

## 5) Quick Evaluation Example

```bash
# Inside Design A GPU container:
cd DesignA_GPU/scripts
bash eval.sh

# Inside Design B container:
cd DesignB/scripts
bash benchmark.sh
```

---

## Technical Details

| Component | Version |
|-----------|---------|
| Base image | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04` |
| TensorFlow | `>=2.16,<2.18` (`tensorflow[and-cuda]`) |
| PyTorch | `2.5.0` + `torchvision 0.20.0` (cu124) |
| CUDA | 12.4 (bundled in base image) |
| cuDNN | Included via `-cudnn-devel-` image variant |
| C++ standard | C++17 (required by CUDA 12.x toolchain) |
| Python | 3.x (system, Ubuntu 24.04) |
