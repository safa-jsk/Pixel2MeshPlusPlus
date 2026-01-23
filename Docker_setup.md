# Pixel2Mesh++ (Design A / CPU) — Docker Setup on a New PC (Ubuntu 22.04)

This sets up a **reproducible CPU Docker environment** for Pixel2Mesh++ using **TensorFlow 1.15**.

---

## 1) Install Docker (Ubuntu 22.04)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo docker run --rm hello-world
```

## 2) Build Docker Image

```bash
docker build --no-cache -f Dockerfile.cpu -t p2mpp:cpu .
```

## 3) Run Docker Container

```bash
docker run --rm -it \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD":/workspace \
  -w /workspace \
  p2mpp:cpu
```

## 4) Sanity Check inside Container

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tflearn, yaml, cv2; print('imports OK')"
```

## 5) Run the demo (after weights are placed) [Not verified]

```bash
python demo.py --input_folder ./data/demo/ --output_folder ./outputs/demo/ --checkpoint ./ckpt/pretrained_model.ckpt
```

---
