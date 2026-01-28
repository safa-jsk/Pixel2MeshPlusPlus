# Design B Quick Commands Reference

**Quick copy-paste commands for Design B GPU implementation**

---

## STEP 1: Choose Your Environment

### Option A: Docker (Recommended)

```bash
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus

# Build Docker image (10-15 min)
docker build -f env/gpu/Dockerfile -t p2mpp-gpu:latest .

# Run container with GPU access
docker run --gpus all -it -v $(pwd):/workspace p2mpp-gpu:latest bash

# Inside container:
cd /workspace
```

### Option B: Native Python (if CUDA 11+ installed)

```bash
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus

# Install dependencies
pip install -r env/gpu/requirements_gpu.txt

# Verify GPU detection
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## STEP 2: Setup & Build

### One-Command Setup (Recommended)

```bash
bash env/gpu/setup_and_verify.sh
```

### Or Manual Steps

**Build CUDA ops:**

```bash
bash env/gpu/build_ops.sh
```

**Verify GPU:**

```bash
nvidia-smi
nvcc --version
python3 -c "import tensorflow as tf; print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')"
```

**Test op loading:**

```bash
cd external
python3 << 'EOF'
import tensorflow as tf
try:
    mod = tf.load_op_library('./tf_nndistance_so.so')
    print("✓ Op loaded successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
EOF
cd ..
```

---

## STEP 3: Run Inference Test

**Terminal 1 - Monitor GPU:**

```bash
watch -n 1 nvidia-smi
```

**Terminal 2 - Run inference:**

```bash
# Create test set
head -n 5 data/test_list.txt > data/designB_eval_test.txt

# Run inference
python test_p2mpp.py \
    --config cfgs/p2mpp.yaml \
    --checkpoint results/refine_p2mpp/models/meshnet.ckpt-10 \
    --eval_list data/designB_eval_test.txt
```

---

## STEP 4: Benchmark CPU vs GPU

```bash
bash env/gpu/benchmark.sh
```

**Check results:**

```bash
cat outputs/designB/benchmark/design_a_times.txt
cat outputs/designB/benchmark/design_b_times.txt
```

---

## STEP 5: Troubleshooting Commands

### Check GPU Status

```bash
nvidia-smi
nvidia-smi -l 1  # Update every 1 second
```

### Check CUDA

```bash
nvcc --version
which nvcc
```

### Check TensorFlow GPU

```bash
python3 << 'EOF'
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
EOF
```

### View Logs

```bash
# Build logs
cat outputs/designB/logs/build_log.txt

# Benchmark logs
ls -lh outputs/designB/benchmark/logs/

# Inference logs
tail -f outputs/designB/logs/*.log
```

### Set CUDA Path (if needed)

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

### Verify Build

```bash
cd external
ls -lh tf_*_so.so
python3 -c "import tensorflow as tf; mod = tf.load_op_library('./tf_nndistance_so.so'); print('✓')"
cd ..
```

---

## STEP 6: View Results

**Benchmark results:**

```bash
cat outputs/designB/benchmark/design_a_times.txt
cat outputs/designB/benchmark/design_b_times.txt
```

**GPU info:**

```bash
cat outputs/designB/benchmark/system_info.txt
```

**All logs:**

```bash
ls -lh outputs/designB/logs/
ls -lh outputs/designB/benchmark/logs/
```

---

## COMPLETE WORKFLOW (Copy & Paste)

### Full Docker Setup

```bash
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus
docker build -f env/gpu/Dockerfile -t p2mpp-gpu:latest .
docker run --gpus all -it -v $(pwd):/workspace p2mpp-gpu:latest bash
cd /workspace
bash env/gpu/setup_and_verify.sh
bash env/gpu/benchmark.sh
cat outputs/designB/benchmark/design_*_times.txt
```

### Full Native Setup

```bash
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus
pip install -r env/gpu/requirements_gpu.txt
bash env/gpu/setup_and_verify.sh
bash env/gpu/benchmark.sh
cat outputs/designB/benchmark/design_*_times.txt
```

---

## QUICK REFERENCE

| Command                                       | Purpose                 | Time      |
| --------------------------------------------- | ----------------------- | --------- |
| `docker build ...`                            | Build GPU container     | 15 min    |
| `pip install -r env/gpu/requirements_gpu.txt` | Install Python packages | 10 min    |
| `bash env/gpu/setup_and_verify.sh`            | Auto setup              | 5-10 min  |
| `bash env/gpu/build_ops.sh`                   | Build CUDA ops          | 5-10 min  |
| `bash env/gpu/benchmark.sh`                   | CPU vs GPU test         | 20-30 min |
| `nvidia-smi`                                  | Check GPU               | instant   |
| `python test_p2mpp.py ...`                    | Run inference           | 2-5 min   |

---

## Environment Variables (if needed)

```bash
# Set CUDA location
export CUDA_HOME=/usr/local/cuda

# Add to library path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add to PATH
export PATH=$CUDA_HOME/bin:$PATH

# Enable TensorFlow logging
export TF_CPP_MIN_LOG_LEVEL=0

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## All File Paths

```
design_b/                                      ← You are here
├── README.md
├── SETUP_GUIDE.md
├── QUICK_COMMANDS.md                         ← This file
└── docs/
    ├── GPU_STRATEGY.md
    └── IMPLEMENTATION_ROADMAP.md

env/gpu/                                       ← Scripts here
├── Dockerfile
├── requirements_gpu.txt
├── QUICKSTART.md
├── setup_and_verify.sh
├── build_ops.sh
└── benchmark.sh

Results go to:
└── outputs/designB/
    ├── benchmark/
    ├── logs/
    ├── quality_check/
    └── poster_figs/
```

---

For detailed instructions, see:

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Full setup guide
- [README.md](README.md) - Overview
- [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) - 7-phase plan
