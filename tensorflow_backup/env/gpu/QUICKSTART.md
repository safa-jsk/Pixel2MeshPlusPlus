# Design B Quick Start - RTX4070

## Prerequisites

- RTX4070 GPU with NVIDIA drivers installed
- CUDA 11.2+ on system (or Docker installed)
- 12GB free GPU memory
- Linux system (Ubuntu 20.04+ recommended)

---

## Option 1: Docker (Recommended - No system CUDA needed)

### Step 1: Build Docker image

```bash
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus
docker build -f env/gpu/Dockerfile -t p2mpp-gpu:latest .
```

**Expected time:** 10-15 minutes (TensorFlow 1.x GPU installation is large)

### Step 2: Run container with GPU

```bash
docker run --gpus all -it -v $(pwd):/workspace p2mpp-gpu:latest bash
# Inside container, you're already in /workspace
```

### Step 3: Inside container - Build ops

```bash
cd /workspace
chmod +x env/gpu/build_ops.sh
bash env/gpu/build_ops.sh
```

**Expected output:**

```
==== Design B: Building CUDA Custom Ops ====
...
✓ tf_nndistance_so.so loaded successfully
✓ tf_approxmatch_so.so loaded successfully
All done! .so files are ready for GPU execution.
```

---

## Option 2: Native Environment (If CUDA 11.2+ already installed)

### Step 1: Create Python environment

```bash
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus

# Using pip
pip install -r env/gpu/requirements_gpu.txt

# OR using conda
conda create -n p2mpp-gpu python=3.7
conda activate p2mpp-gpu
pip install -r env/gpu/requirements_gpu.txt
```

### Step 2: Verify TensorFlow sees GPU

```bash
python3 << 'EOF'
import tensorflow as tf
print("GPU devices:", tf.config.list_physical_devices('GPU'))
EOF
```

Should output: `GPU devices: [PhysicalDevice(...)]`

### Step 3: Build ops

```bash
chmod +x env/gpu/build_ops.sh
bash env/gpu/build_ops.sh
```

---

## Running Tests

### Quick Inference Test (5 samples)

```bash
# In terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# In terminal 2: Run inference
python test_p2mpp.py --config cfgs/p2mpp.yaml --checkpoint results/refine_p2mpp/models/meshnet.ckpt-10
```

**Expected GPU behavior:**

- GPU memory rises during first batch
- GPU utilization spikes during `nn_distance` calls
- No errors about missing ops

### Full Benchmark (Design A vs B)

```bash
# This will take 15-30 minutes
chmod +x env/gpu/benchmark.sh
bash env/gpu/benchmark.sh
```

**Output:**

- `outputs/designB/benchmark/design_a_times.txt` - CPU baseline
- `outputs/designB/benchmark/design_b_times.txt` - GPU times
- `outputs/designB/benchmark/logs/` - Detailed logs
- Summary will print speedup factor

---

## Troubleshooting

### Issue: "cannot find -lcudart"

**Fix:** Set CUDA path

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Issue: "failed to load custom op"

**Potential causes:**

1. Op wasn't built - re-run `bash env/gpu/build_ops.sh`
2. ABI mismatch - in `build_ops.sh`, check `-D_GLIBCXX_USE_CXX11_ABI=0` flag
3. Wrong Python version - TF 1.15 needs Python 3.5-3.7

### Issue: GPU not detected after TensorFlow import

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA
nvcc --version

# Force GPU in Docker
docker run --gpus all ...
```

### Issue: Compilation hangs or crashes

```bash
# Try clean rebuild
cd external
make clean
rm -f *.cu.o *.so
# Then re-run build_ops.sh
```

---

## Key Files & Locations

```
Pixel2MeshPlusPlus/
├── env/gpu/
│   ├── Dockerfile                    # Docker image
│   ├── requirements_gpu.txt          # Python deps
│   ├── build_ops.sh                  # ← RUN THIS FIRST
│   ├── setup_and_verify.sh           # One-command setup
│   └── benchmark.sh                  # Run A vs B test
│
├── external/
│   ├── tf_nndistance_so.so          # ← Will be rebuilt
│   ├── tf_approxmatch_so.so         # ← Will be rebuilt
│   ├── tf_nndistance_g.cu           # CUDA kernel
│   └── Makefile                      # Build config
│
├── outputs/designB/
│   ├── benchmark/
│   │   ├── design_a_times.txt
│   │   ├── design_b_times.txt
│   │   └── system_info.txt
│   └── logs/
│       ├── build_log.txt
│       └── gpu_monitor_*.txt
│
└── docs/
    └── designB_gpu_strategy.md       # ← Strategy doc
```

---

## One-Command Quick Setup

```bash
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus
chmod +x env/gpu/setup_and_verify.sh
bash env/gpu/setup_and_verify.sh
```

This will:

1. Check GPU
2. Build ops
3. Verify loading
4. Report any issues

---

## Expected Timeline

| Task             | Time          |
| ---------------- | ------------- |
| Docker build     | 15 min        |
| Op compilation   | 5-10 min      |
| GPU verification | 5 min         |
| Benchmark run    | 20 min        |
| **Total**        | **45-50 min** |

---

## Success Indicators

✅ When you see these, Design B GPU is working:

1. `nvidia-smi` shows RTX4070
2. `tf_nndistance_so.so loaded successfully`
3. GPU memory increases during inference
4. Speedup factor >= 1.5x in benchmark results
5. No errors in `outputs/designB/logs/`

---

## Next Steps

After successful setup:

1. Run full benchmark: `bash env/gpu/benchmark.sh`
2. Visualize results: See `outputs/designB/benchmark/`
3. Generate Report: Create `docs/ch4_designB_spec_and_verification.md`
4. Create poster figures: Side-by-side mesh comparisons

---

## Support

All setup scripts have detailed logging:

- Build logs: `outputs/designB/logs/build_log.txt`
- GPU monitoring: `outputs/designB/logs/gpu_monitor_*.txt`
- Inference logs: `outputs/designB/logs/design_*.log`

Check these if anything fails!
