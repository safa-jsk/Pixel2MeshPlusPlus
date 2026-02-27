# Design B Setup Guide - RTX4070 GPU Acceleration

**Date Created:** January 28, 2026  
**Status:** ✅ READY TO EXECUTE  
**Hardware:** RTX4070 (12GB VRAM)

---

## Overview

This guide provides complete setup instructions for Design B GPU acceleration on your RTX4070. The implementation adds GPU acceleration to the Pixel2Mesh++ inference pipeline using existing CUDA custom ops.

**Expected Outcome:** 3-10x speedup on Chamfer distance computations, benchmarked against Design A (CPU baseline).

---

## Files Organization

```
Pixel2MeshPlusPlus/
├── design_b/                              ← Design B Documentation
│   ├── README.md
│   ├── SETUP_GUIDE.md                     ← You are here
│   ├── FILE_INDEX.md
│   ├── QUICK_COMMANDS.md
│   └── docs/
│       ├── GPU_STRATEGY.md
│       └── IMPLEMENTATION_ROADMAP.md
│
├── env/gpu/                               ← GPU Environment (stays separate)
│   ├── Dockerfile
│   ├── requirements_gpu.txt
│   ├── QUICKSTART.md
│   ├── setup_and_verify.sh
│   ├── build_ops.sh
│   └── benchmark.sh
│
└── outputs/designB/                       ← Results (will be generated)
    ├── benchmark/
    ├── logs/
    ├── quality_check/
    └── poster_figs/
```

---

## Quick Start (5 minutes)

### Option A: Docker (Recommended - No system CUDA needed)

```bash
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus

# Build Docker image (10-15 min)
docker build -f env/gpu/Dockerfile -t p2mpp-gpu:latest .

# Run container with GPU access
docker run --gpus all -it -v $(pwd):/workspace p2mpp-gpu:latest bash

# Inside container, setup and verify
cd /workspace
bash env/gpu/setup_and_verify.sh
```

### Option B: Native Python (If CUDA 11+ installed)

```bash
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus

# Install Python packages
pip install -r env/gpu/requirements_gpu.txt

# Setup and verify
bash env/gpu/setup_and_verify.sh
```

---

## 7-Phase Implementation

### Phase 1: Environment Setup (15 min)

- Docker build or pip install TensorFlow 1.x GPU
- Verify GPU detected: `nvidia-smi`

### Phase 2: Build CUDA Ops (5-10 min)

```bash
bash env/gpu/build_ops.sh
```

- Compiles `tf_nndistance_so.so` and `tf_approxmatch_so.so`
- Creates optimized kernels for RTX4070

### Phase 3: Verify GPU Integration (10 min)

```bash
# Check GPU detection
nvidia-smi

# Test op loading
cd external
python3 -c "import tensorflow as tf; tf.load_op_library('./tf_nndistance_so.so')"
```

### Phase 4: Run Inference Test (5-10 min)

```bash
# Create small test set
head -n 5 data/test_list.txt > data/designB_eval_test.txt

# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Run inference
python test_p2mpp.py --config cfgs/p2mpp.yaml --checkpoint results/refine_p2mpp/models/meshnet.ckpt-10
```

### Phase 5: Benchmark CPU vs GPU (20 min)

```bash
bash env/gpu/benchmark.sh
```

- Measures CPU (Design A) baseline
- Measures GPU (Design B) performance
- Calculates speedup factor
- Saves results to `outputs/designB/benchmark/`

### Phase 6: Quality Verification (10 min)

- Compare 5-10 output meshes from both runs
- Visual inspection: meshes should look identical
- Check vertex/face counts match

### Phase 7: Generate Report (15 min)

- Document results in `docs/ch4_designB_spec_and_verification.md`
- Create comparison figures
- Archive benchmark logs

---

## Expected Results

| Metric          | Value                 |
| --------------- | --------------------- |
| GPU Memory Used | 2-4 GB                |
| GPU Utilization | 40-80%                |
| Speedup Factor  | 3-10x                 |
| Setup Time      | 45-90 min             |
| Output Quality  | Identical to Design A |

---

## Key Scripts

### env/gpu/setup_and_verify.sh (Recommended - One command)

```bash
bash env/gpu/setup_and_verify.sh
```

Runs all setup steps:

1. Checks GPU availability
2. Builds custom ops
3. Tests op loading
4. Reports any issues

**Duration:** 5-10 minutes (after environment setup)

### env/gpu/build_ops.sh (Manual building)

```bash
bash env/gpu/build_ops.sh
```

Compiles CUDA kernels for RTX4070:

- Detects CUDA/TensorFlow configuration
- Compiles tf_nndistance and tf_approxmatch
- Creates .so files
- Tests loading

**Duration:** 5-10 minutes

### env/gpu/benchmark.sh (Performance testing)

```bash
bash env/gpu/benchmark.sh
```

Measures CPU vs GPU performance:

- Runs Design A (CPU) on 5 samples
- Runs Design B (GPU) on same samples
- Measures runtimes
- Calculates speedup

**Duration:** 20-30 minutes

---

## Troubleshooting

### GPU Not Detected

```bash
# Check if nvidia-smi works
nvidia-smi

# In Docker, add --gpus flag
docker run --gpus all ...

# Check TensorFlow GPU support
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Build Fails

```bash
# Check build logs
cat outputs/designB/logs/build_log.txt

# Set CUDA path if needed
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rebuild
bash env/gpu/build_ops.sh
```

### Op Load Fails

- Check ABI compatibility: `-D_GLIBCXX_USE_CXX11_ABI=0` in build_ops.sh
- Verify GCC/G++ versions match TensorFlow compilation
- See detailed logs: `outputs/designB/logs/build_log.txt`

### Minimal Speedup

- Profile to find actual bottleneck
- Check GPU memory allocation
- Monitor GPU utilization: `nvidia-smi`
- See full troubleshooting in [env/gpu/QUICKSTART.md](../env/gpu/QUICKSTART.md)

---

## Success Criteria

✅ Design B is successful when:

1. **Build Phase:**
   - `build_ops.sh` completes without errors
   - Output shows "✓ ops loaded successfully"

2. **Inference Phase:**
   - `nvidia-smi` shows GPU memory used
   - No CUDA errors in logs
   - Mesh outputs generated

3. **Benchmark Phase:**
   - Speedup >= 1.5x measured (typically 3-10x)
   - Results saved to `outputs/designB/benchmark/`

4. **Quality Phase:**
   - Output meshes visually identical to Design A
   - No geometric corruption

5. **Documentation Phase:**
   - Report written: `docs/ch4_designB_spec_and_verification.md`
   - Figures generated: `outputs/designB/poster_figs/`

---

## Files & Paths

### Documentation

- [design_b/README.md](README.md) - Folder overview
- [design_b/SETUP_GUIDE.md](SETUP_GUIDE.md) - This file
- [design_b/FILE_INDEX.md](FILE_INDEX.md) - Complete file navigation
- [design_b/QUICK_COMMANDS.md](QUICK_COMMANDS.md) - Command reference
- [design_b/docs/GPU_STRATEGY.md](docs/GPU_STRATEGY.md) - Technical strategy
- [design_b/docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) - Full plan

### Environment & Scripts (in env/gpu/)

- `Dockerfile` - Docker container setup
- `requirements_gpu.txt` - Python dependencies
- `QUICKSTART.md` - Quick reference
- `setup_and_verify.sh` - Auto setup (executable)
- `build_ops.sh` - Build ops (executable)
- `benchmark.sh` - Benchmarking (executable)

### Results (will be generated)

- `outputs/designB/benchmark/` - Timing data
- `outputs/designB/logs/` - Build and execution logs
- `outputs/designB/quality_check/` - Comparison images
- `outputs/designB/poster_figs/` - Thesis figures

---

## Command Quick Reference

```bash
# Setup (pick one option)
docker build -f env/gpu/Dockerfile -t p2mpp-gpu:latest .
docker run --gpus all -it -v $(pwd):/workspace p2mpp-gpu:latest bash

# OR
pip install -r env/gpu/requirements_gpu.txt

# Then run setup
bash env/gpu/setup_and_verify.sh

# Benchmark
bash env/gpu/benchmark.sh

# Check results
cat outputs/designB/benchmark/design_*_times.txt
```

See [QUICK_COMMANDS.md](QUICK_COMMANDS.md) for full command reference.

---

## Next Steps

1. **Now:** Read this guide (you're done!)
2. **Next:** Choose Docker or Native and run setup
3. **Then:** Execute `bash env/gpu/setup_and_verify.sh`
4. **Then:** Run `bash env/gpu/benchmark.sh`
5. **Finally:** Document results and generate figures

---

## Timeline

- Environment setup: 15 min
- Build + verify: 15-20 min
- Inference test: 10 min
- Benchmark: 20-30 min
- Quality check: 10 min
- Report: 15 min
- **Total: 80-90 minutes**

---

## Support

All scripts have detailed logging:

- Build logs: `outputs/designB/logs/build_log.txt`
- GPU monitoring: `outputs/designB/logs/gpu_monitor_*.txt`
- Inference logs: `outputs/designB/benchmark/logs/*.log`

For more details:

- [FILE_INDEX.md](FILE_INDEX.md) - Complete file navigation
- [QUICK_COMMANDS.md](QUICK_COMMANDS.md) - Command reference
- [docs/GPU_STRATEGY.md](docs/GPU_STRATEGY.md) - Technical details
- [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) - Full plan
- [../env/gpu/QUICKSTART.md](../env/gpu/QUICKSTART.md) - Quick start

---

**Status:** ✅ Ready to Execute

Start with Step 1: Choose your environment option and begin setup.
