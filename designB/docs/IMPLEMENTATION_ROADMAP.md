# Design B Implementation Roadmap - RTX4070

**Project:** Pixel2Mesh++ CUDA Hotspot Acceleration  
**Hardware:** NVIDIA RTX4070 (12GB VRAM, Compute Capability 8.9)  
**Status:** Ready to Execute  
**Created:** January 28, 2026

---

## Overview

This document outlines the step-by-step implementation plan to execute Design B (GPU acceleration) on your RTX4070.

**Goal:** Add GPU acceleration to the Pixel2Mesh++ inference pipeline without changing the model architecture.

**Expected Outcome:** 3-10x speedup on Chamfer distance computations + benchmarked comparison vs Design A (CPU baseline)

---

## File Organization

All Design B documentation is in `design_b/` folder:

```
design_b/
├── README.md                           ← Overview
├── SETUP_GUIDE.md                      ← Complete setup guide
├── IMPLEMENTATION_ROADMAP.md           ← You are here
├── FILE_INDEX.md                       ← Navigation
├── QUICK_COMMANDS.md                   ← Command reference
│
└── docs/
    ├── GPU_STRATEGY.md                 ← Technical approach
    └── IMPLEMENTATION_ROADMAP.md       ← This file

GPU Environment (separate, stays in root):
└── env/gpu/
    ├── Dockerfile
    ├── requirements_gpu.txt
    ├── QUICKSTART.md
    ├── setup_and_verify.sh
    ├── build_ops.sh
    └── benchmark.sh
```

---

## Phase 1: Environment Setup (45 min)

### Option A: Docker (Recommended)

**Advantages:** Isolated environment, no system dependency conflicts  
**Requirements:** Docker + NVIDIA container toolkit

```bash
# Build image (10-15 min)
docker build -f env/gpu/Dockerfile -t p2mpp-gpu:latest .

# Run container with GPU
docker run --gpus all -it -v $(pwd):/workspace p2mpp-gpu:latest bash

# Inside container:
cd /workspace
bash env/gpu/build_ops.sh
```

### Option B: Native Python Environment

**Advantages:** Direct access to filesystem  
**Requirements:** CUDA 11.2+ already installed on system

```bash
# Install dependencies
pip install -r env/gpu/requirements_gpu.txt

# Verify GPU detection
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Build ops
bash env/gpu/build_ops.sh
```

**Verification:** You should see GPU detected in both cases.

---

## Phase 2: Build Custom CUDA Ops (5-10 min)

The script `env/gpu/build_ops.sh` will:

1. Detect CUDA/TensorFlow on your system
2. Compile `tf_nndistance_so.so` and `tf_approxmatch_so.so`
3. Test that ops load successfully

```bash
bash env/gpu/build_ops.sh
```

**Expected Output:**

```
==== Design B: Building CUDA Custom Ops ====
GPU: RTX4070 (Compute Capability 8.9)
CUDA: 11.2+
...
✓ tf_nndistance_so.so loaded successfully
✓ tf_approxmatch_so.so loaded successfully
All done! .so files are ready for GPU execution.
```

**If build fails:** See troubleshooting in [SETUP_GUIDE.md](../SETUP_GUIDE.md)

---

## Phase 3: Verify GPU Integration (10 min)

### Quick GPU Detection Test

```bash
python3 << 'EOF'
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")
EOF
```

### Test Op Loading

```bash
cd external
python3 << 'EOF'
import tensorflow as tf
try:
    mod = tf.load_op_library('./tf_nndistance_so.so')
    print("✓ Op loads successfully - GPU ready!")
except Exception as e:
    print(f"✗ Op load failed: {e}")
EOF
cd ..
```

---

## Phase 4: Run Inference Test (5-10 min)

Before benchmarking, verify inference works end-to-end:

### Terminal 1: Monitor GPU

```bash
watch -n 1 nvidia-smi
```

Watch for GPU memory usage and utilization during inference.

### Terminal 2: Run inference

```bash
# Use first 5 samples as quick test
head -n 5 data/test_list.txt > data/designB_eval_test.txt

python test_p2mpp.py \
    --config cfgs/p2mpp.yaml \
    --checkpoint results/refine_p2mpp/models/meshnet.ckpt-10 \
    --eval_list data/designB_eval_test.txt
```

**Success indicators:**

- No errors about missing ops
- GPU memory > 100 MB in `nvidia-smi`
- GPU utilization > 20% during inference
- Mesh output files generated in `outputs/`

---

## Phase 5: Benchmark Design A vs B (20 min)

This is the core measurement: CPU baseline vs GPU acceleration.

```bash
bash env/gpu/benchmark.sh
```

This script will:

1. Run 5-sample test on CPU (Design A) with warmup
2. Run same 5 samples on GPU (Design B) with warmup
3. Compare runtimes and calculate speedup

**Output locations:**

- Raw timings: `outputs/designB/benchmark/design_a_times.txt`, `design_b_times.txt`
- GPU monitoring: `outputs/designB/benchmark/gpu_*.txt`
- Detailed logs: `outputs/designB/benchmark/logs/`

**Expected results:**

```
DESIGN A (CPU):  Mean time: 120.5s ± 2.3s
DESIGN B (GPU):   Mean time: 45.2s ± 1.8s
SPEEDUP:          2.67x faster
```

(Actual numbers depend on your system and sample count)

---

## Phase 6: Quality Verification (10 min)

Verify outputs are correct (Design B produces same meshes as Design A):

```bash
# Compare first sample from both runs
python3 utils/visualize.py \
    --mesh_a outputs/designA/results/sample_001.obj \
    --mesh_b outputs/designB/results/sample_001.obj \
    --output outputs/designB/quality_check/comparison_001.png
```

**Success criteria:**

- Meshes look qualitatively identical
- Vertex/face counts match
- No NaN or corrupted geometry

---

## Phase 7: Generate Report (15 min)

Document your results:

### Create Design B Specification Document

```bash
cat > docs/ch4_designB_spec_and_verification.md << 'EOF'
# Design B: GPU Acceleration Specification & Verification

## Implementation Summary
- **Approach:** Enable existing custom CUDA ops (tf_nndistance, tf_approxmatch)
- **Hardware:** RTX4070 with CUDA 11.2
- **Model:** No architecture changes from Design A

## Benchmark Results

| Metric | Design A (CPU) | Design B (GPU) | Improvement |
|--------|---|---|---|
| Avg time per 5 samples | XX.Xs | YY.Ys | ZZ.Zx speedup |
| GPU memory usage | N/A | ~XXX MB | - |
| GPU utilization | 0% | ~YY% | - |

## Quality Check
- ✓ Mesh outputs identical to Design A
- ✓ No geometric corruption
- ✓ GPU successfully used for NN distance computation

## Files Modified
- external/tf_nndistance_so.so (rebuilt for RTX4070)
- external/tf_approxmatch_so.so (rebuilt for RTX4070)

EOF
```

### Create Performance Comparison Figure

```bash
# Extract and visualize benchmark data
python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import statistics

# Read benchmark results
with open("outputs/designB/benchmark/design_a_times.txt") as f:
    times_a = [float(x) for x in f if x.strip()]
with open("outputs/designB/benchmark/design_b_times.txt") as f:
    times_b = [float(x) for x in f if x.strip()]

# Create comparison bar chart
fig, ax = plt.subplots(figsize=(8, 6))
categories = ['CPU (Design A)', 'GPU (Design B)']
means = [statistics.mean(times_a), statistics.mean(times_b)]
stds = [statistics.stdev(times_a) if len(times_a)>1 else 0,
        statistics.stdev(times_b) if len(times_b)>1 else 0]

ax.bar(categories, means, yerr=stds, capsize=10, color=['blue', 'green'], alpha=0.7)
ax.set_ylabel('Time (seconds)')
ax.set_title('Design A vs Design B Benchmark (RTX4070)')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/designB/poster_figs/a_vs_b_runtime_comparison.png', dpi=150)
print(f"Saved comparison chart")
EOF
```

---

## Timeline

| Phase     | Task                           | Time          | Status               |
| --------- | ------------------------------ | ------------- | -------------------- |
| 1         | Environment setup (Docker/Env) | 15 min        | Ready                |
| 2         | Build custom ops               | 5-10 min      | Ready                |
| 3         | Verify GPU integration         | 10 min        | Ready                |
| 4         | Inference test                 | 5-10 min      | Ready                |
| 5         | Benchmark (A vs B)             | 20 min        | Ready                |
| 6         | Quality check                  | 10 min        | Ready                |
| 7         | Report generation              | 15 min        | Ready                |
| **Total** |                                | **80-90 min** | **Ready to Execute** |

---

## Execution Checklist

### Pre-Execution

- [ ] RTX4070 detected by `nvidia-smi`
- [ ] CUDA 11+ visible (`nvcc --version`)
- [ ] Enough disk space (20GB for Docker, 5GB for native)
- [ ] 2-3 hours blocked for full execution

### Execution

- [ ] Environment setup complete (`docker build` or `pip install`)
- [ ] Custom ops built successfully (`build_ops.sh`)
- [ ] Ops load without errors (test in Python)
- [ ] Inference runs on test data
- [ ] Benchmark completes (check `outputs/designB/benchmark/`)
- [ ] Speedup measured (target >= 1.5x)
- [ ] Quality check passed (meshes look good)
- [ ] Report generated (`docs/ch4_designB_spec_and_verification.md`)

### Post-Execution

- [ ] Commit changes to `design-b-cuda-hotspots` branch
- [ ] Tag commit with speedup metadata
- [ ] Archive benchmark logs
- [ ] Generate poster figures

---

## Quick Start (TL;DR)

```bash
# Navigate to repo
cd /home/crystal/Documents/Thesis/Pixel2MeshPlusPlus

# Option 1: Docker
docker build -f env/gpu/Dockerfile -t p2mpp-gpu:latest .
docker run --gpus all -it -v $(pwd):/workspace p2mpp-gpu:latest bash
cd /workspace && bash env/gpu/build_ops.sh

# Option 2: Native
pip install -r env/gpu/requirements_gpu.txt
bash env/gpu/build_ops.sh

# Then run benchmark
bash env/gpu/benchmark.sh

# Check results
cat outputs/designB/benchmark/design_*_times.txt
```

---

## Support & Troubleshooting

### Common Issues

**Q: "cannot find -lcudart"**  
A: Set `export CUDA_HOME=/usr/local/cuda` before running build_ops.sh

**Q: "failed to load custom NNDistance op"**  
A: Check `outputs/designB/logs/build_log.txt` for compilation errors; may need to adjust ABI flags

**Q: "No GPU detected"**  
A: Verify with `nvidia-smi`; in Docker, use `docker run --gpus all ...`

**Q: "ModuleNotFoundError: tensorflow"**  
A: Activate environment: `pip install tensorflow-gpu==1.15.5` or `conda activate p2mpp-gpu`

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Check CUDA installation
nvcc --version

# Check TensorFlow GPU support
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# View build logs
cat outputs/designB/logs/build_log.txt

# View inference logs
ls -lh outputs/designB/logs/*.log
```

---

## What Happens Next (Design C)

After Design B is complete:

- **Design C** will introduce code modernization (optional)
- **Design B remains unchanged** - pure GPU acceleration, same architecture
- Comparison: A (CPU) vs B (GPU) vs C (Modern)

---

## Files & Locations

All Design B documentation is organized:

```
design_b/
├── README.md                    ← Overview
├── SETUP_GUIDE.md              ← Setup instructions
├── IMPLEMENTATION_ROADMAP.md   ← This file
├── FILE_INDEX.md               ← Complete navigation
├── QUICK_COMMANDS.md           ← Command reference
│
└── docs/
    ├── GPU_STRATEGY.md         ← Technical strategy
    └── IMPLEMENTATION_ROADMAP.md ← This file

GPU Environment (stays in env/gpu/):
└── env/gpu/
    ├── Dockerfile
    ├── requirements_gpu.txt
    ├── QUICKSTART.md
    ├── setup_and_verify.sh
    ├── build_ops.sh
    └── benchmark.sh

Results (will be generated):
└── outputs/designB/
    ├── benchmark/
    ├── logs/
    ├── quality_check/
    └── poster_figs/
```

---

## Success Criteria

Design B is **DONE** when:

1. ✅ Custom ops compile without errors
2. ✅ Ops load in Python (`tf.load_op_library` succeeds)
3. ✅ Inference runs on RTX4070 (no errors)
4. ✅ Speedup >= 1.5x measured
5. ✅ Output meshes match Design A (visual inspection)
6. ✅ Benchmark results documented in `outputs/designB/`
7. ✅ Report written in `docs/ch4_designB_spec_and_verification.md`

---

**Status:** ✅ **READY TO EXECUTE**

All infrastructure is in place. You can start with Phase 1 immediately.

**Next Action:** See [SETUP_GUIDE.md](../SETUP_GUIDE.md) - Pick environment and run setup.
