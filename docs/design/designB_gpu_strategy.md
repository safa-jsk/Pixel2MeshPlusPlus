# Design B GPU Strategy - RTX4070 Implementation

**Date:** January 28, 2026  
**Target Hardware:** NVIDIA RTX4070 (12GB VRAM, Compute Capability 8.9)  
**Framework:** TensorFlow 1.x GPU

---

## Executive Summary

Design B implements **GPU acceleration on custom CUDA ops** (B-Path-1: Preferred approach) using TensorFlow 1.x with the RTX4070.

The strategy focuses on enabling the pre-existing custom ops (`tf_nndistance_so.so`, `tf_approxmatch_so.so`) for GPU execution without modifying the model architecture.

---

## Selected Path: B-Path-1 (GPU Custom Op)

### Why This Path?

1. **Existing Infrastructure:** The repo already has compiled CUDA kernels (`tf_nndistance_g.cu`, `tf_approxmatch_g.cu`)
2. **Major Bottleneck:** Nearest-neighbor / Chamfer distance is the primary hotspot in Pixel2Mesh++ inference
3. **No Architecture Changes:** Using existing ops maintains fair A vs B comparison
4. **Hardware Fit:** RTX4070 has modern CUDA compute capability (8.9) compatible with latest CUDA toolkits

### Expected Improvements

- **Memory:** RTX4070 has 12GB VRAM (sufficient for inference batches)
- **Compute:** 5888 CUDA cores at 2.475 GHz
- **Target Speedup:** 5-10x on Chamfer distance operations (typical for GPU vs CPU)

---

## Implementation Steps

### Phase 1: Environment Setup (1-2 hours)

**Challenges with TensorFlow 1.x on modern hardware:**

- TF 1.x officially ends at Python 3.7, CUDA 10.0
- RTX4070 requires CUDA 11+ for stable driver support
- Solution: Use **TensorFlow 1.15.5** (last TF 1.x release) with CUDA 11.2 + compatible cuDNN

**Environment Options:**

1. **Docker (Recommended)**
   - Use `nvidia/cuda:11.2.2-cudnn8-runtime` base image
   - Pre-configured CUDA/cuDNN stack
   - Isolates legacy TF1.x from system Python
   - Command: `docker build -f env/gpu/Dockerfile -t p2mpp-gpu:latest .`

2. **Native Python (Alternative)**
   - Create conda/venv environment
   - Manual CUDA 11.2 + cuDNN 8.1 installation
   - Higher chance of conflicts but more control

**Verification Steps:**

```bash
nvidia-smi          # GPU visible
nvcc --version      # CUDA toolkit found
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Phase 2: Build Custom Ops (1-2 hours)

**Build Process:**

1. Extract TensorFlow compile/link flags:

   ```bash
   python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
   ```

2. Compile CUDA kernels with RTX4070 optimizations:

   ```bash
   nvcc -arch=sm_89 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c tf_nndistance_g.cu ...
   ```

3. Link into shared libraries (`.so` files)

4. Test loading:
   ```bash
   python -c "import tensorflow as tf; tf.load_op_library('./external/tf_nndistance_so.so')"
   ```

**Provided Script:**

- `env/gpu/build_ops.sh` - Automated build with error checking

**Typical Build Time:** 5-10 minutes

**Artifacts:**

- `external/tf_nndistance_so.so` (rebuilt)
- `external/tf_approxmatch_so.so` (rebuilt)
- `outputs/designB/logs/build_log.txt` (full build output)

### Phase 3: Verify GPU Placement (30 minutes)

**Confirm that expensive ops run on GPU:**

1. Monitor GPU during inference:

   ```bash
   watch -n 1 nvidia-smi
   ```

2. Expected observations:
   - GPU memory usage > 0 MB (after warmup)
   - GPU utilization spikes during `nn_distance` calls
   - No CPU overload warnings

3. Optional: Enable TensorFlow device logs:
   ```python
   import os
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Enable logging
   ```

---

## Risks & Mitigations

| Risk                                      | Probability | Mitigation                                                     |
| ----------------------------------------- | ----------- | -------------------------------------------------------------- |
| CUDA 11.2 / TF 1.15 incompatibility       | Medium      | Test import in isolated env first; fallback to CUDA 10.0       |
| Custom op compile failure (linker errors) | Medium      | Verify ABI flags; pin versions; check GCC/G++ versions         |
| Minimal speedup (I/O bound)               | Low         | Profile code; if bottleneck is elsewhere, document in Design C |
| RTX4070 driver issues                     | Low         | Use latest driver; containerize for reproducibility            |

---

## Success Criteria

Design B is **successful** when:

1. ✅ Custom ops compile without errors
2. ✅ Ops load successfully in Python (no import errors)
3. ✅ Inference runs on same data as Design A
4. ✅ GPU utilization > 30% during inference (nvidia-smi shows memory use)
5. ✅ Output meshes match Design A qualitatively (no shape degradation)
6. ✅ Speedup >= 1.5x (any speedup is valid, but 3-5x expected)

---

## Files Created/Modified

### New Files:

- `env/gpu/Dockerfile` - Container with TF 1.x GPU
- `env/gpu/requirements_gpu.txt` - Python dependencies
- `env/gpu/build_ops.sh` - Automated CUDA op build
- `env/gpu/setup_and_verify.sh` - Setup + verification
- `env/gpu/benchmark.sh` - A vs B benchmarking
- `outputs/designB/logs/build_log.txt` - Build output
- `outputs/designB/benchmark/system_info.txt` - GPU info

### Unchanged:

- Model code (modules/models_p2mpp.py)
- Weights (results/_/models/_)
- Config (cfgs/p2mpp.yaml)

---

## Timeline

| Phase      | Duration   | Deliverable                     |
| ---------- | ---------- | ------------------------------- |
| Env setup  | 1-2h       | Docker image built, GPU visible |
| Build ops  | 1-2h       | .so files compiled, ops load    |
| Verify GPU | 30m        | GPU memory used, logs clean     |
| Benchmark  | 1h         | A vs B times recorded           |
| **Total**  | **3-5.5h** | **Speedup measured**            |

---

## Next Steps (after successful build)

1. Run benchmark: `bash env/gpu/benchmark.sh`
2. Create quality comparison: 5-10 mesh side-by-side renderings
3. Document results in `docs/ch4_designB_spec_and_verification.md`
4. Generate poster figures for Chapter 4

---

## References

- TensorFlow 1.x GPU Guide: https://www.tensorflow.org/install/gpu_setup
- NVIDIA CUDA 11.2 + cuDNN: https://developer.nvidia.com/cuda-11.2-toolkit
- RTX4070 Specs: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/
