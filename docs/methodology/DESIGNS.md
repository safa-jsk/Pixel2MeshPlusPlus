# Design Specifications - Pixel2Mesh++

This document details the four design implementations of Pixel2Mesh++ used in the thesis methodology.

---

## Design Summary Table

| Design           | Framework          | Compute Target | Key Optimizations             | Status      |
| ---------------- | ------------------ | -------------- | ----------------------------- | ----------- |
| **DESIGN.A**     | TensorFlow 1.15    | CPU            | None (baseline)               | ✅ Complete |
| **DESIGN.A_GPU** | TensorFlow 1.x/2.x | GPU            | Simple CUDA enablement        | ✅ Complete |
| **DESIGN.B**     | PyTorch 2.1+       | GPU            | CAMFM A2a-A2d optimizations   | ✅ Complete |
| **DESIGN.C**     | PyTorch 2.1+       | GPU            | DESIGN.B + DALI data pipeline | 🚧 Planned  |

---

## DESIGN.A - TensorFlow CPU Baseline

### Overview

Original Pixel2Mesh++ implementation running on CPU. Serves as the performance baseline for comparison.

### Configuration

| Property              | Value                                |
| --------------------- | ------------------------------------ |
| **Entrypoint Script** | `designA/run_designA_eval.sh`        |
| **Stage 1 Script**    | `designA/eval_designA_stage1.py`     |
| **Stage 2 Script**    | `designA/eval_designA_stage2.py`     |
| **Metrics Script**    | `designA/compute_metrics.py`         |
| **Config File**       | `cfgs/mvp2m.yaml`, `cfgs/p2mpp.yaml` |
| **Docker Required**   | Yes (`p2mpp:cpu` container)          |

### Key Config Flags

```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
sesscfg = tf.ConfigProto()
sesscfg.gpu_options.allow_growth = True
sesscfg.allow_soft_placement = True
```

### Expected Outputs

| Output Type     | Path                                             | Description      |
| --------------- | ------------------------------------------------ | ---------------- |
| Coarse meshes   | `outputs/designA/eval_meshes/*_predict.xyz`      | Stage 1 output   |
| Refined meshes  | `outputs/designA/eval_meshes/*_predict.xyz`      | Stage 2 output   |
| Ground truth    | `outputs/designA/eval_meshes/*_ground.xyz`       | GT point clouds  |
| Stage 1 timing  | `outputs/designA/benchmark/stage1_timings.txt`   | Per-stage timing |
| Stage 2 timing  | `outputs/designA/benchmark/stage2_timings.txt`   | Per-stage timing |
| Combined timing | `outputs/designA/benchmark/combined_timings.txt` | Total timing     |
| Metrics CSV     | `outputs/designA/benchmark/metrics_results.csv`  | CD, F1@τ, F1@2τ  |

### Timing Measurement Location

```python
# designA/eval_designA_stage1.py:102-105
t_start = time.time()
out3 = sess.run(model.output3, feed_dict=feed_dict)
t_elapsed = time.time() - t_start
timings.append(t_elapsed)
```

---

## DESIGN.A_GPU - TensorFlow GPU

### Overview

Design A with simple GPU enablement. No other optimizations applied. Demonstrates baseline GPU benefit without algorithmic changes.

### Configuration

| Property              | Value                                      |
| --------------------- | ------------------------------------------ |
| **Entrypoint Script** | `designA_GPU/run_eval_gpu.sh`              |
| **Main Script**       | `designA_GPU/eval_designA_gpu.py`          |
| **Alternative**       | `designA_GPU/eval_designA_gpu_complete.py` |
| **Docker File**       | `designA_GPU/Dockerfile`                   |

### Key Config Flags (Difference from DESIGN.A)

```python
# GPU ENABLED instead of CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Enable GPU

sesscfg = tf.ConfigProto()
sesscfg.gpu_options.allow_growth = True   # Dynamic GPU memory
sesscfg.allow_soft_placement = True       # Allow CPU fallback
sesscfg.log_device_placement = False      # Debug: device placement
```

### Expected Outputs

| Output Type    | Path                                                     | Description       |
| -------------- | -------------------------------------------------------- | ----------------- |
| Refined meshes | `outputs/designA_GPU/eval_meshes/*_predict.xyz`          | Final output      |
| OBJ meshes     | `outputs/designA_GPU/eval_meshes/*_predict.obj`          | OBJ format        |
| Timing CSV     | `outputs/designA_GPU/eval_meshes/timing_results.csv`     | Per-sample timing |
| Summary        | `outputs/designA_GPU/eval_meshes/evaluation_summary.txt` | Stats summary     |

### Timing Measurement Location

```python
# designA_GPU/eval_designA_gpu.py:176-183
start_time = time.time()
out1l, out2l = sess.run([model.output1l, model.output2l], feed_dict=feed_dict)
elapsed = time.time() - start_time
timing_results.append((data_id, elapsed))
```

---

## DESIGN.B - PyTorch GPU Optimized

### Overview

Complete PyTorch reimplementation with CAMFM optimization methodology applied:

- **CAMFM.A2a_GPU_RESIDENCY**: All tensors on GPU, no CPU fallbacks
- **CAMFM.A2b_STEADY_STATE**: Warmup + cuDNN autotune + TF32
- **CAMFM.A2c_MEM_LAYOUT**: Contiguous tensors + pre-allocated buffers
- **CAMFM.A2d_OPTIONAL_ACCEL**: AMP/torch.compile (optional)
- **CAMFM.A3_METRICS**: Integrated quality + performance metrics
- **CAMFM.A5_METHOD**: Reproducible evaluation protocol

### Configuration

| Property              | Value                                  |
| --------------------- | -------------------------------------- |
| **Entrypoint Script** | `designB/fast_inference_v4.py`         |
| **With Metrics**      | `designB/fast_inference_v4_metrics.py` |
| **Quick Commands**    | `designB/QUICK_COMMANDS.md`            |
| **Setup Guide**       | `designB/SETUP_GUIDE.md`               |

### Key Config Flags

```python
# CAMFM.A2b_STEADY_STATE: cuDNN and TF32 optimization
torch.backends.cudnn.benchmark = True        # Auto-tune convolutions
torch.backends.cuda.matmul.allow_tf32 = True # TF32 for matmul
torch.backends.cudnn.allow_tf32 = True       # TF32 for cuDNN

# CAMFM.A2c_MEM_LAYOUT: Contiguous memory
self.initial_coord = torch.from_numpy(...).to(self.device).contiguous()

# CAMFM.A2d_OPTIONAL_ACCEL: Inference mode
@torch.inference_mode()
def infer(self, imgs, cameras):
    ...
```

### Expected Outputs

| Output Type      | Path                                            | Description                  |
| ---------------- | ----------------------------------------------- | ---------------------------- |
| Refined meshes   | `outputs/designB/eval_meshes/*_predict.xyz`     | Point cloud format           |
| OBJ meshes       | `outputs/designB/eval_meshes/*_predict.obj`     | OBJ format                   |
| Timing (console) | stdout                                          | Per-sample and summary stats |
| Metrics CSV      | `outputs/designB/benchmark/metrics_results.csv` | Full metrics                 |

### Timing Measurement Location

```python
# designB/fast_inference_v4.py:298-303
torch.cuda.synchronize()  # CAMFM.A2b: Proper boundary
start = time.time()
mesh_gpu = engine.infer(imgs_tensor, poses)
torch.cuda.synchronize()  # CAMFM.A2b: Proper boundary
elapsed = time.time() - start
times.append(elapsed)
```

### CAMFM Stage Implementation

| CAMFM Stage        | Implementation Location                                        |
| ------------------ | -------------------------------------------------------------- |
| A2a_GPU_RESIDENCY  | `MaxSpeedInferenceEngine.__init__()` - all `.to(device)` calls |
| A2b_STEADY_STATE   | Line 24-26 (backends), `_warmup()` method                      |
| A2c_MEM_LAYOUT     | Line 46-53 `.contiguous()` calls, `delta_coord` pre-allocation |
| A2d_OPTIONAL_ACCEL | `@torch.inference_mode()` decorator                            |
| A3_METRICS         | `fast_inference_v4_metrics.py` - integrated metrics            |
| A5_METHOD          | Evaluation list, seed fixing, documented protocol              |

---

## DESIGN.C - PyTorch + DALI Data Pipeline (Planned)

### Overview

Extension of Design B with GPU-native data pipeline using NVIDIA DALI for FaceScape dataset.

### Target Data Pipeline

```
DATA.READ_CPU           → File system read
DATA.DECODE_GPU_NVJPEG  → GPU JPEG decoding
DATA.RESIZE_GPU         → GPU resize to 224×224
DATA.NORMALIZE_GPU      → GPU normalization
DATA.DALI_BRIDGE_PYTORCH → Zero-copy to PyTorch tensors
```

### Configuration (Planned)

| Property              | Value                                      |
| --------------------- | ------------------------------------------ |
| **Entrypoint Script** | `designC/fast_inference_dali.py` (planned) |
| **Data Pipeline**     | `designC/dali_pipeline.py` (planned)       |
| **Dataset**           | FaceScape (multi-view face meshes)         |

### Expected Benefits

- Eliminate CPU decode bottleneck
- Minimize Host-to-Device copies
- Overlap data loading with inference

---

## Comparison Matrix

| Aspect         | DESIGN.A | DESIGN.A_GPU | DESIGN.B      | DESIGN.C      |
| -------------- | -------- | ------------ | ------------- | ------------- |
| Framework      | TF 1.15  | TF 1.x/2.x   | PyTorch 2.1+  | PyTorch 2.1+  |
| Compute        | CPU      | GPU          | GPU           | GPU           |
| Warmup         | None     | 1 iteration  | 15 iterations | 15 iterations |
| Memory Layout  | Default  | Default      | Contiguous    | Contiguous    |
| Tensor Cores   | N/A      | No           | TF32          | TF32          |
| cuDNN Autotune | N/A      | No           | Yes           | Yes           |
| Data Pipeline  | CPU      | CPU          | CPU           | GPU (DALI)    |
| Metrics        | Separate | Integrated   | Integrated    | Integrated    |

---

## Running Each Design

### DESIGN.A (Docker required)

```bash
cd /home/safa-jsk/Documents/Pixel2MeshPlusPlus
docker run --rm -v $(pwd):/workspace -w /workspace/designA p2mpp:cpu \
    bash run_designA_eval.sh
```

### DESIGN.A_GPU (Docker with GPU)

```bash
cd /home/safa-jsk/Documents/Pixel2MeshPlusPlus/designA_GPU
docker build -t p2mpp:gpu .
docker run --gpus all --rm -v $(pwd)/..:/workspace -w /workspace/designA_GPU p2mpp:gpu \
    python eval_designA_gpu.py
```

### DESIGN.B (Native PyTorch)

```bash
cd /home/safa-jsk/Documents/Pixel2MeshPlusPlus
conda activate p2mesh
python designB/fast_inference_v4_metrics.py \
    --test_file data/designB_eval_full.txt \
    --output_dir outputs/designB/eval_meshes
```

---

## Related Documents

- [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) - Visual pipeline diagrams
- [TRACEABILITY_MATRIX.md](TRACEABILITY_MATRIX.md) - Code-to-stage mapping
- [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) - Timing methodology

---

_Last Updated: February 2026_
