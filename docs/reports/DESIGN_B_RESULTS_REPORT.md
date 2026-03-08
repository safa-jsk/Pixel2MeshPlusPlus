# Design B: Evaluation Results Report

## Pixel2Mesh++ PyTorch-Native 2-Stage Inference — GPU Accelerated

**Date:** 2026-03-08  
**Framework:** PyTorch 2.6.0+cu126  
**Hardware:** NVIDIA GeForce RTX 4090 (CUDA 12.6.3)  
**Dataset:** ShapeNet — 1000 samples across 6 categories  
**Threshold τ:** 0.0001

---

## Executive Summary

This report documents the full 1000-sample evaluation of Design B: a complete
PyTorch port of the Pixel2Mesh++ pipeline. Both stages (MVP2M coarse
reconstruction and MeshNet refinement) are re-implemented natively in PyTorch
with AMP autocast, operating on checkpoints converted from the original
TensorFlow `.npz` weight files.

**Key Results:**

- ✅ **1000 samples** evaluated across 6 ShapeNet categories
- ✅ Total inference time: **26.94s** · Mean: **27.5ms/sample**
- ✅ Throughput: **36.3 samples/sec** — **~1.81× faster** than Design A GPU (49.81ms/sample)
- ✅ **Chamfer Distance: 0.4281 × 10⁻³** (valid, correctly computed via PyTorch Chamfer)
- ✅ **F1@τ: 67.31%** · **F1@2τ: 77.72%**
- ✅ Best category (F1@τ): **plane** (79.69%)
- ⚠ Worst category (F1@τ): **speaker** (55.80%)

---

## 1. Software & Hardware Configuration

### Hardware

| Component | Specification |
|-----------|--------------|
| CPU       | Intel Core i9-14900K (32 logical cores) |
| RAM       | 64 GB DDR5 |
| GPU       | NVIDIA GeForce RTX 4090 |
| CUDA      | 12.6.3 |

### Software Stack

| Component        | Version      | Notes |
|-----------------|--------------|-------|
| Framework       | PyTorch 2.6.0+cu126 | Native CUDA support |
| Python          | 3.12         | Ubuntu 24.04 system |
| AMP Autocast    | Enabled      | `torch.cuda.amp.autocast` |
| torch.compile   | Disabled     | Not used in this run |
| Chamfer Backend | PyTorch      | Pure PyTorch fallback (CUDA ext not loaded) |
| Docker Image    | p2mpp:latest | nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 base |

### Pipeline Architecture

```
Input (3× 224×224 RGB images)
        │
        ▼
  Stage 1: MVP2MNet (PyTorch)       ← GPU + AMP autocast
  — Checkpoints: artifacts/checkpoints/torch/mvp2m_converted.npz
  — Coarse 3D mesh from multi-view images
  — Output: 1248-vertex point cloud
        │
        ▼
  Stage 2: MeshNetPyTorch (exact)   ← GPU + AMP autocast
  — Checkpoints: artifacts/checkpoints/torch/meshnet_converted.npz
  — Graph convolution refinement
  — Output: 2562-vertex refined mesh (.obj + .xyz)
        │
        ▼
  Metrics: Chamfer Distance, F1@τ, F1@2τ (PyTorch Chamfer)
```

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Total samples | 1000 |
| Eval list | `DesignB/designB_eval_list.txt` |
| Image source | `data/ShapeNetRendering/<category>/<item>/rendering/` |
| Ground truth | `data/p2mppdata/test/<category>/<item>.dat` |

### Category Distribution

| ShapeNet ID | Category | Samples |
|-------------|----------|---------|
| 02691156 | airplane / plane | 167 |
| 02958343 | car | 167 |
| 03001627 | chair | 167 |
| 03636649 | lamp | 166 |
| 03691459 | speaker | 166 |
| 04379243 | table | 167 |
| **Total** | | **1000** |

---

## 3. Evaluation Configuration

| Parameter | Value |
|-----------|-------|
| Stage 1 checkpoint | `artifacts/checkpoints/torch/mvp2m_converted.npz` |
| Stage 2 checkpoint | `artifacts/checkpoints/torch/meshnet_converted.npz` |
| Mesh data template | `assets/data_templates/iccv_p2mpp.dat` |
| GPU warmup iterations | 15 |
| AMP autocast | Enabled |
| tau threshold | 0.0001 |

---

## 4. Timing Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Total samples | 1000 |
| Samples timed | 978 (rest resumed from disk) |
| Total inference time | 26.94s |
| Mean time/sample | 27.5ms |
| Std dev | 4.2ms |
| Min | 25.9ms |
| Max | 157.6ms (first inference, warmup effect) |
| Throughput | **36.3 samples/sec** |

### Speed Comparison vs. Other Designs

| Design | Framework | Mean Time | Throughput | Speedup vs. CPU |
|--------|-----------|-----------|------------|-----------------|
| A (CPU) | TensorFlow (CPU) | ~2,129ms/sample | 0.47 sa/s | 1× (baseline) |
| A (GPU) | TensorFlow (GPU) | 49.81ms/sample | 20.08 sa/s | 42.7× |
| **B (GPU)** | **PyTorch (GPU)** | **27.5ms/sample** | **36.3 sa/s** | **~77× vs CPU, 1.81× vs A-GPU** |

---

## 5. Quality Metrics

### Overall

| Metric | Value |
|--------|-------|
| Chamfer Distance (CD) | **0.4281 × 10⁻³** |
| CD std dev | 0.5784 × 10⁻³ |
| F1@τ (τ=0.0001) | **67.31%** |
| F1@2τ | **77.72%** |

### Per-Category Results

| ShapeNet ID | Category | Count | CD (×10⁻³) | F1@τ (%) | F1@2τ (%) |
|-------------|----------|-------|------------|----------|-----------|
| 02691156 | plane | 167 | 0.2476 | **79.69** | 87.15 |
| 02958343 | car | 167 | 0.2152 | 71.61 | 82.44 |
| 03001627 | chair | 167 | 0.4109 | 64.02 | 74.86 |
| 03636649 | lamp | 166 | 0.6952 | 62.88 | 71.48 |
| 03691459 | speaker | 166 | 0.5901 | 55.80 | 70.06 |
| 04379243 | table | 167 | 0.4118 | 69.80 | 80.28 |
| **Mean** | | **1000** | **0.4281** | **67.31** | **77.72** |

---

## 6. Metric Comparison: Design A CPU vs. Design A GPU vs. Design B

| Design | CD (×10⁻³) | F1@τ (%) | F1@2τ (%) | Notes |
|--------|------------|----------|-----------|-------|
| A (CPU) | 0.4057 | 66.50 | 80.33 | TF CPU, valid CD |
| A (GPU) | N/A (negative) | 56.67 | 73.27 | TF GPU, CD invalid — squared vs. raw distances |
| **B (GPU)** | **0.4281** | **67.31** | **77.72** | PyTorch, valid CD |

**Key observations:**
1. Design B (PyTorch) achieves comparable quality to Design A CPU with **~77× speedup** on CPU timing.
2. Design B F1@τ (67.31%) slightly exceeds Design A CPU F1@τ (66.50%), within noise range.
3. Design B F1@2τ (77.72%) is slightly lower than Design A CPU (80.33%), possibly due to checkpoint conversion precision losses.
4. Design B CD (0.4281×10⁻³) is slightly higher than Design A CPU (0.4057×10⁻³), also consistent with minor checkpoint conversion differences.
5. The PyTorch implementation does **not** exhibit the negative CD bug present in Design A GPU (TF).

---

## 7. Output Files

| Location | Contents |
|----------|----------|
| `artifacts/outputs/designB/eval_meshes/` | 1000 × `_predict.obj`, `_predict.xyz`, `_ground.xyz` |
| `artifacts/outputs/designB/benchmark/metrics_results.csv` | Per-sample CD, F1@τ, F1@2τ, timing |
| `artifacts/outputs/designB/benchmark/metrics_summary.txt` | Overall + category summary text |
| `artifacts/outputs/designB/benchmark/combined_timings.txt` | Timing statistics |

---

## 8. Resume/Fault Tolerance

The evaluation engine (`fast_inference_v4_metrics.py`) supports seamless resume:
- Each sample writes `_predict.xyz` and `_ground.xyz` before proceeding
- On restart, the engine detects existing files, reloads from disk, recomputes metrics, and skips inference
- The run script (`DesignB/scripts/run_designB_eval.sh`) retries up to 10 times on crash

---

## 9. Conclusion

Design B demonstrates that a clean PyTorch port of Pixel2Mesh++ successfully
replicates the quality of the original TensorFlow implementation while delivering
significant performance gains. The ~1.81× speedup over Design A GPU (despite both
running on the same RTX 4090) reflects PyTorch's more efficient GPU utilization
and AMP autocast overhead reduction compared to TF's legacy graph-mode execution.

Chamfer Distance and F1 scores are all valid (no metric computation bugs), making
Design B the recommended baseline for future work.
