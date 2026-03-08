# Design A (CPU Baseline): Evaluation Results Report

## Pixel2Mesh++ Sequential 2-Stage Inference — CPU Baseline

**Date:** 2026-03-08  
**Framework:** TensorFlow 2.17.1 (tf.compat.v1 graph mode)  
**Hardware:** CPU-only (CUDA_VISIBLE_DEVICES="")  
**Dataset:** ShapeNet — 1000 samples across 6 categories  
**Threshold τ:** 0.0001

---

## Executive Summary

This report documents the full 1000-sample evaluation of the Design A baseline:
a sequential 2-stage TensorFlow CPU pipeline (MVP2M coarse reconstruction → P2MPP
refinement). Results serve as the **reference baseline** against which GPU-accelerated
designs (Design A GPU, Design B) are compared.

**Key Results:**

- ✅ **1000 samples** evaluated across 6 ShapeNet categories
- ✅ Chamfer Distance: **0.4057 × 10⁻³**
- ✅ F1@τ: **66.50%** · F1@2τ: **80.33%**
- ✅ Total pipeline time: **2129.47s (~35.5 min)** · Average: **2.129s/sample**
- ✅ Best category: **plane** (CD=0.230×10⁻³, F1@τ=81.98%)
- ⚠ Worst category: **speaker** (CD=0.586×10⁻³, F1@τ=54.42%)

---

## 1. Software & Hardware Configuration

### Hardware

| Component | Specification |
|-----------|--------------|
| CPU       | Intel Core i9-14900K (32 logical cores) |
| RAM       | 64 GB DDR5 |
| GPU       | Not used (CPU-only execution) |

### Software Stack

| Component        | Version     | Notes |
|-----------------|-------------|-------|
| Framework       | TensorFlow 2.17.1 | graph mode via tf.compat.v1 |
| Python          | 3.12        | Ubuntu 24.04 system |
| tflearn         | 0.5.0       | patched for TF 2.16+ / Pillow 10+ compat |
| NumPy           | 1.26.x      | |
| Docker Image    | p2mpp:latest | nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 base |
| CUDA            | 12.6.3      | unused (CPU run) |

### Pipeline Architecture

```
Input (3× 224×224 RGB images)
        │
        ▼
  Stage 1: MVP2M (MeshNetMVP2M)
  — Coarse 3D mesh from multi-view images
  — Output: 1248-vertex point cloud
        │
        ▼
  Stage 2: P2MPP (MeshNet)
  — Graph convolution refinement
  — Output: 2562-vertex refined mesh (.obj)
        │
        ▼
  Metrics: Chamfer Distance, F1@τ, F1@2τ
```

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Total samples | 1000 |
| Categories | 6 |
| Source | ShapeNet (Pixel2Mesh++ test split) |
| Image resolution | 224 × 224 px |
| Views per sample | 3 |
| Ground-truth format | Point cloud (`.dat`) |

### Category Distribution

| Category ID | Name    | Samples |
|-------------|---------|---------|
| 02691156    | plane   | 167     |
| 02958343    | car     | 167     |
| 03001627    | chair   | 167     |
| 03636649    | lamp    | 166     |
| 03691459    | speaker | 166     |
| 04379243    | table   | 167     |

---

## 3. Quality Metrics

### Overall Results

| Metric              | Value            |
|---------------------|-----------------|
| Chamfer Distance    | 0.4057 × 10⁻³   |
| CD Std Dev          | ± 0.4936 × 10⁻³ |
| F1@τ (τ=0.0001)     | 66.50%          |
| F1@2τ (τ=0.0002)    | 80.33%          |

### Per-Category Breakdown

| Category ID | Name    | n   | CD (×10⁻³) | F1@τ (%) | F1@2τ (%) |
|-------------|---------|-----|-----------|---------|---------|
| 02691156    | plane   | 167 | 0.2301    | 81.98   | 89.98   |
| 02958343    | car     | 167 | 0.2544    | 66.25   | 82.47   |
| 03001627    | chair   | 167 | 0.3933    | 61.37   | 77.35   |
| 03636649    | lamp    | 166 | 0.5672    | 65.13   | 77.40   |
| 03691459    | speaker | 166 | 0.5855    | 54.42   | 71.84   |
| 04379243    | table   | 167 | 0.4058    | 69.77   | 82.91   |

### Observations

- **Plane and car** have the lowest CD and highest F1, likely due to their relatively
  simple, compact geometry being well-captured by graph-convolution deformation.
- **Speaker and lamp** have the highest CD and lowest F1 — lamp's thin elongated
  structures and speaker's flat box topology are challenging for mesh deformation
  from an ellipsoid initialisation.
- The gap between F1@τ and F1@2τ (~14 pp overall) indicates that a meaningful
  fraction of predicted surface points fall in the 1–2τ range, suggesting
  systematic small-displacement errors rather than catastrophic failures.

---

## 4. Timing Results

### Pipeline Timing Summary

| Stage | Total (s) | Mean (s) | Median (s) | Min (s) | Max (s) | Std Dev (s) |
|-------|-----------|---------|-----------|--------|--------|------------|
| Stage 1 — MVP2M coarse | 67.76 | 0.068 | 0.073 | 0.033 | 0.239 | 0.019 |
| Stage 2 — P2MPP refine | 2061.71 | 2.062 | 2.069 | 1.991 | 2.172 | 0.022 |
| **Combined** | **2129.47** | **2.129** | — | — | — | — |

### Notes

- Stage 1 accounts for only **3.2%** of total wall time; Stage 2 dominates at
  **96.8%**.
- Stage 2 timing is remarkably consistent (σ=0.022s, CV≈1%) indicating
  graph convolution cost is purely input-size driven with no data-dependent
  variation.
- Total wall time including TF session startup and data loading: ~35.5 minutes.
- Timings were accumulated across multiple process runs (segfault resilience);
  each per-sample time was persisted to CSV immediately after inference to ensure
  the total is accurate.

### Throughput

| Metric | Value |
|--------|-------|
| Samples/second (Stage 1) | 14.8 |
| Samples/second (Stage 2) | 0.49 |
| Samples/second (combined) | 0.47 |

---

## 5. Output Artefacts

| Artefact | Path |
|----------|------|
| Predicted meshes (.obj) | `artifacts/outputs/designA/eval_meshes/*_predict.obj` |
| Predicted point clouds (.xyz) | `artifacts/outputs/designA/eval_meshes/*_predict.xyz` |
| Ground-truth point clouds (.xyz) | `artifacts/outputs/designA/eval_meshes/*_ground.xyz` |
| Per-sample metrics CSV | `artifacts/outputs/designA/benchmark/metrics_results.csv` |
| Metrics summary | `artifacts/outputs/designA/benchmark/metrics_summary.txt` |
| Stage 1 timings (summary) | `artifacts/outputs/designA/benchmark/stage1_timings.txt` |
| Stage 2 timings (summary) | `artifacts/outputs/designA/benchmark/stage2_timings.txt` |
| Per-sample timings CSV | `artifacts/outputs/designA/benchmark/stage1_timings_per_sample.csv` |
| Combined timing summary | `artifacts/outputs/designA/benchmark/combined_timings.txt` |

---

## 6. Reproducibility

```bash
# Build image
docker build -t p2mpp:latest .

# Run Design A CPU evaluation
docker rm -f p2mpp-designA-cpu 2>/dev/null || true
docker run -d \
  --name p2mpp-designA-cpu \
  --cpus="8" --memory="12g" \
  -e CUDA_VISIBLE_DEVICES="" \
  -v "$(pwd)":/workspace \
  p2mpp:latest \
  bash -c "
    cp /tmp/tf_ops_prebuilt/*.so /workspace/external/tf_ops/prebuilt/ 2>/dev/null || true
    python3 -c \"
import importlib.util, os
base = list(importlib.util.find_spec('tflearn').submodule_search_locations)[0]
path = os.path.join(base, 'data_utils.py')
src = open(path).read().replace('Image.ANTIALIAS', 'Image.LANCZOS')
open(path, 'w').write(src)
\"
    cd /workspace/DesignA_CPU/scripts
    bash run_designA_eval.sh 2>&1 | tee /workspace/outputs/designA_eval.log
  "

docker logs -f p2mpp-designA-cpu
```

> **Note:** The Pillow `Image.ANTIALIAS` patch is applied at container startup
> because the base image predates the Dockerfile fix. A future image rebuild
> will make this unnecessary.

---

## 7. Known Issues & Limitations

| Issue | Impact | Status |
|-------|--------|--------|
| TF session memory leak causes segfault ~637 samples in | Requires auto-retry loop | Mitigated by resume logic in `run_designA_eval.sh` |
| tflearn 0.5.0 incompatible with TF 2.16+ (`is_sequence`) | Import failure | Patched in Dockerfile |
| tflearn 0.5.0 incompatible with Pillow 10+ (`Image.ANTIALIAS`) | Import failure | Patched at container startup |
| CPU-only; no GPU utilisation | Slow Stage 2 (2.06s/sample) | By design — this is the CPU baseline |

---

## 8. Baseline Reference

These results constitute the **Design A CPU baseline** for this thesis project.
All subsequent designs (Design A GPU, Design B PyTorch) will be compared against
these numbers:

| Reference Metric | Baseline Value |
|-----------------|---------------|
| Chamfer Distance | 0.4057 × 10⁻³ |
| F1@τ            | 66.50%        |
| F1@2τ           | 80.33%        |
| Throughput      | 0.47 samples/s |
| Time per sample | 2.129s        |
