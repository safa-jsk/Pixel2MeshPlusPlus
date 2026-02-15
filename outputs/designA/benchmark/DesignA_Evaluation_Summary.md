# Design A Evaluation Summary

## Overview

| Property                   | Value                                   |
| -------------------------- | --------------------------------------- |
| **Design**                 | Design A (TensorFlow 1.15 CPU Baseline) |
| **Last Evaluation Date**   | February 1, 2026                        |
| **Samples Evaluated**      | 1,000                                   |
| **Current Eval List Size** | 1,000 (updated February 3, 2026)        |

## Pipeline Architecture

```
Input Images (24 views) → VGG16 Features → MVP2M Coarse Mesh → P2MPP Refinement → Final Mesh
     Stage 1 (~0.202s)                                Stage 2 (~3.447s)
```

### Stage 1: MVP2M (Multi-View Pixel2Mesh)

- **Framework**: TensorFlow 1.15
- **Hardware**: CPU-only
- **Model**: `data/p2mpp_models/coarse_mvp2m/models/meshnetmvp2m.ckpt-50`
- **Output**: Coarse mesh (156 vertices, 462 edges, 308 faces)

### Stage 2: P2MPP (Pixel2Mesh++ Refinement)

- **Framework**: TensorFlow 1.15
- **Hardware**: CPU-only
- **Model**: `data/p2mpp_models/refine_p2mpp/models/meshnet.ckpt-10`
- **Output**: Refined mesh (2466 vertices, 7398 edges, 4932 faces)

## Performance Metrics

### Timing Statistics (per sample)

| Metric            | Value           |
| ----------------- | --------------- |
| Average Stage 1   | 0.202s ± 0.029s |
| Average Stage 2   | 3.447s ± 0.118s |
| **Average Total** | **3.648s**      |
| Min Time          | 3.235s          |
| Max Time          | 4.970s          |

### Throughput

| Metric                    | Value                   |
| ------------------------- | ----------------------- |
| Samples per minute        | ~16.4                   |
| Total time (1000 samples) | 3648.37s (60.8 minutes) |
| Stage 1 total             | 201.74s (3.4 minutes)   |
| Stage 2 total             | 3446.63s (57.4 minutes) |

## Quality Metrics

### Overall Performance

| Metric               | Value                   |
| -------------------- | ----------------------- |
| **Chamfer Distance** | 0.00040363 ± 0.00049255 |
| **F1@τ (τ=0.0001)**  | 66.64%                  |
| **F1@2τ**            | 80.45%                  |

### Per-Category Performance

| Category | Name    | CD (×10⁻⁴) | F1@τ   | F1@2τ  | Samples |
| -------- | ------- | ---------- | ------ | ------ | ------- |
| 02691156 | Plane   | 2.25       | 82.36% | 90.24% | 173     |
| 02958343 | Car     | 2.56       | 66.22% | 82.42% | 172     |
| 03001627 | Chair   | 4.02       | 61.26% | 77.25% | 172     |
| 03636649 | Lamp    | 5.67       | 65.13% | 77.40% | 166     |
| 03691459 | Speaker | 5.81       | 54.58% | 72.01% | 168     |
| 04379243 | Table   | 4.02       | 69.85% | 83.03% | 172     |

### Category Rankings

**Best F1@τ Performance:**

1. Plane (82.36%)
2. Table (69.85%)
3. Car (66.22%)

**Best Chamfer Distance:**

1. Plane (0.000225)
2. Car (0.000256)
3. Chair/Table (0.000402)

## Dataset Configuration

### Current Evaluation List (Updated Feb 3, 2026)

| Category  | ID       | Samples  | Percentage |
| --------- | -------- | -------- | ---------- |
| Plane     | 02691156 | 167      | 16.7%      |
| Car       | 02958343 | 167      | 16.7%      |
| Chair     | 03001627 | 167      | 16.7%      |
| Table     | 04379243 | 167      | 16.7%      |
| Speaker   | 03691459 | 166      | 16.6%      |
| Lamp      | 03636649 | 166      | 16.6%      |
| **Total** |          | **1000** | **100%**   |

### Data Paths

- **Ground Truth Meshes**: `data/p2mppdata/test/`
- **Rendered Images**: `data/ShapeNetImages/ShapeNetRendering/`
- **Evaluation List**: `designA/designA_eval_list.txt`

## Output Files

| File                          | Description                     |
| ----------------------------- | ------------------------------- |
| `metrics_results.csv`         | Per-sample quality metrics      |
| `metrics_summary.txt`         | Aggregated metrics summary      |
| `timing_results_detailed.csv` | Per-sample timing data          |
| `combined_timings.txt`        | Stage-wise timing breakdown     |
| `system_info.txt`             | Hardware/software configuration |

## Running Evaluation

```bash
cd /home/safa-jsk/Documents/Pixel2MeshPlusPlus/designA
./run_designA_eval.sh
```

## Notes

1. **CPU Baseline**: This serves as the baseline for comparison with Design B (PyTorch GPU)
2. **Reproducibility**: Random seed 42 used for sample selection
3. **Metrics Threshold**: τ = 0.0001 (standard Pixel2Mesh threshold)
4. **Full Run Complete**: 1000 samples evaluated on February 1, 2026

---

_Generated: February 4, 2026_
