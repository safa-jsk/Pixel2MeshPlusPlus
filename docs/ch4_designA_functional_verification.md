# Design A: Functional Verification Report

## Pixel2Mesh++ Baseline on CPU

**Author:** [Your Name]  
**Date:** January 27, 2026  
**Commit Hash:** `3e9a0c30ede446283b272e4092a52e9e85e73184`  
**Branch:** `Design_A`

---

## 1. Executive Summary

This document provides functional verification of the Pixel2Mesh++ baseline implementation running on CPU-only hardware. The system successfully reconstructs 3D meshes from multi-view 2D images using pretrained ShapeNet models. All 35 evaluation samples from 6 ShapeNet categories were processed without errors, achieving an average inference time of **3.10 seconds per sample** on CPU.

**Key Achievements:**

- ✅ Reproducible Docker-based environment (TensorFlow 1.15, Python 3.6)
- ✅ Successful loading of official pretrained weights
- ✅ 100% completion rate on fixed evaluation subset (35 samples)
- ✅ Quantified baseline performance metrics for CPU execution
- ✅ Generated mesh outputs ready for visualization and quality assessment

---

## 2. Environment Specification

### 2.1 Hardware Configuration

| Component        | Specification                       |
| ---------------- | ----------------------------------- |
| **CPU**          | [To be filled from system_info.txt] |
| **RAM**          | [To be filled from system_info.txt] |
| **Storage**      | SSD/HDD                             |
| **OS**           | Linux (Ubuntu/Debian-based)         |
| **Architecture** | x86_64                              |

### 2.2 Software Stack

| Component        | Version           | Purpose                   |
| ---------------- | ----------------- | ------------------------- |
| **Docker**       | Latest            | Containerized environment |
| **Python**       | 3.6               | Runtime environment       |
| **TensorFlow**   | 1.15 (CPU-only)   | Deep learning framework   |
| **NumPy**        | Latest compatible | Numerical operations      |
| **OpenCV (cv2)** | Latest compatible | Image processing          |
| **NetworkX**     | Latest compatible | Graph operations          |
| **SciPy**        | Latest compatible | Scientific computing      |

**Container Image:** `p2mpp:cpu`

**Environment Creation:**

```bash
docker build -f Dockerfile.cpu -t p2mpp:cpu .
docker run --rm -it -v "$PWD":/workspace -w /workspace p2mpp:cpu
```

### 2.3 Repository Information

- **Repository:** `walsvid/Pixel2MeshPlusPlus`
- **Fork Branch:** `Design_A`
- **Commit:** `3e9a0c30ede446283b272e4092a52e9e85e73184`
- **Original Paper:** "Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation" (ICCV 2019)

---

## 3. Model Architecture & Pretrained Weights

### 3.1 Network Architecture

**Two-Stage Pipeline:**

1. **Stage 1: Coarse MVP2M (Multi-View Pixel2Mesh)**
   - Input: Multi-view RGB images (3 views, 224×224 each)
   - Backbone: VGG-16 CNN for perceptual feature extraction
   - Graph Network: GCN-based mesh deformation
   - Template: Ellipsoid initialization (156 → 628 → 2466 vertices)
   - Output: Coarse 3D mesh

2. **Stage 2: Refined P2MPP (Pixel2Mesh++)**
   - Input: Coarse mesh + multi-view images
   - Refinement: Additional GCN layers with image-guided deformation
   - Output: High-resolution 3D mesh (2466 vertices, detailed geometry)

### 3.2 Pretrained Weights

| Model Component     | Checkpoint Location                                | Training Data       |
| ------------------- | -------------------------------------------------- | ------------------- |
| **Stage 1 (MVP2M)** | `results/coarse_mvp2m/models/meshnetmvp2m.ckpt-50` | ShapeNet (epoch 50) |
| **Stage 2 (P2MPP)** | `results/refine_p2mpp/models/meshnet.ckpt-10`      | ShapeNet (epoch 10) |

**Weight Source:** Official pretrained models from authors (Google Drive)  
**Verification:** Models loaded successfully, no checkpoint errors

---

## 4. Evaluation Dataset

### 4.1 Subset Definition

**Evaluation List:** `designA/designA_eval_list.txt`  
**Total Samples:** 35  
**Data Source:** ShapeNet preprocessed test set

**Category Distribution:**

| Category ID | Category Name | Sample Count |
| ----------- | ------------- | ------------ |
| 02691156    | Airplane      | 8            |
| 02958343    | Car           | 6            |
| 03001627    | Chair         | 8            |
| 04379243    | Table         | 6            |
| 03636649    | Loudspeaker   | 4            |
| 03691459    | Lamp          | 3            |

**Data Structure:**

- **Preprocessed features:** `data/p2mppdata/test/*.dat` (labels, camera poses)
- **Multi-view images:** `data/ShapeNetImages/ShapeNetRendering/{category}/{model}/rendering/*.png`
- **Views per sample:** 3 (indices: 0, 6, 7)
- **Image resolution:** 224×224 RGB

### 4.2 Selection Rationale

The 35-sample subset represents diverse object categories from ShapeNet, ensuring coverage of different geometric complexities:

- **Symmetric objects:** Airplanes, lamps
- **Complex topology:** Chairs, tables with legs
- **Smooth surfaces:** Cars
- **Fine details:** Loudspeakers with grilles

This fixed subset enables reproducible comparison between Design A (CPU baseline) and Design B (GPU optimization).

---

## 5. Pipeline Execution & Results

### 5.1 Inference Pipeline

```
Input Images (3 views)
    ↓
[Stage 1: Coarse MVP2M]
    ├─ VGG-16 Feature Extraction
    ├─ Graph Convolution (GCN)
    └─ Mesh Deformation (ellipsoid → coarse mesh)
    ↓
Coarse Mesh (2466 vertices)
    ↓
[Stage 2: Refined P2MPP]
    ├─ Perceptual Feature Projection
    ├─ Graph-based Refinement
    └─ Final Mesh Output
    ↓
Refined Mesh (.xyz, .obj)
```

### 5.2 Execution Commands

**Single Demo Run:**

```bash
python demo.py
# Output: data/demo/predict.obj
```

**Batch Evaluation (35 samples):**

```bash
cd designA
bash quick_start_designA.sh
# Runs Stage 1, then Stage 2 sequentially
```

**Individual Stage Execution:**

```bash
# Stage 1 only
python eval_designA_stage1.py \
    --eval-list designA_eval_list.txt \
    --output-dir ../outputs/designA/eval_meshes

# Stage 2 only
python eval_designA_stage2.py \
    --eval-list designA_eval_list.txt \
    --coarse-dir ../outputs/designA/eval_meshes \
    --output-dir ../outputs/designA/eval_meshes
```

### 5.3 Output Files

**Generated Artifacts (per sample):**

- `{sample}_ground.xyz` - Ground truth point cloud
- `{sample}_predict.xyz` - Predicted point cloud (2466 points)
- `{sample}_predict.obj` - Mesh with faces (for visualization)

**Total Files Generated:** 105 (35 samples × 3 files each)

**Example Outputs:**

```
outputs/designA/eval_meshes/
├── 02691156_d068bfa97f8407e423fc69eefd95e6d3_00_ground.xyz
├── 02691156_d068bfa97f8407e423fc69eefd95e6d3_00_predict.xyz
├── 02691156_d068bfa97f8407e423fc69eefd95e6d3_00_predict.obj
├── 02958343_cbeb8998de880db684479d4c559a58d_00_ground.xyz
├── 02958343_cbeb8998de880db684479d4c559a58d_00_predict.xyz
├── 02958343_cbeb8998de880db684479d4c559a58d_00_predict.obj
└── ... (99 more files)
```

---

## 6. Performance Benchmarks (CPU Baseline)

### 6.1 Timing Results

**Combined 2-Stage Pipeline:**

| Metric                  | Value     |
| ----------------------- | --------- |
| **Total Samples**       | 35        |
| **Stage 1 Total Time**  | 6.56s     |
| **Stage 2 Total Time**  | 101.95s   |
| **Combined Total Time** | 108.51s   |
| **Average per Sample**  | **3.10s** |

**Stage 1 (Coarse MVP2M) Breakdown:**

| Metric           | Value  |
| ---------------- | ------ |
| **Total Time**   | 6.56s  |
| **Average Time** | 0.187s |
| **Median Time**  | 0.167s |
| **Min Time**     | 0.151s |
| **Max Time**     | 0.777s |
| **Std Dev**      | 0.102s |

**Stage 2 (Refined P2MPP) Breakdown:**

| Metric           | Value   |
| ---------------- | ------- |
| **Total Time**   | 101.95s |
| **Average Time** | 2.913s  |
| **Median Time**  | 2.884s  |
| **Min Time**     | 2.846s  |
| **Max Time**     | 3.404s  |
| **Std Dev**      | 0.091s  |

### 6.2 Performance Analysis

**Key Observations:**

1. **Stage 2 Dominates Runtime:** 94% of total time (101.95s / 108.51s)
   - Graph convolution operations on 2466-vertex mesh
   - Perceptual feature projection from multi-view images
   - CPU-bound tensor operations

2. **Stage 1 is Fast:** 6% of total time (6.56s / 108.51s)
   - Smaller mesh (156 → 628 vertices initially)
   - Primarily CNN forward pass (highly optimized)

3. **Low Variance:** Consistent performance across samples
   - Stage 1: σ = 0.102s (relative 54%)
   - Stage 2: σ = 0.091s (relative 3.1%)
   - Indicates stable inference pipeline

4. **First Sample Overhead:** Stage 1 max time (0.777s) likely includes:
   - Model initialization
   - TensorFlow graph compilation
   - Memory allocation

### 6.3 Comparison to GPU Expectations

Based on typical CNN/GCN acceleration factors:

| Metric              | CPU Baseline (Actual) | GPU Expected | Speedup Factor |
| ------------------- | --------------------- | ------------ | -------------- |
| **Avg Time/Sample** | 3.10s                 | ~0.3-0.5s    | **6-10×**      |
| **Stage 1**         | 0.187s                | ~0.05s       | 3-4×           |
| **Stage 2**         | 2.913s                | ~0.25-0.45s  | **6-12×**      |

_GPU estimates based on typical TensorFlow 1.x CNN+GCN speedups. Design B will provide actual measurements._

---

## 7. Verification Evidence

### 7.1 Execution Logs

**Stage 1 Completion:**

```
======================================================================
Stage 1 Complete!
Coarse meshes saved to: ../outputs/designA/eval_meshes
  (saved as *_predict.xyz for Stage 2 compatibility)
Timing stats: 6.56s total, 0.187s avg
======================================================================
```

**Stage 2 Completion:**

```
======================================================================
Stage 2 Complete!
Refined meshes saved to: ../outputs/designA/eval_meshes
Timing stats: 101.95s total, 2.913s avg
======================================================================
```

**Success Rate:** 100% (35/35 samples completed without errors)

### 7.2 File Integrity Check

```bash
$ ls outputs/designA/eval_meshes/*.obj | wc -l
35

$ ls outputs/designA/eval_meshes/*.xyz | wc -l
70  # 35 ground truth + 35 predictions
```

**All expected files generated successfully.**

### 7.3 Mesh Format Verification

Example `.obj` file structure:

```
v -0.123456 0.234567 0.345678
v -0.223456 0.334567 0.445678
...
f 1 2 3
f 2 3 4
...
```

- **Total vertices:** 2466 per mesh
- **Total faces:** [Varies by template, typically ~5000]
- **Format:** Standard Wavefront OBJ (compatible with MeshLab, Blender)

---

## 8. Known Issues & Limitations

### 8.1 Technical Issues Encountered & Resolved

**Issue 1: TensorFlow Graph Conflict**

- **Problem:** Initial attempt to build both Stage 1 and Stage 2 models in same session caused dimension mismatch errors
- **Root Cause:** Graph convolution layers with identical names in both models
- **Solution:** Split into separate Python scripts running in isolated TensorFlow sessions

**Issue 2: DataFetcher Mesh Naming**

- **Problem:** Stage 2 expected `*_predict.xyz` but Stage 1 saved `*_coarse.xyz`
- **Root Cause:** Inconsistent naming convention between stages
- **Solution:** Modified Stage 1 to save as `*_predict.xyz` for compatibility

**Issue 3: Feature Projection Out-of-Bounds**

- **Problem:** Sample 7 failed with `indices[27464] = [1, 112, 69] does not index into param shape [3,112,112,32]`
- **Root Cause:** Bilinear sampling didn't clamp coordinates after scaling
- **Solution:** Added bounds clamping in `modules/layers.py` for all feature map resolutions

### 8.2 Current Limitations

1. **CPU-Only Performance:** ~3.1s per sample is too slow for real-time applications
2. **No Custom CUDA Ops:** NNDistance falls back to CPU implementation (acceptable for baseline)
3. **Fixed Template Topology:** All outputs have 2466 vertices regardless of object complexity
4. **Multi-View Requirement:** Needs exactly 3 pre-rendered views per object

### 8.3 Design B Optimization Targets

Based on profiling, Stage 2 refinement (94% of runtime) is the primary candidate for GPU acceleration:

- Graph convolution operations
- Perceptual feature projection
- Sparse tensor operations

---

## 9. Reproducibility Instructions

### 9.1 Prerequisites

1. Docker installed
2. ~12GB disk space (ShapeNetRendering dataset)
3. ~4GB RAM minimum

### 9.2 Step-by-Step Reproduction

```bash
# 1. Clone repository
git clone https://github.com/walsvid/Pixel2MeshPlusPlus.git
cd Pixel2MeshPlusPlus
git checkout Design_A

# 2. Build Docker image
docker build -f Dockerfile.cpu -t p2mpp:cpu .

# 3. Download ShapeNetRendering (if not present)
cd data
wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
tar -xzf ShapeNetRendering.tgz
cd ..

# 4. Start container
docker run --rm -it -v "$PWD":/workspace -w /workspace p2mpp:cpu

# 5. Inside container: Run evaluation
cd designA
bash quick_start_designA.sh
```

**Expected Runtime:** ~2-3 minutes (including model loading)

### 9.3 Verification Commands

```bash
# Check output files
ls -lh outputs/designA/eval_meshes/*.obj | wc -l  # Should be 35

# View timing stats
cat outputs/designA/benchmark/combined_timings.txt

# Verify first mesh
head outputs/designA/eval_meshes/02691156_d068bfa97f8407e423fc69eefd95e6d3_00_predict.obj
```

---

## 10. Conclusion

Design A successfully establishes a **reproducible baseline** for Pixel2Mesh++ inference on CPU-only hardware. All functional requirements are met:

✅ **Environment:** Containerized, version-locked  
✅ **Model:** Official pretrained weights loaded correctly  
✅ **Execution:** 100% success rate on 35-sample evaluation set  
✅ **Performance:** Quantified baseline (3.10s/sample average)  
✅ **Outputs:** 35 meshes in standard .obj format

**CPU Baseline Established:** 3.10 seconds per sample

This baseline provides a **reference point** for Design B GPU optimization, where we expect **6-10× speedup** to achieve near real-time performance (~0.3-0.5s per sample).

---

## Appendices

### Appendix A: Evaluation Sample List

See `designA/designA_eval_list.txt` for complete list of 35 samples.

**Sample Format:** `{categoryID}_{modelID}_{viewID}.dat`

Example:

```
02691156_d068bfa97f8407e423fc69eefd95e6d3_00.dat
02958343_cbeb8998de880db684479d4c559a58d_00.dat
03001627_cbc9014bb6ce3d902ff834514c92e8f_00.dat
```

### Appendix B: Directory Structure

```
Pixel2MeshPlusPlus/
├── designA/                          # Design A scripts
│   ├── designA_eval_list.txt         # 35-sample evaluation list
│   ├── eval_designA_stage1.py        # Stage 1 inference
│   ├── eval_designA_stage2.py        # Stage 2 inference
│   ├── run_designA_eval.sh           # Orchestration script
│   └── quick_start_designA.sh        # User-friendly wrapper
├── outputs/designA/
│   ├── eval_meshes/                  # 105 output files (35×3)
│   └── benchmark/                    # Timing statistics
│       ├── stage1_timings.txt
│       ├── stage2_timings.txt
│       └── combined_timings.txt
├── data/
│   ├── p2mppdata/test/               # Preprocessed .dat files
│   └── ShapeNetImages/
│       └── ShapeNetRendering/        # Multi-view rendered images
└── results/
    ├── coarse_mvp2m/models/          # Stage 1 weights
    └── refine_p2mpp/models/          # Stage 2 weights
```

### Appendix C: Software Dependencies

**Core Libraries:**

- TensorFlow 1.15.0 (CPU)
- Python 3.6.x
- NumPy
- OpenCV (cv2)
- NetworkX
- SciPy
- tflearn

**Custom Modules:**

- `modules/models_mvp2m.py` - Stage 1 architecture
- `modules/models_p2mpp.py` - Stage 2 architecture
- `modules/layers.py` - Graph convolution layers
- `utils/dataloader.py` - Multi-threaded data loading

### Appendix D: References

1. Wen, C., Zhang, Y., Li, Z., & Fu, Y. (2019). Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation. _ICCV 2019_.
2. Original Repository: https://github.com/walsvid/Pixel2MeshPlusPlus
3. ShapeNet Dataset: https://shapenet.org/

---

**Document Version:** 1.0  
**Last Updated:** January 27, 2026  
**Status:** Design A Complete ✅
