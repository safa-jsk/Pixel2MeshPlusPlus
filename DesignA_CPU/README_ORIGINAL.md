# Design A Baseline Evaluation Guide

## Overview

This directory contains the evaluation infrastructure for **Design A** baseline performance measurements on the Pixel2Mesh++ model (CPU-only).

## Files

- **`eval_designA_complete.py`** - Complete 2-stage inference pipeline (Stage 1: Coarse + Stage 2: Refined)
- **`eval_designA.py`** - Stage 2 only (requires pre-computed coarse meshes)
- **`run_designA_eval.sh`** - Bash script to run evaluation with prerequisites check
- **`data/designA_eval_list.txt`** - 35 sample evaluation list (verified)
- **`data/EVAL_LIST_README.md`** - Documentation of eval list

## Prerequisites

1. **Docker container running** (`p2mpp:cpu`)
2. **Model weights extracted** to:
   - `results/coarse_mvp2m/models/` (Stage 1 checkpoints)
   - `results/refine_p2mpp/models/` (Stage 2 checkpoints)
3. **Data files present**:
   - `data/p2mppdata/test/` (preprocessed .dat files)
   - `data/ShapeNetImages/ShapeNetRendering/` (rendered images)
   - `data/iccv_p2mpp.dat` (mesh template)
   - `data/face3.obj` (face topology)

## Running the Evaluation

### Inside Docker Container:

```bash
# Start Docker container
docker run --rm -it \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD":/workspace \
  -w /workspace \
  p2mpp:cpu

# Run evaluation
bash run_designA_eval.sh
```

### Alternative - Direct Python:

```bash
python eval_designA_complete.py \
    --eval_list data/designA_eval_list.txt \
    --output_dir outputs/designA/eval_meshes
```

## Expected Output

After successful completion:

```
outputs/designA/
├── eval_meshes/
│   ├── 02691156_d004d0e86ea3d77a65a6ed5c458f6a32_00_coarse.xyz
│   ├── 02691156_d004d0e86ea3d77a65a6ed5c458f6a32_00_ground.xyz
│   ├── 02691156_d004d0e86ea3d77a65a6ed5c458f6a32_00_predict.xyz
│   ├── 02691156_d004d0e86ea3d77a65a6ed5c458f6a32_00_predict.obj  ← Final mesh
│   └── ... (35 samples × 4 files = 140 files)
└── benchmark/
    ├── timing_results_detailed.csv  ← Per-sample timing
    └── summary_stats.txt            ← Overall statistics
```

## Output Files Explained

- **`*_ground.xyz`** - Ground truth point cloud (reference)
- **`*_coarse.xyz`** - Stage 1 output (coarse mesh)
- **`*_predict.xyz`** - Stage 2 output (refined mesh, points only)
- **`*_predict.obj`** - Stage 2 output in OBJ format (can open in MeshLab/Blender)

## Performance Metrics (A6)

The evaluation automatically generates:

### `timing_results_detailed.csv`

```csv
sample_id,stage1_time_sec,stage2_time_sec,total_time_sec
02691156_d004d0e86ea3d77a65a6ed5c458f6a32_00.dat,8.234,2.156,10.390
...
```

### `summary_stats.txt`

```
Design A Baseline Performance Summary
==================================================
Number of samples: 35

Timing Statistics (per sample):
--------------------------------------------------
  Average Stage 1: 8.123s ± 0.456s
  Average Stage 2: 2.234s ± 0.123s
  Average Total:   10.357s ± 0.512s
  Min time:        9.456s
  Max time:        11.234s

Total Processing:
--------------------------------------------------
  Total time: 362.50s (6.04 minutes)
==================================================
```

## Next Steps (Design A Roadmap)

- ✅ **A5 Complete**: Batch inference on eval list
- [ ] **A6**: Runtime measurement (automated in script)
- [ ] **A7**: Select 6-12 best meshes for poster
- [ ] **A8**: Write functional verification document

## Troubleshooting

### Issue: Missing ShapeNet images

If you get errors about missing image files:

- Ensure `data/ShapeNetImages/ShapeNetRendering/` exists
- Check that category folders match the eval list

### Issue: Out of memory

For CPU evaluation with limited RAM:

- Reduce eval list size (use first 10-20 samples)
- Close other applications

### Issue: Checkpoint not found

```bash
# Verify model weights are extracted
ls -la results/coarse_mvp2m/models/
ls -la results/refine_p2mpp/models/
```

## Hardware Specifications

Record your system info for the thesis:

```bash
# Inside Docker container
python -c "import tensorflow as tf; print(tf.__version__)"
lscpu | grep "Model name"
free -h
```

Save this to `outputs/designA/benchmark/system_info.txt`
