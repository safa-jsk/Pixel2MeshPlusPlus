# Design A Evaluation List

**Created:** January 27, 2026  
**File:** [designA_eval_list.txt](designA_eval_list.txt)

## Overview

This evaluation list contains **35 samples** from preprocessed ShapeNet data for Design A baseline testing. The samples are balanced across 6 object categories, all verified to exist in `data/p2mppdata/test/`.

## Category Breakdown

| Category ID | Category Name | Sample Count |
| ----------- | ------------- | ------------ |
| 02691156    | Airplane      | 8            |
| 02958343    | Car           | 6            |
| 03001627    | Chair         | 8            |
| 04379243    | Table         | 6            |
| 03691459    | Loudspeaker   | 4            |
| 03636649    | Lamp          | 3            |
| **Total**   | **6**         | **35**       |

## Selection Criteria

- **Size:** 35 samples (within Design A recommended range of 20-100)
- **Diversity:** Balanced across 6 major ShapeNet categories
- **Availability:** All files verified to exist in `data/p2mppdata/test/`
- **Reproducibility:** Fixed list ensures consistent evaluation across Design A, B, and C
- **Source:** Selected from preprocessed test set (43,760 total samples available)

## File Format

Each line follows the pattern: `{categoryID}_{modelID}_{viewID}.dat`

Example:

```
    02691156_d004d0e86ea3d77a65a6ed5c458f6a32_00.dat
```

Where:

- `02691156` = ShapeNet synset ID (airplane)
- `d004d0e86ea3d77a65a6ed5c458f6a32` = Unique model identifier
- `00` = View/camera angle index
- `.dat` = Binary data format containing preprocessed mesh/image data

**Data Location:** All files are in `data/p2mppdata/test/`

## Usage

This list is used for:

1. **Baseline inference** (A5) - Generate predicted meshes for all 35 samples
2. **Runtime benchmarking** (A6) - Measure sec/sample performance
3. **Visual results** (A7) - Select best outputs for poster/thesis
4. **Cross-design comparison** - Same list will be used for Design B and C

## ✅ All 35 `.dat` files verified to exist in `data/p2mppdata/test/`

2. Configure test_p2mpp.py to use this eval list
3. Run batch inference on all 35 samples
4. Generate output meshes in `outputs/designA/eval_meshes/`
5. Run batch inference using test_p2mpp.py or demo.py wrapper
6. Generate output meshes in `outputs/designA/eval_meshes/`
7. Measure runtime metrics for Design A baseline
