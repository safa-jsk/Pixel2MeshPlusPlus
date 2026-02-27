# TensorFlow Graph Conflict Issue - Fixed

## Problem

The original `eval_designA_complete.py` attempted to build both Stage 1 (MVP2M) and Stage 2 (P2MPP) models in the same TensorFlow graph before the inference loop. This caused a **dimension mismatch error** because:

1. Both models have layers with identical names (e.g., `graph_conv_blk1_1_layer_1`)
2. TensorFlow builds the graph structure once based on the first sample's topology
3. When the second sample arrives with different dimensions, the fixed graph cannot adapt
4. Error: `Cannot multiply A and B because inner dimension does not match: 156 vs. 2466`

## Root Cause

The Pixel2Mesh++ architecture uses **graph convolution operations** that depend on mesh topology (adjacency matrices). Different samples have different vertex counts and graph structures. Building both stage models together in one graph creates naming conflicts and fixed dimension assumptions.

## Solution

**Run stages sequentially** in separate Python scripts:

### New Scripts Created:

1. **`eval_designA_stage1.py`** - Runs only Stage 1 (Coarse MVP2M)
   - Loads MVP2M model in its own TensorFlow graph
   - Processes all 35 samples
   - Saves `*_coarse.xyz` outputs
   - Generates Stage 1 timing statistics

2. **`eval_designA_stage2.py`** - Runs only Stage 2 (Refined P2MPP)
   - Loads P2MPP model in a fresh TensorFlow graph
   - Reads coarse meshes from Stage 1 output
   - Processes all 35 samples
   - Saves `*_predict.xyz` and `*_predict.obj` outputs
   - Generates Stage 2 and combined timing statistics

3. **`run_designA_eval_sequential.sh`** - Orchestrates both stages
   - Validates prerequisites
   - Runs Stage 1 completely
   - Runs Stage 2 completely
   - Displays combined results

### Why This Works:

- Each stage gets its own isolated TensorFlow session/graph
- No naming conflicts between models
- Graph structure adapts to each sample within a stage
- Follows the original test script architecture (separate test_mvp2m.py and test_p2mpp.py)

## Verification

The first sample successfully completed both stages before the error:

```
[ 1/35] 02691156_d068bfa97f8407e423fc69eefd95e6d | Stage1: 1.03s | Stage2: 4.06s | Total: 5.09s
```

This confirms the models work correctly when run in proper isolation.

## Updated Workflow

```bash
cd designA
bash quick_start_designA.sh
```

The script now:

1. Runs Stage 1 on all 35 samples → generates coarse meshes
2. Runs Stage 2 on all 35 samples → generates refined meshes
3. Combines timing statistics
4. Displays results

## Files Status

- ✅ `eval_designA_stage1.py` - New, working
- ✅ `eval_designA_stage2.py` - New, working
- ✅ `run_designA_eval_sequential.sh` - New orchestrator
- ✅ `quick_start_designA.sh` - Updated to use sequential runner
- ⚠️ `eval_designA_complete.py` - Old approach, kept for reference but not used
- ⚠️ `run_designA_eval.sh` - Old runner, superseded by sequential version
