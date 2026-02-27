# Traceability Matrix - Pixel2Mesh++

This document maps thesis methodology stages (DESIGN._, CAMFM._, DATA.\*) to specific code locations, implementations, and evidence artifacts.

---

## Stage ID Legend

| Prefix         | Category       | Description                             |
| -------------- | -------------- | --------------------------------------- |
| `DESIGN.A`     | Design A       | TensorFlow CPU baseline                 |
| `DESIGN.A_GPU` | Design A GPU   | TensorFlow with simple GPU enablement   |
| `DESIGN.B`     | Design B       | PyTorch GPU with CAMFM optimizations    |
| `DESIGN.C`     | Design C       | Design B + DALI data pipeline (planned) |
| `CAMFM.A2a`    | GPU Residency  | All tensors on GPU, no CPU fallbacks    |
| `CAMFM.A2b`    | Steady State   | Warmup + timing boundaries + autotune   |
| `CAMFM.A2c`    | Memory Layout  | Pre-allocation + contiguous tensors     |
| `CAMFM.A2d`    | Optional Accel | AMP/torch.compile if stable             |
| `CAMFM.A3`     | Metrics        | Quality + performance metrics export    |
| `CAMFM.A5`     | Method         | Evidence bundle + repeatable steps      |
| `DATA.*`       | Data Pipeline  | GPU-native data loading (Design C)      |

---

## Traceability Matrix

### DESIGN.A - TensorFlow CPU Baseline

| StageID  | File Path                        | Function/Class     | What It Does                    | Evidence Artifact                               |
| -------- | -------------------------------- | ------------------ | ------------------------------- | ----------------------------------------------- |
| DESIGN.A | `designA/run_designA_eval.sh`    | main script        | Orchestrates 3-stage evaluation | `outputs/designA/benchmark/`                    |
| DESIGN.A | `designA/eval_designA_stage1.py` | `main()`           | Stage 1 coarse mesh inference   | `outputs/designA/benchmark/stage1_timings.txt`  |
| DESIGN.A | `designA/eval_designA_stage2.py` | `main()`           | Stage 2 refined mesh inference  | `outputs/designA/benchmark/stage2_timings.txt`  |
| DESIGN.A | `designA/compute_metrics.py`     | `main()`           | Compute CD, F1@τ, F1@2τ         | `outputs/designA/benchmark/metrics_results.csv` |
| DESIGN.A | `modules/models_mvp2m.py`        | `MeshNetMVP2M`     | Stage 1 model definition        | N/A                                             |
| DESIGN.A | `modules/models_p2mpp.py`        | `MeshNet`          | Stage 2 model definition        | N/A                                             |
| DESIGN.A | `modules/chamfer.py`             | `nn_distance()`    | Chamfer distance computation    | N/A                                             |
| DESIGN.A | `modules/layers.py`              | `GraphConvolution` | GCN layer implementation        | N/A                                             |
| DESIGN.A | `utils/dataloader.py`            | `DataFetcher`      | CPU data loading thread         | N/A                                             |

### DESIGN.A_GPU - TensorFlow GPU

| StageID      | File Path                                 | Function/Class | What It Does                                       | Speedup Change                        | Evidence Artifact                                        |
| ------------ | ----------------------------------------- | -------------- | -------------------------------------------------- | ------------------------------------- | -------------------------------------------------------- |
| DESIGN.A_GPU | `designA_GPU/eval_designA_gpu.py`         | `main()`       | GPU-enabled inference                              | Enable GPU via `CUDA_VISIBLE_DEVICES` | `outputs/designA_GPU/eval_meshes/evaluation_summary.txt` |
| DESIGN.A_GPU | `designA_GPU/eval_designA_gpu.py:30`      | GPU config     | `os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)` | CPU → GPU execution                   | `outputs/designA_GPU/eval_meshes/timing_results.csv`     |
| DESIGN.A_GPU | `designA_GPU/eval_designA_gpu.py:120-123` | SessionConfig  | `allow_growth=True, allow_soft_placement=True`     | Memory efficiency                     | N/A                                                      |
| DESIGN.A_GPU | `designA_GPU/eval_designA_gpu.py:145-153` | Warmup         | Single warmup iteration                            | Reduce cold-start variance            | N/A                                                      |

### DESIGN.B - PyTorch GPU Optimized

| StageID  | File Path                                 | Function/Class       | What It Does            | Speedup Change          | Evidence Artifact                               |
| -------- | ----------------------------------------- | -------------------- | ----------------------- | ----------------------- | ----------------------------------------------- |
| DESIGN.B | `designB/fast_inference_v4.py`            | `main()`             | Main entrypoint         | Full CAMFM pipeline     | `outputs/designB/eval_meshes/`                  |
| DESIGN.B | `designB/fast_inference_v4_metrics.py`    | `main()`             | Entrypoint with metrics | Integrated metrics      | `outputs/designB/benchmark/metrics_results.csv` |
| DESIGN.B | `designB/modules/models_mvp2m_pytorch.py` | `MVP2MNet`           | Stage 1 PyTorch model   | TF → PyTorch conversion | N/A                                             |
| DESIGN.B | `designB/modules/models_p2mpp_exact.py`   | `MeshNetPyTorch`     | Stage 2 PyTorch model   | TF → PyTorch conversion | N/A                                             |
| DESIGN.B | `designB/modules/chamfer_pytorch.py`      | `chamfer_distance()` | PyTorch Chamfer         | GPU-native metrics      | N/A                                             |

### CAMFM.A2a_GPU_RESIDENCY

| StageID   | File Path                                 | Line  | Code Pattern                    | What It Does           | Impact                 |
| --------- | ----------------------------------------- | ----- | ------------------------------- | ---------------------- | ---------------------- |
| CAMFM.A2a | `designB/fast_inference_v4.py`            | 46-47 | `.to(self.device).contiguous()` | Initial coord on GPU   | Eliminate H2D transfer |
| CAMFM.A2a | `designB/fast_inference_v4.py`            | 48-50 | `_sparse_to_torch()`            | Sparse supports on GPU | GPU-resident adjacency |
| CAMFM.A2a | `designB/fast_inference_v4.py`            | 51-52 | `pool_idx.to(device)`           | Pool indices on GPU    | GPU-resident pooling   |
| CAMFM.A2a | `designB/fast_inference_v4.py`            | 54-56 | `sample_coord.to(device)`       | Sample coords on GPU   | GPU-resident sampling  |
| CAMFM.A2a | `designB/fast_inference_v4.py`            | 99    | `.to(self.device).coalesce()`   | Sparse coalescing      | Efficient sparse ops   |
| CAMFM.A2a | `designB/modules/models_mvp2m_pytorch.py` | N/A   | All `nn.Module`                 | Model on GPU           | No CPU fallback        |

### CAMFM.A2b_STEADY_STATE

| StageID   | File Path                      | Line    | Code Pattern                                   | What It Does        | Impact                |
| --------- | ------------------------------ | ------- | ---------------------------------------------- | ------------------- | --------------------- |
| CAMFM.A2b | `designB/fast_inference_v4.py` | 24      | `torch.backends.cudnn.benchmark = True`        | cuDNN autotune      | Faster convolutions   |
| CAMFM.A2b | `designB/fast_inference_v4.py` | 25      | `torch.backends.cuda.matmul.allow_tf32 = True` | TF32 matmul         | Faster matrix ops     |
| CAMFM.A2b | `designB/fast_inference_v4.py` | 26      | `torch.backends.cudnn.allow_tf32 = True`       | TF32 cuDNN          | Faster convolutions   |
| CAMFM.A2b | `designB/fast_inference_v4.py` | 101-113 | `_warmup()`                                    | 15-iteration warmup | cuDNN plan caching    |
| CAMFM.A2b | `designB/fast_inference_v4.py` | 298     | `torch.cuda.synchronize()`                     | Pre-inference sync  | Accurate timing start |
| CAMFM.A2b | `designB/fast_inference_v4.py` | 301     | `torch.cuda.synchronize()`                     | Post-inference sync | Accurate timing end   |

### CAMFM.A2c_MEM_LAYOUT

| StageID   | File Path                      | Line  | Code Pattern              | What It Does             | Impact                |
| --------- | ------------------------------ | ----- | ------------------------- | ------------------------ | --------------------- |
| CAMFM.A2c | `designB/fast_inference_v4.py` | 46    | `.contiguous()`           | Contiguous initial_coord | Cache-friendly access |
| CAMFM.A2c | `designB/fast_inference_v4.py` | 54-56 | `sample_adj.contiguous()` | Contiguous adjacency     | Cache-friendly GCN    |
| CAMFM.A2c | `designB/fast_inference_v4.py` | 82-84 | `delta_coord` pre-alloc   | Pre-allocated buffer     | Avoid runtime alloc   |
| CAMFM.A2c | `designB/fast_inference_v4.py` | 170   | `proj_feat.view()`        | Reshape without copy     | Zero-copy reshape     |

### CAMFM.A2d_OPTIONAL_ACCEL

| StageID   | File Path                              | Line | Code Pattern              | What It Does           | Impact                    |
| --------- | -------------------------------------- | ---- | ------------------------- | ---------------------- | ------------------------- |
| CAMFM.A2d | `designB/fast_inference_v4.py`         | 203  | `@torch.inference_mode()` | Inference mode         | Disable autograd overhead |
| CAMFM.A2d | `designB/fast_inference_v4_metrics.py` | N/A  | `torch.cuda.amp.autocast` | AMP (optional)         | FP16 acceleration         |
| CAMFM.A2d | `designB/fast_inference_v4_metrics.py` | N/A  | `torch.compile`           | Compilation (optional) | Kernel fusion             |

### CAMFM.A3_METRICS

| StageID  | File Path                              | Function/Class       | What It Does        | Evidence Artifact                               |
| -------- | -------------------------------------- | -------------------- | ------------------- | ----------------------------------------------- |
| CAMFM.A3 | `designB/fast_inference_v4_metrics.py` | `compute_metrics()`  | CD, F1@τ, F1@2τ     | `outputs/designB/benchmark/metrics_results.csv` |
| CAMFM.A3 | `designB/modules/chamfer_pytorch.py`   | `chamfer_distance()` | GPU Chamfer         | N/A                                             |
| CAMFM.A3 | `designB/fast_inference_v4.py`         | `main()` stdout      | Timing statistics   | Console output                                  |
| CAMFM.A3 | `designA/compute_metrics.py`           | `main()`             | TF Chamfer metrics  | `outputs/designA/benchmark/metrics_results.csv` |
| CAMFM.A3 | `outputs/designA/benchmark/`           | N/A                  | All benchmark files | `DesignA_Evaluation_Summary.md`                 |

### CAMFM.A5_METHOD

| StageID  | File Path                              | Location              | What It Does          | Evidence Artifact       |
| -------- | -------------------------------------- | --------------------- | --------------------- | ----------------------- |
| CAMFM.A5 | `designA/designA_eval_list.txt`        | N/A                   | 1000 balanced samples | Reproducible evaluation |
| CAMFM.A5 | `designA/eval_designA_stage1.py:33-34` | `np.random.seed(123)` | Seed fixing           | Reproducibility         |
| CAMFM.A5 | `docs/BENCHMARK_PROTOCOL.md`           | N/A                   | Timing methodology    | Protocol documentation  |
| CAMFM.A5 | `docs/DESIGNS.md`                      | N/A                   | Design specifications | Method documentation    |

### DATA.\* (Design C - Planned)

| StageID                  | File Path                            | Function/Class        | What It Does     | Impact           |
| ------------------------ | ------------------------------------ | --------------------- | ---------------- | ---------------- |
| DATA.READ_CPU            | `designC/dali_pipeline.py` (planned) | `fn.readers.file`     | File system read | Async prefetch   |
| DATA.DECODE_GPU_NVJPEG   | `designC/dali_pipeline.py` (planned) | `fn.decoders.image`   | GPU JPEG decode  | CPU decode → GPU |
| DATA.RESIZE_GPU          | `designC/dali_pipeline.py` (planned) | `fn.resize`           | GPU resize       | CPU resize → GPU |
| DATA.NORMALIZE_GPU       | `designC/dali_pipeline.py` (planned) | `fn.normalize`        | GPU normalize    | CPU norm → GPU   |
| DATA.DALI_BRIDGE_PYTORCH | `designC/dali_pipeline.py` (planned) | `DALIGenericIterator` | Zero-copy bridge | Minimize H2D     |

---

## Evidence Artifacts Summary

| Design       | Artifact Path                                             | Contents           |
| ------------ | --------------------------------------------------------- | ------------------ |
| DESIGN.A     | `outputs/designA/benchmark/combined_timings.txt`          | Total timing stats |
| DESIGN.A     | `outputs/designA/benchmark/metrics_results.csv`           | Per-sample CD, F1  |
| DESIGN.A     | `outputs/designA/benchmark/DesignA_Evaluation_Summary.md` | Full summary       |
| DESIGN.A     | `outputs/designA/eval_meshes/*.xyz`                       | Output meshes      |
| DESIGN.A_GPU | `outputs/designA_GPU/eval_meshes/evaluation_summary.txt`  | GPU timing stats   |
| DESIGN.A_GPU | `outputs/designA_GPU/eval_meshes/timing_results.csv`      | Per-sample timing  |
| DESIGN.B     | `outputs/designB/eval_meshes/*.obj`                       | Output meshes      |
| DESIGN.B     | `outputs/designB/benchmark/`                              | Metrics and timing |

---

## Code Tag Locations

The following in-code tags have been added for traceability:

| Tag                                    | File                                   | Line | Purpose                  |
| -------------------------------------- | -------------------------------------- | ---- | ------------------------ |
| `[DESIGN.B][CAMFM.A2a_GPU_RESIDENCY]`  | `designB/fast_inference_v4.py`         | 45   | Model data GPU placement |
| `[DESIGN.B][CAMFM.A2b_STEADY_STATE]`   | `designB/fast_inference_v4.py`         | 23   | cuDNN/TF32 config        |
| `[DESIGN.B][CAMFM.A2b_STEADY_STATE]`   | `designB/fast_inference_v4.py`         | 101  | Warmup loop              |
| `[DESIGN.B][CAMFM.A2c_MEM_LAYOUT]`     | `designB/fast_inference_v4.py`         | 82   | Pre-allocation           |
| `[DESIGN.B][CAMFM.A2d_OPTIONAL_ACCEL]` | `designB/fast_inference_v4.py`         | 203  | inference_mode           |
| `[DESIGN.B][CAMFM.A3_METRICS]`         | `designB/fast_inference_v4_metrics.py` | 47   | Metrics integration      |
| `[DESIGN.A]`                           | `designA/eval_designA_stage1.py`       | 29   | CPU enforcement          |
| `[DESIGN.A_GPU]`                       | `designA_GPU/eval_designA_gpu.py`      | 30   | GPU enablement           |

---

## Related Documents

- [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) - Visual pipeline diagrams
- [DESIGNS.md](DESIGNS.md) - Design specifications
- [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) - Timing methodology

---

_Last Updated: February 2026_
