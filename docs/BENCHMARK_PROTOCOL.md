# Benchmark Protocol - Pixel2Mesh++

This document specifies the benchmarking methodology used for timing and performance evaluation across all designs.

---

## 1. Overview

All performance measurements follow CAMFM.A2b_STEADY_STATE and CAMFM.A5_METHOD principles to ensure reproducible, accurate timing.

| Aspect                   | Requirement                                      |
| ------------------------ | ------------------------------------------------ |
| **Warmup**               | Required before timed region                     |
| **Synchronization**      | GPU sync at timing boundaries                    |
| **Excluded from Timing** | Disk I/O, data loading, metric computation       |
| **Included in Timing**   | Forward pass only (inference)                    |
| **Repetitions**          | Minimum 100 samples for statistical significance |

---

## 2. Warmup Protocol

### Purpose

Warmup iterations allow:

1. cuDNN to cache optimal kernel configurations
2. GPU frequency to stabilize at boost clock
3. JIT compilation to complete (PyTorch)
4. Memory allocators to pre-allocate pools

### Design-Specific Warmup

| Design       | Warmup Count | Implementation                     |
| ------------ | ------------ | ---------------------------------- |
| DESIGN.A     | 0            | None (baseline measurement)        |
| DESIGN.A_GPU | 1            | Single forward pass before timing  |
| DESIGN.B     | 15           | Extended warmup for cuDNN autotune |

### DESIGN.B Warmup Code

```python
# designB/fast_inference_v4.py:101-113
def _warmup(self):
    dummy_img = torch.randn(3, 3, 224, 224, device=self.device)
    dummy_cam = np.array([[0, 25, 0, 1.9, 25], ...])

    with torch.inference_mode():
        for _ in range(15):  # [CAMFM.A2b] Extended warmup
            _ = self.stage1_model(...)
            _ = self.stage2_model.cnn(dummy_img)
            ...
    torch.cuda.synchronize()
```

---

## 3. Synchronization Rules

### GPU Timing Boundaries

**Critical**: GPU operations are asynchronous. Without explicit synchronization, CPU timing will underestimate actual GPU time.

```python
# CORRECT: Proper GPU timing
torch.cuda.synchronize()  # Wait for previous work
start = time.time()
output = model(input)
torch.cuda.synchronize()  # Wait for inference to complete
elapsed = time.time() - start

# INCORRECT: Will measure kernel launch time only
start = time.time()
output = model(input)  # Asynchronous!
elapsed = time.time() - start  # Wrong!
```

### TensorFlow Synchronization

```python
# TensorFlow handles sync implicitly in sess.run()
start_time = time.time()
out = sess.run(model.output, feed_dict=feed_dict)  # Blocks until complete
elapsed = time.time() - start_time
```

### Synchronization Locations

| Design       | Pre-Sync Location          | Post-Sync Location         |
| ------------ | -------------------------- | -------------------------- |
| DESIGN.A     | N/A (CPU)                  | N/A (CPU)                  |
| DESIGN.A_GPU | Implicit                   | Implicit (sess.run)        |
| DESIGN.B     | `fast_inference_v4.py:298` | `fast_inference_v4.py:301` |

---

## 4. Excluded from Timed Region

The following operations are **excluded** from timing measurements:

| Operation                | Reason                  | Location             |
| ------------------------ | ----------------------- | -------------------- |
| Image file loading       | Disk I/O variability    | Before timing loop   |
| Image preprocessing      | CPU-bound, not model    | Before timing loop   |
| Model checkpoint loading | One-time cost           | `__init__()`         |
| Mesh data loading        | One-time cost           | `__init__()`         |
| Mesh file saving         | Post-inference I/O      | After timing loop    |
| Metric computation       | Analysis, not inference | After timing loop    |
| Console printing         | I/O overhead            | Outside timed region |

### Timing Region Definition

```
┌─────────────────────────────────────────────────────────────┐
│  EXCLUDED: Model loading, data loading, warmup              │
├─────────────────────────────────────────────────────────────┤
│  TIMED REGION:                                               │
│    sync_start()                                              │
│    ├── Feature extraction (VGG/CNN)                         │
│    ├── Feature projection                                    │
│    ├── Stage 1: Graph convolutions + pooling                │
│    ├── Stage 2: DRB1 + DRB2 deformation                     │
│    └── Final mesh coordinates                                │
│    sync_end()                                                │
├─────────────────────────────────────────────────────────────┤
│  EXCLUDED: Mesh saving, metric computation, logging         │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Timing Measurement Code

### DESIGN.A (TensorFlow CPU)

```python
# designA/eval_designA_stage1.py:102-105
t_start = time.time()
out3 = sess.run(model.output3, feed_dict=feed_dict)
t_elapsed = time.time() - t_start
timings.append(t_elapsed)
```

### DESIGN.A_GPU (TensorFlow GPU)

```python
# designA_GPU/eval_designA_gpu.py:176-183
start_time = time.time()
out1l, out2l = sess.run([model.output1l, model.output2l], feed_dict=feed_dict)
elapsed = time.time() - start_time
timing_results.append((data_id, elapsed))
```

### DESIGN.B (PyTorch GPU)

```python
# designB/fast_inference_v4.py:296-303
torch.cuda.synchronize()  # [CAMFM.A2b] Pre-sync
start = time.time()
mesh_gpu = engine.infer(imgs_tensor, poses)
torch.cuda.synchronize()  # [CAMFM.A2b] Post-sync
elapsed = time.time() - start
times.append(elapsed)
```

---

## 6. Statistical Reporting

### Required Statistics

| Statistic  | Formula                                                      | Purpose                 |
| ---------- | ------------------------------------------------------------ | ----------------------- |
| Mean       | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$                    | Average latency         |
| Std Dev    | $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$ | Variability             |
| Median     | Middle value when sorted                                     | Robust central tendency |
| Min        | $\min(x_i)$                                                  | Best-case latency       |
| Max        | $\max(x_i)$                                                  | Worst-case latency      |
| Throughput | $\frac{n}{\sum x_i}$ samples/sec                             | Processing rate         |

### Example Output Format

```
=====================================
TIMING STATISTICS
=====================================
Total samples: 1000
Mean latency: 45.2ms ± 3.1ms
Median latency: 44.8ms
Min latency: 41.2ms
Max latency: 58.3ms
Throughput: 22.1 samples/sec
Total time: 45.2s
=====================================
```

---

## 7. Quality Metrics Protocol

### Metrics Computed

| Metric           | Formula                                | Threshold  |
| ---------------- | -------------------------------------- | ---------- | ---------------------------------------------------- | --- | ----------------------------------------- | --- |
| Chamfer Distance | $CD = \frac{1}{                        | P          | }\sum*{p \in P} \min*{q \in Q} \|p-q\|^2 + \frac{1}{ | Q   | }\sum*{q \in Q} \min*{p \in P} \|q-p\|^2$ | N/A |
| F1@τ             | $F1 = \frac{2 \cdot P \cdot R}{P + R}$ | τ = 0.0001 |
| F1@2τ            | $F1 = \frac{2 \cdot P \cdot R}{P + R}$ | τ = 0.0002 |

Where:

- $P = \frac{|\{p : d(p, Q) \leq \tau\}|}{|P|}$ (Precision)
- $R = \frac{|\{q : d(q, P) \leq \tau\}|}{|Q|}$ (Recall)

### Metric Computation Location

| Design   | Script                                 | Output                                          |
| -------- | -------------------------------------- | ----------------------------------------------- |
| DESIGN.A | `designA/compute_metrics.py`           | `outputs/designA/benchmark/metrics_results.csv` |
| DESIGN.B | `designB/fast_inference_v4_metrics.py` | `outputs/designB/benchmark/metrics_results.csv` |

---

## 8. Environment Requirements

### Hardware Documentation

```bash
# Capture system info before benchmarking
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cat /proc/cpuinfo | grep "model name" | head -1
free -h
```

### Software Versions

```bash
python --version
pip show torch tensorflow | grep Version
nvidia-smi | grep "CUDA Version"
```

### Expected Environment

| Component | DESIGN.A | DESIGN.A_GPU | DESIGN.B             |
| --------- | -------- | ------------ | -------------------- |
| Python    | 3.6      | 3.6-3.8      | 3.8-3.10             |
| Framework | TF 1.15  | TF 1.x/2.x   | PyTorch 2.1+         |
| CUDA      | N/A      | 10.0-11.x    | 11.8+                |
| GPU       | N/A      | NVIDIA       | NVIDIA (RTX/Ampere+) |

---

## 9. Reproducibility Checklist

### Before Benchmarking

- [ ] Close other GPU applications
- [ ] Set fixed random seed
- [ ] Use consistent evaluation list
- [ ] Verify checkpoint versions
- [ ] Document hardware/software versions

### During Benchmarking

- [ ] Run warmup iterations
- [ ] Use proper synchronization
- [ ] Exclude I/O from timing
- [ ] Log per-sample times

### After Benchmarking

- [ ] Compute statistics
- [ ] Save raw timing data
- [ ] Generate summary report
- [ ] Validate against expected range

---

## 10. Known Issues

### cuDNN Non-Determinism

cuDNN benchmark mode (`torch.backends.cudnn.benchmark = True`) selects algorithms based on input size. For reproducible timing:

- Use consistent input sizes
- Allow warmup for algorithm selection caching

### TF32 Precision

TF32 provides ~2x speedup but with slightly reduced precision (19-bit mantissa vs 23-bit for FP32). For Pixel2Mesh++, this does not affect mesh quality.

### First-Sample Variance

First few samples after warmup may still show higher variance. Consider excluding first 5-10 samples from statistics.

---

## Related Documents

- [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) - Visual pipeline diagrams
- [DESIGNS.md](DESIGNS.md) - Design specifications
- [TRACEABILITY_MATRIX.md](TRACEABILITY_MATRIX.md) - Code-to-stage mapping

---

_Last Updated: February 2026_
