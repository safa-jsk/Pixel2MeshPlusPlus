# Design A - GPU Enabled

This folder contains Design A with **GPU enabled** (no other optimizations from Design B).

## Purpose

Run the original TensorFlow 1.15 Pixel2Mesh++ implementation with GPU acceleration to compare against:

- Design A (CPU-only): Original implementation
- Design B (PyTorch GPU + AMP): Fully optimized implementation

## Key Differences from Design A

| Aspect                   | Design A (CPU)  | Design A GPU        |
| ------------------------ | --------------- | ------------------- |
| **CUDA_VISIBLE_DEVICES** | `''` (disabled) | `'0'` (enabled)     |
| **GPU Memory**           | N/A             | `allow_growth=True` |
| **Warmup**               | None            | 1 iteration         |
| **Framework**            | TensorFlow 1.15 | TensorFlow 1.15     |
| **Other Optimizations**  | None            | None                |

## Files

```
designA_GPU/
├── eval_designA_gpu.py    # Main evaluation script (GPU enabled)
├── run_eval_gpu.sh        # Docker run script
└── README.md              # This file
```

## Requirements

### CUDA Compatibility Issue

⚠️ **Important**: TensorFlow 1.15 requires **CUDA 10.0** and **cuDNN 7.4**, which are incompatible with modern NVIDIA GPUs (RTX 30xx, 40xx series require CUDA 11+).

| GPU Generation | Compute Capability | Required CUDA | TF 1.15 Compatible? |
| -------------- | ------------------ | ------------- | ------------------- |
| GTX 10xx       | 6.1                | CUDA 9+       | ✅ Yes              |
| RTX 20xx       | 7.5                | CUDA 10+      | ✅ Yes              |
| RTX 30xx       | 8.6                | CUDA 11+      | ❌ No               |
| RTX 40xx       | 8.9                | CUDA 11.8+    | ❌ No               |

### Docker Images

```bash
# TensorFlow 1.15 GPU (CUDA 10.0) - for older GPUs
docker pull tensorflow/tensorflow:1.15.5-gpu

# Alternative: TensorFlow 2.x with TF1 compatibility mode
docker pull tensorflow/tensorflow:2.10.0-gpu
```

## Usage

### Option 1: Run via Shell Script

```bash
cd /path/to/Pixel2MeshPlusPlus
chmod +x designA_GPU/run_eval_gpu.sh
./designA_GPU/run_eval_gpu.sh
```

### Option 2: Run Directly with Docker

```bash
cd /path/to/Pixel2MeshPlusPlus

# With TensorFlow 1.15 GPU
docker run --rm --gpus all \
    -v "$PWD:/workspace" \
    -w /workspace/designA_GPU \
    tensorflow/tensorflow:1.15.5-gpu \
    python eval_designA_gpu.py \
        --eval_list ../data/designA_eval_1000.txt \
        --output_dir ../outputs/designA_GPU/eval_1000 \
        --gpu_id 0
```

### Option 3: Run Locally (if TF 1.15 GPU is installed)

```bash
cd /path/to/Pixel2MeshPlusPlus/designA_GPU
python eval_designA_gpu.py \
    --eval_list ../data/designA_eval_1000.txt \
    --output_dir ../outputs/designA_GPU/eval_1000 \
    --gpu_id 0
```

## Expected Output

```
========================================================================
DESIGN A - GPU EVALUATION COMPLETE
========================================================================
Configuration:
  GPU Enabled: YES
  GPU ID: 0
  Framework: TensorFlow 1.15.5
========================================================================
Performance Metrics:
  Samples processed: 1000
  Mean latency: XXX.XXms ± XX.XXms
  Min latency: XXX.XXms
  Max latency: XXX.XXms
  Throughput: X.XX samples/sec
  Total time: XXX.XXs (X.XXmin)
========================================================================
```

## Output Files

| File                     | Description                                |
| ------------------------ | ------------------------------------------ |
| `*_predict.xyz`          | Predicted point cloud (2466 points)        |
| `*_predict.obj`          | Predicted mesh (2466 vertices, 4928 faces) |
| `*_ground.xyz`           | Ground truth point cloud                   |
| `timing_results.csv`     | Per-sample timing data                     |
| `evaluation_summary.txt` | Overall performance summary                |

## Comparison Context

This evaluation allows a fair comparison:

| Design             | Framework       | Hardware | Expected Latency         |
| ------------------ | --------------- | -------- | ------------------------ |
| Design A (CPU)     | TensorFlow 1.15 | CPU      | ~3547 ms                 |
| **Design A (GPU)** | TensorFlow 1.15 | GPU      | ~500-1000 ms (estimated) |
| Design B (GPU+AMP) | PyTorch 2.1     | GPU      | ~65 ms                   |

## Troubleshooting

### "No GPU devices found"

If running on RTX 30xx/40xx GPU:

```
TensorFlow 1.15 cannot use this GPU (requires CUDA 10.0)
Your GPU requires CUDA 11+
```

**Solution**: The benchmark will run on CPU instead, which is the same as Design A original.

### "CUDA driver version is insufficient"

```bash
# Check your CUDA version
nvidia-smi

# TF 1.15 needs CUDA 10.0, not 11.x or 12.x
```

**Solution**: Use an older GPU or accept CPU-only execution.
