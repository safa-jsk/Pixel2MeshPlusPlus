#!/usr/bin/env python3
"""
Quick GPU speed test for PyTorch Pixel2Mesh++ on RTX 4070
Tests inference speed without requiring trained weights
"""

import torch
import torch.nn as nn
import time
import sys

sys.path.insert(0, '/workspace')
from modules.models_p2mpp_pytorch import Pixel2MeshPyTorch


def test_gpu_speed(num_samples=35, num_runs=2):
    """Test GPU inference speed"""
    
    device = torch.device('cuda')
    print(f"=== PyTorch GPU Speed Test ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"")
    
    # Create model
    cfg = {
        'hidden_dim': 192,
        'coord_dim': 3,
        'batch_size': 1
    }
    
    model = Pixel2MeshPyTorch(cfg).to(device)
    model.eval()
    
    # Create dummy inputs (batch_size=1)
    batch_size = 1
    num_vertices = 2562  # Standard ellipsoid mesh
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    initial_vertices = torch.randn(batch_size, num_vertices, 3).to(device)
    
    # Create dummy adjacency matrices
    adj_matrix = torch.eye(num_vertices).to(device).unsqueeze(0)
    supports = [adj_matrix, adj_matrix]
    
    # Warmup
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(images, initial_vertices, supports)
    torch.cuda.synchronize()
    
    # Run benchmark
    times = []
    for run in range(num_runs):
        run_times = []
        print(f"\n[Run {run+1}/{num_runs}] Processing {num_samples} samples...")
        
        with torch.no_grad():
            for i in range(num_samples):
                torch.cuda.synchronize()
                start = time.time()
                
                _ = model(images, initial_vertices, supports)
                
                torch.cuda.synchronize()
                elapsed = time.time() - start
                run_times.append(elapsed)
                
                if (i + 1) % 10 == 0:
                    avg_so_far = sum(run_times) / len(run_times)
                    print(f"  Processed {i+1}/{num_samples} samples (avg: {avg_so_far*1000:.2f}ms/sample)")
        
        total_time = sum(run_times)
        avg_time = total_time / num_samples
        times.append(total_time)
        
        print(f"[Run {run+1}] Total: {total_time:.2f}s, Avg: {avg_time*1000:.2f}ms/sample")
    
    # Summary
    mean_time = sum(times) / len(times)
    print(f"\n{'='*50}")
    print(f"PYTORCH GPU RESULTS (RTX 4070)")
    print(f"{'='*50}")
    print(f"Samples: {num_samples}")
    print(f"Runs: {num_runs}")
    print(f"Mean time: {mean_time:.2f}s")
    print(f"Throughput: {num_samples/mean_time:.2f} samples/sec")
    print(f"")
    
    # Compare with TensorFlow baseline
    tf_cpu_time = 6.96  # From benchmark_tf210.log (Design A)
    speedup = tf_cpu_time / mean_time
    print(f"TensorFlow CPU baseline: {tf_cpu_time:.2f}s")
    print(f"PyTorch GPU (this run): {mean_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"")
    
    if speedup > 2.0:
        print(f"✓ EXCELLENT GPU acceleration achieved!")
    elif speedup > 1.2:
        print(f"✓ Good GPU acceleration achieved!")
    else:
        print(f"⚠ Limited acceleration - may need optimization")
    
    return mean_time, speedup


if __name__ == '__main__':
    import sys
    
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    test_gpu_speed(num_samples, num_runs)
