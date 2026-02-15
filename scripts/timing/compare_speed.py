#!/usr/bin/env python3
"""
Direct Speed Comparison: Design A vs Design B

This script measures the per-sample inference time for each design.
"""

import subprocess
import time
import sys

def main():
    # Test files
    test_samples = [
        "02691156_d004d0e86ea3d77a65a6ed5c458f6a32_00.dat",
        "02691156_b8e27c4f593ebe78c593b94c9c5f0efa_00.dat", 
        "02958343_36b4b4bcc80eb3a7d5e70fd26ab5e4_00.dat",
        "02958343_c7c1a03d9bade3188a95d2b8b4ad25be_00.dat",
        "03001627_a4d52e3bc0f35f4d49ec54f84c97e5ee_00.dat",
    ]
    
    print("=" * 60)
    print("SPEED COMPARISON: Design A vs Design B")
    print("=" * 60)
    
    # Create test file
    with open('/tmp/speed_test.txt', 'w') as f:
        for s in test_samples:
            f.write(s + '\n')
    
    # Copy to containers
    subprocess.run(['docker', 'cp', '/tmp/speed_test.txt', 
                    'friendly_beaver:/workspace/data/speed_test.txt'], check=True)
    subprocess.run(['docker', 'cp', '/tmp/speed_test.txt',
                    'p2mpp-pytorch:/workspace/data/speed_test.txt'], check=True)
    
    n_samples = len(test_samples)
    
    # === Design B (PyTorch GPU) ===
    print("\n[Design B] PyTorch GPU - Fast Unified Inference")
    cmd = '''docker exec p2mpp-pytorch bash -c "
cd /workspace && python pytorch_impl/fast_inference.py \\
    --test_file data/speed_test.txt \\
    --output_dir outputs/designB/speed_test"'''
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    design_b_total = time.time() - start
    
    # Extract mean time from output
    import re
    match = re.search(r'Mean time: ([\d.]+)ms', result.stdout)
    if match:
        design_b_per_sample = float(match.group(1))
    else:
        design_b_per_sample = (design_b_total / n_samples) * 1000
    
    print(f"  Total time: {design_b_total:.2f}s")
    print(f"  Per sample: {design_b_per_sample:.1f}ms")
    
    # === Design A (TensorFlow CPU) - Stage 1 only ===
    print("\n[Design A] TensorFlow CPU - Stage 1 (MVP2M)")
    cmd = '''docker exec friendly_beaver bash -c "
cd /workspace && python test_mvp2m.py -f cfgs/mvp2m.yaml \\
    --test_file_path data/speed_test.txt \\
    --test_image_path data/designA_subset/ShapeNetRendering/rendering_only \\
    --coarse_result_file_path outputs/designA/speed_test"'''
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    stage1_time = time.time() - start
    print(f"  Stage 1 time: {stage1_time:.2f}s")
    
    # === Design A - Stage 2 ===
    print("\n[Design A] TensorFlow CPU - Stage 2 (P2MPP)")
    cmd = '''docker exec friendly_beaver bash -c "
cd /workspace && python test_p2mpp.py -f cfgs/p2mpp.yaml \\
    --test_file_path data/speed_test.txt \\
    --test_image_path data/designA_subset/ShapeNetRendering/rendering_only \\
    --coarse_result_file_path outputs/designA/speed_test \\
    --test_data_path outputs/designA/speed_test"'''
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    stage2_time = time.time() - start
    print(f"  Stage 2 time: {stage2_time:.2f}s")
    
    design_a_total = stage1_time + stage2_time
    design_a_per_sample = (design_a_total / n_samples) * 1000
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nDesign A (TensorFlow CPU):")
    print(f"  Total: {design_a_total:.2f}s for {n_samples} samples")
    print(f"  Per sample: {design_a_per_sample:.1f}ms")
    
    print(f"\nDesign B (PyTorch GPU):")
    print(f"  Total: {design_b_total:.2f}s for {n_samples} samples")
    print(f"  Per sample: {design_b_per_sample:.1f}ms")
    
    speedup = design_a_per_sample / design_b_per_sample
    print(f"\n{'='*60}")
    if speedup > 1:
        print(f"SPEEDUP: {speedup:.2f}x - Design B is FASTER! 🚀")
    else:
        print(f"SLOWDOWN: {1/speedup:.2f}x - Design B is slower")
    print("=" * 60)

if __name__ == '__main__':
    main()
