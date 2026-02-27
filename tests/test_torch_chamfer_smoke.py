#!/usr/bin/env python3
"""
Smoke test: verify the PyTorch Chamfer CUDA extension can be built/loaded.

Run:  python tests/test_torch_chamfer_smoke.py

This test:
1. Checks source files exist
2. Attempts to build the extension (if CUDA available)
3. Runs a tiny forward pass
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHAMFER_DIR = os.path.join(PROJECT_ROOT, 'external', 'torch_chamfer')


def main():
    print("=" * 60)
    print("Smoke Test: PyTorch Chamfer Extension")
    print("=" * 60)

    # 1. Check source files
    print("\n--- Source files ---")
    src_files = ['chamfer_cuda.cpp', 'chamfer.cu', 'setup.py']
    all_found = True
    for f in src_files:
        path = os.path.join(CHAMFER_DIR, f)
        exists = os.path.isfile(path)
        print(f"  {'[OK]' if exists else '[MISSING]'} {f}")
        if not exists:
            all_found = False

    if not all_found:
        print("\nFAIL: Missing source files.")
        return 1

    # 2. Check torch+CUDA availability
    print("\n--- PyTorch + CUDA ---")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  PyTorch not installed — skipping build test")
        return 0

    if not cuda_available:
        print("  No CUDA device — skipping build test")
        return 0

    # 3. Try building
    print("\n--- Build extension ---")
    try:
        result = subprocess.run(
            [sys.executable, 'setup.py', 'build_ext', '--inplace'],
            cwd=CHAMFER_DIR,
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("  Build succeeded")
        else:
            print(f"  Build failed (rc={result.returncode})")
            print(f"  stderr: {result.stderr[:500]}")
            return 1
    except subprocess.TimeoutExpired:
        print("  Build timed out (120s)")
        return 1

    # 4. Try importing and running
    print("\n--- Forward pass ---")
    sys.path.insert(0, CHAMFER_DIR)
    try:
        import chamfer as chamfer_cuda
        a = torch.randn(1, 100, 3).cuda()
        b = torch.randn(1, 100, 3).cuda()
        dist1, dist2 = chamfer_cuda.forward(a, b)
        print(f"  dist1 shape: {dist1.shape}")
        print(f"  dist2 shape: {dist2.shape}")
        print(f"  dist1 mean:  {dist1.mean().item():.6f}")
        print("  Forward pass OK")
    except Exception as e:
        print(f"  Import/run failed: {e}")
        return 1

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
