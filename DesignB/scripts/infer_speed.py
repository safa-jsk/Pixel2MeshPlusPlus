#!/usr/bin/env python3
"""
Design B - Speed-only inference (no metrics)
Wraps src/p2mpp/torch/engine/fast_inference_v4.py

Usage:
    python infer_speed.py --checkpoint artifacts/checkpoints/torch/mvp2m_pytorch.pth \\
                          --dat_file assets/data_templates/iccv_p2mpp.dat \\
                          --image_dir /path/to/test/images
"""
import os, sys

# Path bootstrap
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_TORCH_ENGINE = os.path.join(_PROJECT_ROOT, 'src', 'p2mpp', 'torch', 'engine')

sys.path.insert(0, _TORCH_ENGINE)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src', 'p2mpp', 'torch'))
sys.path.insert(0, _PROJECT_ROOT)

os.chdir(_PROJECT_ROOT)

from fast_inference_v4 import main
import argparse

if __name__ == '__main__':
    # Forward all arguments to the engine's main()
    main()
