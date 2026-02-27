#!/usr/bin/env python3
"""
Design B - Inference with CAMFM metrics (CD, F1@tau, F1@2tau)
Wraps src/p2mpp/torch/engine/fast_inference_v4_metrics.py

Usage:
    python infer_with_metrics.py --checkpoint artifacts/checkpoints/torch/mvp2m_pytorch.pth \\
                                 --dat_file assets/data_templates/iccv_p2mpp.dat \\
                                 --image_dir /path/to/test/images \\
                                 --gt_dir /path/to/ground_truth
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

from fast_inference_v4_metrics import main

if __name__ == '__main__':
    main()
