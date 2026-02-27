# Pixel2Mesh++ Project Structure

This document describes the organized structure of the Pixel2Mesh++ project.

## Directory Overview

```
Pixel2MeshPlusPlus/
├── README.md                    # Main project documentation
├── LICENSE                      # Project license
├── PROJECT_STRUCTURE.md         # This file
├── .gitignore
│
├── designA/                     # Design A: TensorFlow CPU Implementation (Baseline)
│   ├── README.md               # Design A overview and usage
│   ├── run_designA_eval.sh     # Evaluation runner script
│   ├── compute_metrics.py      # Metrics computation
│   ├── metrics.py              # Metrics calculator class
│   ├── eval_*.py               # Evaluation scripts
│   └── designA_eval_list.txt   # Balanced 1000-sample evaluation list
│
├── designA_GPU/                 # Design A: GPU variant (experimental)
│   ├── README.md
│   ├── Dockerfile
│   └── eval_designA_gpu*.py
│
├── designB/                     # Design B: PyTorch GPU Implementation
│   ├── README.md               # Design B overview
│   ├── SETUP_GUIDE.md          # Installation and setup
│   ├── QUICK_COMMANDS.md       # Common commands reference
│   ├── FILE_INDEX.md           # File index and navigation
│   ├── cfgs/                   # Configuration files
│   ├── checkpoints/            # Model checkpoints
│   ├── modules/                # PyTorch model definitions
│   ├── docs/                   # Design B specific documentation
│   ├── fast_inference_v4.py    # Fast inference script
│   └── convert_*.py            # Checkpoint conversion scripts
│
├── docs/                        # All documentation
│   ├── DESIGN_A_QUICK_REFERENCE.md
│   ├── FILE_INDEX.md           # Project file index
│   ├── thesis/                 # Thesis chapters
│   │   ├── CHAPTER_4_1_DESIGN_PROCESS_METHODOLOGY.md
│   │   ├── CHAPTER_4_2_DESIGN_SPECIFICATION.md
│   │   └── ch4_designA_functional_verification.md
│   ├── design/                 # Design documentation
│   │   ├── Design_A.md         # Design A specification
│   │   ├── Design_B.md         # Design B specification
│   │   ├── Design_C.md         # Design C specification
│   │   ├── COMPREHENSIVE_DESIGN_SUMMARY.md
│   │   ├── DESIGN_B_IMPLEMENTATION_PIPELINE.md
│   │   ├── DESIGN_B_METHODOLOGY_PIPELINE.md
│   │   ├── designB_gpu_strategy.md
│   │   └── designB_implementation_roadmap.md
│   ├── setup/                  # Setup guides
│   │   ├── Docker_setup.md
│   │   └── DESIGN_B_SETUP_COMPLETE.md
│   └── reports/                # Analysis and reports
│       ├── DESIGN_B_GPU_ACCELERATION_REPORT.md
│       └── DESIGN_B_REORGANIZATION_SUMMARY.md
│
├── scripts/                     # Utility and testing scripts
│   ├── timing/                 # Timing analysis scripts
│   │   ├── tf_timing.py
│   │   ├── tf_timing_full.py
│   │   ├── tf_timing_full2.py
│   │   └── compare_speed.py
│   ├── tests/                  # Test scripts
│   │   ├── tf_simple_test.py
│   │   └── tf_stage1_test.py
│   └── shell/                  # Shell scripts
│       ├── DESIGN_B_COMMANDS.sh
│       ├── run_designB_inference.sh
│       └── wait_and_test.sh
│
├── docker/                      # Docker configuration
│   └── Dockerfile.cpu
│
├── data/                        # Dataset and data files
│   ├── README.md               # Data documentation
│   ├── train_list.txt
│   ├── test_list.txt
│   ├── demo/                   # Demo data
│   └── p2mpp_models/           # Pre-trained model checkpoints
│       ├── coarse_mvp2m/
│       └── refine_p2mpp/
│
├── outputs/                     # Evaluation results and outputs
│   ├── designA/
│   │   └── benchmark/          # Evaluation metrics and summaries
│   ├── designA_GPU/
│   └── temp/                   # Temporary files (nohup, logs)
│
├── modules/                     # Core TensorFlow modules (Design A)
│   ├── __init__.py
│   ├── chamfer.py
│   ├── config.py
│   ├── layers.py
│   ├── losses.py
│   ├── models_mvp2m.py
│   └── models_p2mpp.py
│
├── external/                    # External dependencies (TensorFlow ops)
│   ├── tf_approxmatch.py
│   ├── tf_nndistance.py
│   └── ...
│
├── utils/                       # Shared utility functions
│   ├── __init__.py
│   ├── dataloader.py
│   ├── tools.py
│   ├── visualize.py
│   └── xyz2obj.py
│
├── cfgs/                        # Configuration files (Design A)
│   ├── mvp2m.yaml
│   └── p2mpp.yaml
│
├── results/                     # Generated results
│   ├── coarse_mvp2m/
│   └── refine_p2mpp/
│
├── tensorflow_backup/           # TensorFlow files backup
│
└── Core Scripts (Root Level)
    ├── demo.py                  # Quick demo script
    ├── train_mvp2m.py           # Train coarse network
    ├── train_p2mpp.py           # Train refinement network
    ├── test_mvp2m.py            # Test coarse network
    ├── test_p2mpp.py            # Test refinement network
    ├── generate_mvp2m_intermediate.py  # Generate intermediate results
    ├── f_score.py               # F-score evaluation
    └── cd_distance.py           # Chamfer distance evaluation
```

## Key Components

### Design A (TensorFlow CPU - Baseline)

- **Location**: `designA/`
- **Framework**: TensorFlow 1.15
- **Execution**: Docker container `p2mpp:cpu`
- **Purpose**: Original implementation baseline

### Design B (PyTorch GPU - Accelerated)

- **Location**: `designB/`
- **Framework**: PyTorch 2.1.2+cu118
- **Execution**: Native with CUDA
- **Purpose**: GPU-accelerated implementation

### Documentation

- **Location**: `docs/`
- **Content**: Thesis chapters, design specs, setup guides, reports

### Evaluation

- **Evaluation List**: `designA/designA_eval_list.txt` (1000 balanced samples)
- **Results**: `outputs/designA/benchmark/`
- **Metrics**: Chamfer Distance, F1@τ, F1@2τ

## Quick Commands

### Run Design A Evaluation (Docker)

```bash
docker run --rm -v $(pwd):/workspace -w /workspace p2mpp:cpu \
    bash designA/run_designA_eval.sh
```

### Run Design B Inference

```bash
cd designB
python inference/fast_inference.py --config cfgs/default.yaml
```

## File Organization Rules

1. **Root Level**: Only essential files (README, LICENSE, main scripts)
2. **Design-Specific**: Each design has its own folder
3. **Documentation**: All .md docs go to `docs/` subfolders
4. **Scripts**: Utility scripts go to `scripts/` with categorization
5. **Outputs**: All generated results go to `outputs/`

---

_Last Updated: February 2026_
