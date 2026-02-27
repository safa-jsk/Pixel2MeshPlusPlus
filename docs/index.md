# Documentation Index

Master index for the Pixel2Mesh++ Model Pipeline (Model 2) repository.

## Quick Links

| What                      | Where                                                       |
|---------------------------|-------------------------------------------------------------|
| **How to run Design A**   | [DesignA_CPU/README.md](../DesignA_CPU/README.md)           |
| **How to run Design B**   | [DesignB/README.md](../DesignB/README.md)                   |
| **Design C plan**         | [DesignC/README.md](../DesignC/README.md)                   |
| **Docker setup**          | [setup/Docker_setup.md](setup/Docker_setup.md)              |
| **Project root README**   | [../README.md](../README.md)                                |

---

## Methodology

| Document                         | Description                                        |
|----------------------------------|----------------------------------------------------|
| [DESIGNS.md](methodology/DESIGNS.md)                           | Overview of all three design variants              |
| [PIPELINE_OVERVIEW.md](methodology/PIPELINE_OVERVIEW.md)       | End-to-end pipeline description                    |
| [BENCHMARK_PROTOCOL.md](methodology/BENCHMARK_PROTOCOL.md)     | CAMFM benchmark methodology                        |
| [TRACEABILITY_MATRIX.md](methodology/TRACEABILITY_MATRIX.md)   | Requirements → Implementation traceability         |
| [Design_A.md](methodology/Design_A.md)                         | Design A (TF CPU baseline) details                  |
| [Design_B.md](methodology/Design_B.md)                         | Design B (PyTorch GPU) details                      |
| [Design_C.md](methodology/Design_C.md)                         | Design C (FaceScape domain adaptation) details      |
| [DESIGN_A_QUICK_REFERENCE.md](methodology/DESIGN_A_QUICK_REFERENCE.md) | Design A quick reference card            |
| [DESIGN_B_IMPLEMENTATION_PIPELINE.md](methodology/DESIGN_B_IMPLEMENTATION_PIPELINE.md) | Design B implementation pipeline |
| [DESIGN_B_METHODOLOGY_PIPELINE.md](methodology/DESIGN_B_METHODOLOGY_PIPELINE.md) | Design B CAMFM methodology             |
| [COMPREHENSIVE_DESIGN_SUMMARY.md](methodology/COMPREHENSIVE_DESIGN_SUMMARY.md) | Summary across all designs              |

---

## Setup & Environment

| Document                         | Description                                        |
|----------------------------------|----------------------------------------------------|
| [Docker_setup.md](setup/Docker_setup.md)                       | Docker image build & usage                         |
| [DESIGN_B_SETUP_COMPLETE.md](setup/DESIGN_B_SETUP_COMPLETE.md) | Design B PyTorch environment setup                 |

---

## Reports

| Document                         | Description                                        |
|----------------------------------|----------------------------------------------------|
| [DESIGN_B_GPU_ACCELERATION_REPORT.md](reports/DESIGN_B_GPU_ACCELERATION_REPORT.md) | GPU acceleration benchmarks |
| [DESIGN_B_REORGANIZATION_SUMMARY.md](reports/DESIGN_B_REORGANIZATION_SUMMARY.md) | Code reorganization summary  |

---

## Thesis Chapters

| Document                         | Description                                        |
|----------------------------------|----------------------------------------------------|
| [ch4_designA_functional_verification.md](thesis/ch4_designA_functional_verification.md) | Chapter 4 — Design A verification |
| [CHAPTER_4_1_DESIGN_PROCESS_METHODOLOGY.md](thesis/CHAPTER_4_1_DESIGN_PROCESS_METHODOLOGY.md) | Chapter 4.1 — Design process |
| [CHAPTER_4_2_DESIGN_SPECIFICATION.md](thesis/CHAPTER_4_2_DESIGN_SPECIFICATION.md) | Chapter 4.2 — Design specification |

---

## Directory Structure

```
Pixel2MeshPlusPlus/
├── src/p2mpp/               # Core library
│   ├── tf/                  #   TensorFlow modules, utils, scripts
│   └── torch/               #   PyTorch modules, engine, convert
├── DesignA_CPU/             # Design A — TF CPU baseline
├── DesignA_GPU/             # Design A — TF GPU variant
├── DesignB/                 # Design B — PyTorch GPU (CAMFM)
├── DesignC/                 # Design C — FaceScape (skeleton)
├── external/                # Custom ops (tf_ops/, torch_chamfer/)
├── configs/                 # YAML configurations
├── assets/                  # Static assets (templates, demo, figures)
├── artifacts/               # Runtime outputs (gitignored)
├── data/                    # Data lists and references
├── docker/                  # Dockerfiles
├── env/                     # Environment configs
├── tests/                   # Smoke tests
└── docs/                    # ← you are here
```
