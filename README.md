# Pixel2Mesh++

Implementation of the ICCV'19 paper "Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation".
[Paper](https://arxiv.org/abs/1908.01491) | [Project Page](https://walsvid.github.io/Pixel2MeshPlusPlus)

This repository is organized around **four design variants** for a thesis
comparison study using the **CAMFM methodology** on Ubuntu 24.04.3.

---

## Design Variants

| Design | Directory | Framework | Description |
|--------|-----------|-----------|-------------|
| **A (CPU)** | [`DesignA_CPU/`](DesignA_CPU/) | TensorFlow 1.15 | Original CPU baseline |
| **A (GPU)** | [`DesignA_GPU/`](DesignA_GPU/) | TensorFlow 1.15 | GPU-enabled via Docker |
| **B** | [`DesignB/`](DesignB/) | PyTorch 2.1+ | GPU-optimized (CAMFM) |
| **C** | [`DesignC/`](DesignC/) | PyTorch 2.1+ | FaceScape domain (skeleton) |

Core library code lives in [`src/p2mpp/`](src/p2mpp/) with `tf/` and `torch/` sub-packages.

---

## Quick Start

### Design A — CPU Evaluation

```bash
cd DesignA_CPU/scripts
bash eval.sh            # runs 3-stage evaluation
# or:  bash quick_start_designA.sh
```

### Design B — PyTorch GPU Benchmark

```bash
cd DesignB/scripts
python infer_speed.py --help
python infer_with_metrics.py --help
bash benchmark.sh
```

### Demo (single-image 3D reconstruction)

```bash
cd DesignA_CPU/scripts
bash demo.sh
# Output: artifacts/outputs/temp/predict.obj
```

---

## Repository Layout

```
Pixel2MeshPlusPlus/
├── src/p2mpp/               # Core library (importable)
│   ├── tf/                  #   TensorFlow: modules/, utils/, scripts/
│   └── torch/               #   PyTorch:    modules/, engine/, convert/, utils/
├── DesignA_CPU/             # Design A CPU scripts & eval lists
├── DesignA_GPU/             # Design A GPU scripts & Docker
├── DesignB/                 # Design B wrappers & Docker
├── DesignC/                 # Design C stubs (FaceScape)
├── external/                # Custom CUDA ops
│   ├── tf_ops/              #   TF Chamfer/EMD ops (.so, sources)
│   └── torch_chamfer/       #   PyTorch Chamfer CUDA extension
├── configs/                 # YAML configurations
│   ├── designA/             #   mvp2m.yaml, p2mpp.yaml
│   └── designB/             #   p2mpp_pytorch.yaml
├── assets/                  # Static assets
│   ├── data_templates/      #   iccv_p2mpp.dat, face3.obj
│   ├── demo_inputs/         #   plane1-3.png, cameras.txt
│   └── figures/             #   README images
├── artifacts/               # Runtime outputs (GITIGNORED)
│   ├── checkpoints/         #   tf/, torch/
│   └── outputs/             #   designA/, designA_GPU/, temp/
├── data/                    # Data lists (train_list.txt, test_list.txt)
├── docker/                  # Dockerfiles (cpu)
├── env/                     # Environment configs
├── tests/                   # Smoke tests
└── docs/                    # Documentation (see docs/index.md)
```

---

## Documentation

See **[docs/index.md](docs/index.md)** for the full documentation index.

Key documents:
- [DESIGNS.md](docs/methodology/DESIGNS.md) — All design specifications
- [PIPELINE_OVERVIEW.md](docs/methodology/PIPELINE_OVERVIEW.md) — Pipeline diagrams
- [BENCHMARK_PROTOCOL.md](docs/methodology/BENCHMARK_PROTOCOL.md) — CAMFM timing methodology
- [TRACEABILITY_MATRIX.md](docs/methodology/TRACEABILITY_MATRIX.md) — Code-to-stage mapping
- [Docker_setup.md](docs/setup/Docker_setup.md) — Container setup

### In-Code Tags

Critical code sections are labelled with methodology identifiers:
`[DESIGN.A]`, `[DESIGN.B]`, `[CAMFM.A2a_GPU_RESIDENCY]`,
`[CAMFM.A2b_STEADY_STATE]`, `[CAMFM.A2c_MEM_LAYOUT]`,
`[CAMFM.A2d_OPTIONAL_ACCEL]`, `[CAMFM.A3_METRICS]`.

---

## Training (TensorFlow — Design A)

1. **Train coarse shape** (Stage 1):
   ```bash
   cd DesignA_CPU/scripts && bash train_stage1.sh
   ```
2. **Generate intermediate meshes**:
   ```bash
   cd src/p2mpp/tf/scripts
   python generate_mvp2m_intermediate.py -f ../../configs/designA/mvp2m.yaml
   ```
3. **Train refinement** (Stage 2):
   ```bash
   cd DesignA_CPU/scripts && bash train_stage2.sh
   ```

## Evaluation

See the Quick Start section above, or each Design's README.

---

## Dataset

[ShapeNet](https://www.shapenet.org/) models with rendered views from
[3D-R2N2](https://github.com/chrischoy/3D-R2N2). Train/test splits in
[`data/`](data/).

## Custom CUDA Ops

TensorFlow Chamfer/EMD ops: `external/tf_ops/` — see the included Makefile.
PyTorch Chamfer extension: `external/torch_chamfer/` — build with
`cd external/torch_chamfer && python setup.py build_ext --inplace`.

---

## Citation

```bibtex
@inProceedings{wen2019pixel2mesh++,
  title={Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation},
  author={Chao Wen and Yinda Zhang and Zhuwen Li and Yanwei Fu},
  booktitle={ICCV},
  year={2019}
}
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
