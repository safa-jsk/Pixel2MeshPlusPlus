# Design C — Planning Notes

## Objective

Adapt the Pixel2Mesh++ pipeline (from Design B's PyTorch GPU implementation)
to reconstruct 3D face meshes from the
[FaceScape](https://facescape.nju.edu.cn/) dataset.

## Key Differences from Design B (ShapeNet)

| Aspect          | Design B (ShapeNet)     | Design C (FaceScape)       |
|-----------------|-------------------------|----------------------------|
| Domain          | Generic objects          | Human faces                |
| Input views     | 3 views × 224×224       | TBD (likely 1–3 views)    |
| Init mesh       | Ellipsoid (156 verts)   | TBD (face template?)      |
| Ground truth    | Sampled point clouds    | FaceScape mesh scans       |
| Metrics         | CD, F1@τ                | CD, F1@τ, + face-specific? |

## TODO

- [ ] Download and preprocess FaceScape dataset
- [ ] Design face-specific initial mesh template
- [ ] Adapt data loader for FaceScape image format
- [ ] Fine-tune from Design B checkpoint
- [ ] Implement `infer_facescape.py` logic
- [ ] Implement `eval_facescape.py` metrics
- [ ] Benchmark against face reconstruction baselines

## Data Requirements

```
/data/FaceScape/
├── facescape_trainset/      # Training meshes & images
├── facescape_testset/       # Test meshes & images
└── splits/
    ├── train.csv
    └── test.csv
```

## CAMFM Alignment

Design C will follow the same CAMFM methodology:
- **A2a**: GPU residency for face model
- **A2b**: Steady-state timing with warmup
- **A3**: CD + F1@τ metrics on FaceScape test split
- **A5**: Method reproducibility documentation
