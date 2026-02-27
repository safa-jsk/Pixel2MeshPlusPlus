# Design C — FaceScape Domain Adaptation

Design C extends the Pixel2Mesh++ pipeline (Design B: PyTorch GPU) to the
[FaceScape](https://facescape.nju.edu.cn/) face reconstruction domain.

## Status

> **Not yet implemented.** This directory contains skeleton scripts with
> `--facescape_root` and `--splits_csv` argument parsing.  The actual
> inference/evaluation logic is a TODO.

## Directory Layout

```
DesignC/
├── README.md                ← this file
├── scripts/
│   ├── infer_facescape.py   ← inference stub (--facescape_root, --splits_csv)
│   └── eval_facescape.py    ← evaluation stub
└── docs/
    └── DESIGN_C_PLAN.md     ← design notes
```

## Quick Start (future)

```bash
cd DesignC/scripts
python infer_facescape.py \
    --facescape_root /data/FaceScape \
    --splits_csv    splits/test.csv \
    --checkpoint    ../../artifacts/checkpoints/torch/facescape_model.pth
```
