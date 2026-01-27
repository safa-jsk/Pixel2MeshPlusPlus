# Design A - Quick Reference

All Design A files are now organized in the `designA/` folder.

## 📁 Directory Structure

```
designA/
├── README.md                          ← Main documentation
├── A5_IMPLEMENTATION_SUMMARY.md       ← Implementation details
├── EVAL_LIST_README.md                ← Eval list docs
├── designA_eval_list.txt              ← 35 samples
│
├── eval_designA_complete.py           ← Main: 2-stage pipeline
├── eval_designA.py                    ← Alt: Stage 2 only
│
├── quick_start_designA.sh             ← Interactive runner ⭐
├── run_designA_eval.sh                ← Automated runner
└── collect_system_info.sh             ← System specs

outputs/designA/                       ← Results (auto-created)
├── eval_meshes/                       ← Generated meshes
└── benchmark/                         ← Performance metrics
```

## 🚀 Quick Start

```bash
# Inside Docker container
cd designA
bash quick_start_designA.sh
```

## 📖 Documentation

- **Start here:** [designA/README.md](designA/README.md)
- **Implementation:** [designA/A5_IMPLEMENTATION_SUMMARY.md](designA/A5_IMPLEMENTATION_SUMMARY.md)
- **Full roadmap:** [Design_A.md](Design_A.md)

## ✅ What's Working

- ✅ Evaluation list: 35 verified samples
- ✅ Complete 2-stage inference pipeline
- ✅ Automatic timing measurement (A6)
- ✅ Multiple output formats (.xyz, .obj)
- ✅ Hardware specs collection
- ✅ All paths updated for new structure

## 🎯 Next Steps

1. `cd designA`
2. `bash quick_start_designA.sh`
3. Wait ~6-10 minutes
4. Review results in `../outputs/designA/`

**Note:** All scripts now use relative paths from the `designA/` directory.
