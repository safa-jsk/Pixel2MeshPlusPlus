# Design C Roadmap — Modernized & Face-Targeted Pipeline (with CUDA)

**Goal:** Deliver a **FaceScape-aligned** preliminary system that runs reliably on **Ubuntu 22.04 + modern GPU stack**, produces **face meshes**, and is suitable for thesis Part-2 “alternative design + simulation/functional verification.”  
**Key rule:** Keep scope _minimal and defensible_. This is not a full rewrite of Pixel2Mesh++.

---

## C0. Choose a Design C Variant (pick ONE primary path)

Design C must be clearly different from A/B. You have two viable paths:

### **C-Variant-1: “Legacy Model + Minimal Modernization” (TF1-compat)**

Use Pixel2Mesh++ code but modernize _around_ it:

- reproducible tooling (Docker, scripts)
- data pipeline improvements
- logging/metrics, config hygiene
- stable CUDA execution (if possible)

**Pros:** Closer to Pixel2Mesh++; cleaner story  
**Cons:** TF1 GPU compatibility can be hard on modern hardware

### **C-Variant-2: “Modern Baseline Alternative (PyTorch/TensorFlow2) + FaceScape”**

Keep Pixel2Mesh++ as A/B, but in C you run a **modern mesh reconstruction baseline** that:

- runs on modern CUDA reliably
- consumes FaceScape subset
- outputs meshes you can show in poster

**Pros:** Highest chance of “it runs on your machine”  
**Cons:** Not Pixel2Mesh++; must justify as an alternative solution

**Decision Artifact**

- `docs/designC_variant_decision.md` (chosen variant + why, constraints)

---

## C1. Success Criteria (what “done” means)

By end of Design C you should have:

- ✅ FaceScape data pipeline working (subset)
- ✅ A model/pipeline that outputs **face meshes** (`.obj`) from images
- ✅ CUDA utilized (proof via `nvidia-smi` + logs)
- ✅ Functional verification: qualitative results + (optional) lightweight metric
- ✅ Clear comparison table across A/B/C (runtime + qualitative outcome + notes)

---

## C2. FaceScape Dataset Integration (minimal subset)

> Use a subset to reduce time and compute.

### C2.1 Define a small, repeatable subset

**Recommendation**

- 20–50 identities × 1 expression × fixed views (e.g., 4–8 views)
- Total samples: ~100–400 (enough for verification + poster)

**Tasks**

- [ ] Create `data/facescape_train_list.txt`
- [ ] Create `data/facescape_val_list.txt`
- [ ] Create `data/facescape_test_list.txt`
- [ ] Document exact selection rule (random seed or explicit IDs)

**Artifacts**

- `data/facescape_subset/README.md` (IDs, views, selection method)
- `data/facescape_*_list.txt`

### C2.2 Standardize folder layout (project-facing “canonical structure”)

**Tasks**

- [ ] Build a canonical structure that your pipeline reads consistently:
  - `facescape/images/<sample_id>/<view>.png`
  - `facescape/cameras/<sample_id>/<view>.json` (or `.txt`)
  - `facescape/meshes_gt/<sample_id>.obj` (if available/used)
- [ ] Write a conversion/organizer script if FaceScape raw format differs

**Artifacts**

- `tools/prepare_facescape_subset.py`
- `tools/prepare_facescape_subset.md` (how to run it)

---

## C3. Pipeline Implementation (depends on chosen variant)

# ===========================

# Variant 1: Legacy + Minimal Modernization

# ===========================

## C3-V1. Make Pixel2Mesh++ run reliably on modern Ubuntu

**Tasks**

- [ ] Containerize with pinned versions (documented)
- [ ] Replace fragile paths with config variables
- [ ] Add “one-command run scripts” for:
  - inference
  - benchmark
  - export meshes

**Artifacts**

- `env/designC/Dockerfile`
- `scripts/run_infer_designC.sh`
- `scripts/run_benchmark_designC.sh`

## C3-V1. FaceScape loader + preprocessing

**Tasks**

- [ ] Implement dataset reader that returns:
  - multi-view images
  - camera matrices/params
  - optional GT mesh/points (if used)
- [ ] Ensure it matches model input expectations (image size, normalization)

**Artifacts**

- `data_loader/facescape_dataset.py`
- `docs/facescape_input_spec.md`

## C3-V1. Enable CUDA execution

**Tasks**

- [ ] Ensure TF sees GPU
- [ ] Build/enable custom ops (if used)
- [ ] Validate that major ops run on GPU (device placement logs optional)

**Artifacts**

- `outputs/designC/logs/nvidia_smi_log.txt`
- `outputs/designC/logs/device_placement.txt` (optional)

---

# ===========================

# Variant 2: Modern Baseline Alternative

# ===========================

## C3-V2. Choose a modern baseline model (fastest-to-run)

**Goal:** A working face mesh reconstruction output on modern CUDA.

**Tasks**

- [ ] Select a baseline that:
  - has public code/weights (if possible)
  - supports image→mesh or image→3D face output
  - can be adapted to FaceScape input format
- [ ] Implement minimal wrapper:
  - input = FaceScape images
  - output = `.obj` mesh export

**Artifacts**

- `models/modern_baseline/` (code or submodule)
- `scripts/run_modern_baseline_facescape.sh`

## C3-V2. CUDA verification

**Tasks**

- [ ] Ensure model runs on GPU
- [ ] Capture GPU utilization evidence

**Artifacts**

- `outputs/designC/logs/nvidia_smi_log.txt`

---

## C4. Functional Verification (“Simulation”) Plan

Part-2 needs “simulation/functional verification.” For ML, your “simulation” is controlled experiments.

### C4.1 Verification test cases (minimum)

- **T1:** Single sample inference → mesh exports (pass/fail)
- **T2:** Batch inference on test list → all meshes exported (pass/fail)
- **T3:** Visual sanity: meshes are not degenerate (no NaNs, no exploded geometry)
- **T4:** Runtime benchmark on N samples (avg/sec)

**Artifacts**

- `docs/designC_verification_plan.md`
- `outputs/designC/verification/checklist_results.md`

### C4.2 Optional lightweight metric (if GT is usable)

Pick one:

- Chamfer distance between sampled points (pred vs GT)
- Point-to-surface error
- Simple landmark-based distance (if landmarks exist)

**Artifacts**

- `tools/eval_metric.py`
- `outputs/designC/metrics/metrics.csv`

---

## C5. Benchmarking and Comparison (A vs B vs C)

Use the same _measurement rules_:

- same machine
- same sample count
- warmup excluded

### Metrics to report

- Runtime: `avg_sec/object`, `std_sec/object`
- GPU memory: `peak_mem_MB` (approx)
- Qualitative: 3–6 representative face meshes

**Artifacts**

- `outputs/designC/benchmark/runtime_table.csv`
- `outputs/designC/benchmark/summary.json`
- `docs/a_b_c_comparison_table.md`

---

## C6. Poster-Ready Outputs (faces!)

**Tasks**

- [ ] Select 6–12 FaceScape outputs (good diversity)
- [ ] Render consistent views (front/side/3/4)
- [ ] Create a montage: input images + output meshes

**Artifacts**

- `outputs/designC/poster_figs/facescape_results_grid.png`
- `outputs/designC/poster_figs/pipeline_designC.png`
- `outputs/designC/poster_figs/a_b_c_runtime_chart.png`

---

## C7. Chapter 4.2 Write-Up Structure (Design C)

Include:

- **Objective:** adapt to faces / modern environment stability
- **Requirements/Constraints:** Ubuntu 22.04, modern CUDA, limited time, FaceScape format
- **Model Spec:** chosen variant, inputs/outputs, main components
- **Verification:** tests T1–T4 + sample outputs
- **Comparison:** A/B/C table (runtime + qualitative)

**Artifacts**

- `docs/ch4_designC_spec_and_verification.md`

---

## C8. Risk Register & Mitigations

**Risk:** FaceScape preprocessing/camera alignment mismatch  
**Mitigation:** start with fixed-view subset; validate projection using toolkit; sanity renders

**Risk:** TF1 GPU incompatibility (Variant 1)  
**Mitigation:** fallback to Variant 2 modern baseline; document constraint as design decision

**Risk:** Training too slow / unstable  
**Mitigation:** inference-first; tiny fine-tune only if stable; prioritize poster visuals

---

## C9. Final Deliverables Checklist (Design C)

- [ ] FaceScape subset + loaders + documented structure
- [ ] Working CUDA pipeline producing face meshes
- [ ] Verification logs + mesh outputs + renders
- [ ] Benchmark results for Design C
- [ ] A/B/C comparison table and figures
- [ ] Chapter 4.2 Design C write-up

---

## Suggested Timeline (compact)

- **Day 1:** Choose Variant + prepare FaceScape subset + loader
- **Day 2:** Make pipeline run end-to-end on GPU (export meshes)
- **Day 3:** Benchmarks + verification + poster figures + write-up draft
