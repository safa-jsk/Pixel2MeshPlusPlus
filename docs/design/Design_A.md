# Design A Roadmap — Pixel2Mesh++ Baseline (Legacy, CPU)

**Goal:** Reproduce the official Pixel2Mesh++ pretrained inference pipeline (ShapeNet) on **CPU** and generate **mesh outputs + baseline performance numbers** for Chapter 4 and the poster.

---

## A0. Success Criteria (what “done” means)

By the end of Design A, you should have:

- ✅ A reproducible environment setup (scripted commands + exact versions noted)
- ✅ Pretrained weights downloaded and placed correctly
- ✅ Demo/inference runs on CPU without crashing
- ✅ Predicted meshes exported as `.obj` (and optionally rendered screenshots)
- ✅ Baseline runtime numbers (sec/sample or sec/object) on a fixed subset
- ✅ A short “Functional Verification” section with evidence (logs + outputs)

---

## A1. Repository Setup

### 1. Clone and lock the baseline

- Clone your fork and keep a clean baseline branch.

**Tasks**

- [ ] `git clone <your-fork-url>`
- [ ] `cd Pixel2MeshPlusPlus`
- [ ] Create branch: `design-a-baseline`
- [ ] Record commit hash for the thesis report

**Artifacts**

- `docs/designA_commit.txt` (commit hash, date)

---

## A2. Environment (CPU-first, reproducible)

> **Rule:** CPU first. Do not chase CUDA issues in Design A.

### Option A (Recommended): Docker/Container (legacy userland)

**Tasks**

- [ ] Create a `Dockerfile` or use a legacy base image
- [ ] Install Python (legacy version required by repo) + TensorFlow 1.x (CPU)
- [ ] Install required Python packages from repo `requirements.txt` (if present)

**Artifacts**

- `env/Dockerfile`
- `env/run_container.sh`
- `env/requirements_freeze.txt` (pip freeze output)

### Option B: Conda (if you avoid Docker)

**Tasks**

- [ ] Create conda env (legacy Python)
- [ ] Install TF1 CPU + deps
- [ ] Freeze package list

**Artifacts**

- `env/conda_env.yml` (exported)
- `env/requirements_freeze.txt`

---

## A3. Get the Pretrained ShapeNet Weights

**Tasks**

- [ ] Download pretrained weights from the official Google Drive link
- [ ] Place weights in the expected repository path (as per README)
- [ ] Confirm checkpoint files are discoverable by the code

**Verification**

- [ ] A small script or log line confirming checkpoints load successfully

**Artifacts**

- `weights/README.md` (where weights came from + file list + folder structure)
- Screenshot/log snippet showing “loaded checkpoint …”

---

## A4. Sanity Run: Demo Inference (CPU)

### Step 1: Run the demo exactly as intended

**Tasks**

- [ ] Run `python demo.py` (or the repo’s documented demo command)
- [ ] Ensure output mesh is created (typically `.obj`)

**Verification Checklist**

- [ ] Program completes without exception
- [ ] Output mesh file exists and is non-empty
- [ ] Mesh can be opened in MeshLab/Blender

**Artifacts**

- `outputs/designA/demo/predict.obj`
- `outputs/designA/demo/demo_log.txt`
- `outputs/designA/demo/mesh_preview.png` (screenshot from MeshLab/Blender)

---

## A5. Baseline Inference on a Fixed Subset (ShapeNet)

> Keep the dataset scope small for Part-2 verification and poster.

### Define a repeatable evaluation subset

**Tasks**

- [ ] Create `data/designA_eval_list.txt` (e.g., 20–100 objects)
- [ ] Ensure paths match what the code expects

**Verification**

- [ ] Running inference on this list produces one `.obj` per instance (or consistent outputs)

**Artifacts**

- `data/designA_eval_list.txt`
- `outputs/designA/eval_meshes/` (folder of meshes)

---

## A6. Runtime Measurement (Baseline)

Measure performance in a simple, defensible way.

### Minimal metrics

- **sec/object** (end-to-end inference time)
- **avg sec/object** over N samples
- **hardware info** (CPU model, RAM, OS)

**Tasks**

- [ ] Add a timing wrapper (or use existing logs)
- [ ] Run 3 trials for N objects and average

**Artifacts**

- `outputs/designA/benchmark/runtime_table.csv`
- `outputs/designA/benchmark/system_info.txt`
- `outputs/designA/benchmark/benchmark_log.txt`

**Example runtime_table.csv columns**

- `sample_id, time_sec, notes`
- `avg_time_sec, std_time_sec, n_samples`

---

## A7. Visual Results for Poster

Pick a small set of best/representative meshes.

**Tasks**

- [ ] Select 6–12 results for the poster
- [ ] Produce consistent renders (same camera angle & lighting) or screenshots
- [ ] Include input image + output mesh snapshot pairs

**Artifacts**

- `outputs/designA/poster_figs/result_01_input.png`
- `outputs/designA/poster_figs/result_01_mesh.png`
- `outputs/designA/poster_figs/contact_sheet.png` (optional montage)

---

## A8. Design A “Functional Verification” Write-Up (for Chapter 4)

### What to write (short, clear, evidence-based)

- Environment specification (versions + OS)
- Model source: pretrained weights on ShapeNet
- Pipeline steps: input → network → mesh output
- Verification evidence: logs + output files + screenshots
- Baseline performance table (CPU)

**Artifacts**

- `docs/ch4_designA_functional_verification.md`  
  Include:
  - Table: environment specs
  - Table: runtime baseline
  - Figure: pipeline diagram (simple)
  - Figure: qualitative mesh outputs

---

## A9. Risks & Mitigations (keep it short in report)

**Risk:** TF1 legacy dependency conflicts  
**Mitigation:** containerize + freeze versions

**Risk:** custom op missing (e.g., chamfer op) even in CPU mode  
**Mitigation:** document whether demo requires it; if required, use CPU fallback or included binary (do not CUDA-compile in Design A)

**Risk:** output quality inconsistent  
**Mitigation:** stick to official demo inputs + pretrained config first

---

## A10. Final Deliverables Checklist (Design A)

- [ ] Reproducible env files (`Dockerfile` or `conda_env.yml`)
- [ ] Pretrained weights placed + documented
- [ ] Demo run output `.obj` and screenshot
- [ ] Fixed evaluation subset outputs
- [ ] Runtime baseline table (`.csv`)
- [ ] Chapter 4 Design A verification markdown

---

## Suggested Timeline (fast but realistic)

- **Day 1:** Environment + weights + demo runs
- **Day 2:** Fixed subset inference + runtime measurement
- **Day 3:** Poster-ready renders + Chapter 4 Design A write-up draft

---

## Handoff to Design B

Design B will reuse:

- the same evaluation subset list
- the same inference command/config
- the same metrics
  and will change **only** the performance-critical components (CUDA hotspots), so A vs B is a fair comparison.
