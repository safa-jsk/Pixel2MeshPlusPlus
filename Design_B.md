# Design B Roadmap — Pixel2Mesh++ CUDA Hotspot Acceleration (Legacy Model)

**Goal:** Keep the **same pretrained model + same inputs** as Design A, but enable/accelerate **GPU execution on key hotspots** to reduce inference (and/or training) time.  
**Key rule:** _Do not change the architecture._ Only change execution/performance components so A vs B is a fair comparison.

---

## B0. Success Criteria (what “done” means)

By the end of Design B, you should have:

- ✅ The same inference pipeline as Design A still works (same config, same eval list)
- ✅ GPU is actually being used for at least one major bottleneck (verified by logs + `nvidia-smi`)
- ✅ A vs B benchmark table (sec/object, throughput, and basic GPU utilization evidence)
- ✅ Mesh outputs remain qualitatively similar to Design A (no obvious degradation)
- ✅ Clear documentation of what changed (files/commits) and why those are “hotspots”

---

## B1. Branching & Traceability

**Tasks**

- [ ] Create branch: `design-b-cuda-hotspots` from the exact Design A commit
- [ ] Tag the Design A baseline commit (or record hash)
- [ ] Maintain a `CHANGELOG.md` entry listing every file changed

**Artifacts**

- `docs/designB_commit.txt` (commit hash, date)
- `docs/designB_changes.md` (what changed + reasoning)

---

## B2. Decide the GPU Strategy (minimum viable)

Pixel2Mesh++ commonly has a bottleneck in nearest-neighbor / Chamfer distance computations (often implemented as a custom op).  
Design B focuses on enabling GPU where it matters most, without rewriting the model.

### GPU Strategy Options (pick one primary)

- **B-Path-1 (Preferred):** Build/enable the **GPU custom op** used by the project (e.g., NN distance / Chamfer)
- **B-Path-2:** If GPU op is not feasible: use a **known compatible alternative op** (still same math, just different implementation)
- **B-Path-3:** If TF1 GPU is blocked by hardware/software constraints: use Design B as **“CPU baseline + micro-optimizations”** and move “real GPU” to Design C (documented constraint)

**Decision Artifact**

- `docs/designB_gpu_strategy.md` (chosen path + why)

---

## B3. Set Up GPU-Capable Environment (reproducible)

> Keep this isolated from Design A env. Use a separate container/env folder.

**Tasks**

- [ ] Create GPU container/env that matches TensorFlow 1.x GPU requirements as closely as possible
- [ ] Verify GPU is visible:
  - `nvidia-smi` inside env
  - simple TF test: list physical devices / check CUDA availability

**Artifacts**

- `env/gpu/Dockerfile` or `env/gpu/conda_env.yml`
- `env/gpu/requirements_freeze.txt`
- `outputs/designB/benchmark/system_info.txt` (GPU name, driver, CUDA toolkit version)

**Verification Evidence**

- screenshot/log of `nvidia-smi`
- log snippet showing TF detects GPU (or explicit note if it cannot)

---

## B4. Hotspot 1 — GPU NN-Distance / Chamfer Op

### Step 1: Locate the op and build instructions

**Tasks**

- [ ] Identify where the repo loads the `.so` op (e.g., `external/` folder)
- [ ] Build the op for your CUDA/toolchain OR use an officially provided compatible binary
- [ ] Ensure the correct `.so` is used at runtime

**Verification**

- [ ] Running inference no longer errors on missing op
- [ ] Op loads successfully (log line or no import error)
- [ ] GPU utilization spikes during distance computation (observe in `nvidia-smi`)

**Artifacts**

- `external/` built `.so` (tracked via build script, not necessarily committed)
- `env/gpu/build_ops.sh`
- `outputs/designB/logs/op_load_log.txt`

---

## B5. Hotspot 2 — GPU Placement & Execution (no architecture changes)

### Confirm the model actually runs on GPU

**Tasks**

- [ ] Set environment vars for TF logging (optional) to inspect device placement
- [ ] Run the same eval list as Design A using the same command/config
- [ ] Confirm key ops are placed on GPU

**Verification**

- [ ] `nvidia-smi` shows non-zero memory usage during run
- [ ] Optional: TF device placement logs show GPU usage

**Artifacts**

- `outputs/designB/logs/device_placement.txt` (if enabled)
- `outputs/designB/logs/inference_log.txt`

---

## B6. Benchmarking Plan (A vs B must be apples-to-apples)

Use exactly the same:

- evaluation subset list (`data/designA_eval_list.txt`)
- pretrained weights
- inference script/config
- measurement protocol (warmup + N runs)

### Metrics to collect

- `avg_sec_per_object`
- `throughput_obj_per_min` (optional)
- `peak_gpu_mem_MB` (approx via `nvidia-smi`)
- (optional) CPU utilization snapshot

**Tasks**

- [ ] Run 1 warmup pass (not counted)
- [ ] Run 3 benchmark passes, record times
- [ ] Save raw timings and compute mean/std

**Artifacts**

- `outputs/designB/benchmark/runtime_table.csv`
- `outputs/designB/benchmark/summary.json`
- `outputs/designB/benchmark/nvidia_smi_log.txt`

---

## B7. Output Consistency Check (quality sanity)

Design B must not “break” output quality.

**Tasks**

- [ ] Compare 5–10 meshes from A vs B (same sample IDs)
- [ ] Visual comparison: overlay or side-by-side renders
- [ ] Optional numeric check: vertex count matches, mesh not corrupted

**Artifacts**

- `outputs/designB/quality_check/compare_grid.png`
- `outputs/designB/quality_check/notes.md`

---

## B8. Poster/Report Artifacts for Design B

### What to include in Chapter 4 / poster

- A simple diagram: “Baseline pipeline + GPU hotspot acceleration”
- A table: A vs B runtime
- Evidence: GPU utilization screenshot + 2–3 qualitative mesh results

**Artifacts**

- `docs/ch4_designB_spec_and_verification.md`
- `outputs/designB/poster_figs/a_vs_b_runtime_table.png` (or export from csv)
- `outputs/designB/poster_figs/gpu_usage.png`
- `outputs/designB/poster_figs/results_side_by_side.png`

---

## B9. Risk Register (keep concise)

**Risk:** TF1 GPU incompatibility with modern drivers/hardware  
**Mitigation:** containerize legacy CUDA stack; if blocked, document constraint and shift GPU-heavy work to Design C

**Risk:** Custom op fails to compile due to CUDA/toolchain mismatch  
**Mitigation:** test compile in isolation; pin versions; consider alternate op implementation

**Risk:** Speedup minimal due to remaining CPU bottlenecks  
**Mitigation:** profile runtime; confirm which portion dominates; add minor improvements (batching, I/O) only if they don’t change model behavior

---

## B10. Final Deliverables Checklist (Design B)

- [ ] GPU-capable env setup files (`env/gpu/...`)
- [ ] Custom op build script + evidence of successful load
- [ ] Inference works on the same subset list
- [ ] A vs B benchmark table (mean/std)
- [ ] Side-by-side quality check images
- [ ] Chapter 4 Design B write-up (spec + verification + results)

---

## Suggested Timeline (realistic)

- **Day 1:** GPU env + TF detects GPU (or document constraint)
- **Day 2:** Build/load custom op + run inference once
- **Day 3:** Benchmark A vs B + quality check + poster figures

---

## Handoff to Design C

Design C may introduce limited modernization (data pipeline, code structure, or a modern baseline model).  
Design B should remain strictly “same model, faster execution,” so your thesis can clearly compare:

- **A:** works (CPU)
- **B:** works + faster (CUDA hotspots)
- **C:** works + modern/maintainable and/or better suited for FaceScape
