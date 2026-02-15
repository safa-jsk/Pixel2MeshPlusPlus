# Chapter 4.2: Preliminary Design and Model Specification

## Pixel2Mesh++ Inference System Specification for Multi-View 3D Reconstruction

---

## 4.2.1 System Overview and Design Scope

**System Purpose:**

The Pixel2Mesh++ inference system reconstructs 3D object meshes from multi-view 2D images using a template-based deformation approach. The system takes as input three 224×224 RGB images of an object from different camera viewpoints and produces as output a single 3D mesh in standard `.obj` format, which can be visualized, analyzed, or further processed downstream.

**Design Scope:**

This chapter specifies the system design across three alternative implementations:

- **Design A (Baseline)**: Legacy TensorFlow 1.15.0 CPU-only implementation using official pretrained models on shared CPU hardware. This design establishes functional correctness and performance baseline.

- **Design B (GPU-Accelerated)**: Modernized PyTorch 2.0.1 implementation leveraging NVIDIA CUDA for GPU-accelerated inference on RTX 4070, maintaining identical model functionality and data inputs/outputs as Design A while achieving substantial performance improvement.

- **Design C (Domain Extension)**: Planned extension to integrate FaceScape human face dataset and evaluate generalization, robustness, and preprocessing requirements. Design C has not been executed at the time of this writing and is included as a forward-looking specification.

This chapter focuses on the implemented designs (A and B), with a preliminary specification for Design C. Designs A and B share identical model architecture and inference logic; they differ in framework (TensorFlow vs. PyTorch), compute target (CPU vs. GPU), and deployment environment.

---

## 4.2.2 Functional and Non-Functional Requirements

### Functional Requirements (FR)

| **ID** | **Requirement**                                                                                                                | **Acceptance Criteria**                                                                        | **Design(s)** |
|--------|--------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|---------------|
| FR1    | Load multi-view image inputs (3 views, 224×224 RGB) and metadata (.dat files with camera poses)                               | All input images load without errors; metadata parsed correctly                               | A, B, C       |
| FR2    | Execute Stage 1 (coarse mesh generation) using pretrained MVP2M checkpoint                                                    | Coarse mesh generated; output saved as `*_predict.xyz`; vertex count = 2466                   | A, B          |
| FR3    | Execute Stage 2 (mesh refinement) using pretrained P2MPP checkpoint with Stage 1 output                                       | Refined mesh generated; output saved as `.obj` file; geometry valid (no NaN/Inf)              | A, B          |
| FR4    | Produce consistent, reproducible output across multiple runs on same input                                                     | Repeated inference on identical input yields identical or near-identical output                | A, B, C       |
| FR5    | Log execution metadata (sample ID, view indices, processing timestamps, intermediate timings)                                 | Timing logs saved to text files; per-stage timing recorded                                     | A, B          |
| FR6    | Handle 35-sample evaluation subset (6 ShapeNet categories) without error                                                       | 35/35 samples complete successfully; all output artifacts generated                           | A, B          |
| FR7    | Support batch processing (Design B only) or sequential processing (Design A)                                                  | System scales to process multiple samples with consistent output quality                      | A, B          |
| FR8    | Export final mesh in `.obj` format suitable for visualization and analysis                                                    | Output `.obj` file parseable by standard 3D visualization tools                               | A, B, C       |
| FR9    | Support Design C FaceScape dataset with modified preprocessing (to be implemented)                                             | System accepts FaceScape images/metadata and produces valid meshes or error reports            | C (future)    |

### Non-Functional Requirements (NFR)

| **ID** | **Requirement**                                                                                                                | **Specification**                                                                             | **Design(s)** |
|--------|--------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|---------------|
| NFR1   | Inference latency (inference-only, no I/O)                                                                                     | Design A: ≤ 199ms/sample; Design B: ≤ 10ms/sample; Design C: TBD                             | A, B, C       |
| NFR2   | End-to-end pipeline latency (including I/O and preprocessing)                                                                  | Design A: ≤ 3.10s/sample; Design B: ≤ 100ms/sample target; Design C: TBD                     | A, B, C       |
| NFR3   | Memory footprint (inference)                                                                                                   | Design A: ≤ 3.7 GB RAM; Design B: ≤ 4 GB GPU VRAM; Design C: TBD                             | A, B, C       |
| NFR4   | Reproducibility: deterministic output across runs within floating-point tolerance                                              | Same input → same output (CPU deterministic); GPU variance < 1e-5 in vertex positions          | A, B, C       |
| NFR5   | Portability: all dependencies containerized (Docker)                                                                           | Design A: Docker image p2mpp:cpu on Ubuntu 18.04.3; Design B: Docker image p2mpp-pytorch:gpu | A, B          |
| NFR6   | Scalability: pipeline supports evaluation subsets ranging from 1 to 1000+ samples                                              | Linear or sub-linear time growth as sample count increases; memory usage predictable          | A, B, C       |
| NFR7   | Quality preservation: Design B output qualitatively identical to Design A                                                      | Visual inspection and mesh topology checks confirm no degradation; vertex count identical     | B              |
| NFR8   | Timing measurement accuracy (for performance profiling)                                                                        | Per-stage timing standard deviation ≤ 10% of mean; wall-clock measurement confidence          | A, B          |
| NFR9   | Extensibility: system design supports new datasets (e.g., FaceScape) without architectural modification                        | Data loader abstracted; preprocessing pipeline parameterizable by category/dataset             | A, B, C       |

---

## 4.2.3 Data Specification and Interfaces

### Input Data Formats

**Multi-View Images:**

Each sample consists of three rendered views at fixed indices (0, 6, 7), following the ShapeNet rendering convention. Each view is a PNG image with dimensions 224×224 RGB (no alpha channel).

- **Format**: PNG, 8-bit RGB
- **Resolution**: 224×224 pixels
- **Color Space**: sRGB (gamma-corrected)
- **Source Path**: `data/ShapeNetImages/ShapeNetRendering/{category_id}/{model_id}/rendering/render_{view_index:02d}.png`

**Metadata and Camera Poses:**

Each sample includes a `.dat` file containing preprocessed camera calibration matrices and ground-truth labels.

- **Format**: Binary NumPy `.dat` file (TensorFlow/numpy compatible)
- **Contents**: Camera intrinsics, extrinsics, object category label, model identity
- **Source Path**: `data/p2mppdata/test/{category_id}_{model_id}_00.dat`

### Dataset Subset Definition

The evaluation subset used for Designs A and B comprises 35 samples spanning 6 ShapeNet object categories:

| **Category ID** | **Category Name** | **Sample Count** | **% of Subset** |
|-----------------|-------------------|------------------|-----------------|
| 02691156        | Airplane          | 8                | 22.9%           |
| 03001627        | Chair             | 8                | 22.9%           |
| 02958343        | Car               | 6                | 17.1%           |
| 04379243        | Table             | 6                | 17.1%           |
| 03636649        | Loudspeaker       | 4                | 11.4%           |
| 03691459        | Lamp              | 3                | 8.6%            |
| **Total**       | **—**             | **35**           | **100.0%**      |

**Rationale**: The subset covers diverse geometric complexities (symmetric objects, articulated structures, smooth surfaces, fine details) and is tractable for manual quality inspection while remaining statistically representative.

### Output Artifact Formats

**Stage 1 Output (Coarse Mesh):**

- **Format**: ASCII XYZ point cloud (3D coordinates, one vertex per line)
- **Filename Convention**: `{sample_id}_predict.xyz`
- **Location**: `outputs/designA/eval_meshes/` or `outputs/designB/eval_meshes/`
- **Structure**: 2466 lines (vertices), 3 columns (x, y, z) each
- **Example Path**: `outputs/designA/eval_meshes/02691156_d068bfa97f8407e423fc69eefd95e6d3_00_predict.xyz`

**Stage 2 Output (Final Mesh):**

- **Format**: Wavefront OBJ (vertices + faces)
- **Filename Convention**: `{sample_id}_predict.obj`
- **Location**: `outputs/designA/eval_meshes/` or `outputs/designB/eval_meshes/`
- **Structure**: OBJ file with 2466 vertices and corresponding face list
- **Example Path**: `outputs/designA/eval_meshes/02691156_d068bfa97f8407e423fc69eefd95e6d3_00_predict.obj`

**Ground Truth Mesh (for comparison):**

- **Format**: OBJ file (reference mesh for visual comparison)
- **Location**: `outputs/designA/eval_meshes/{sample_id}_gt.obj`

**Timing and Metadata Logs:**

- **Stage 1 Timing**: `outputs/designA/benchmark/stage1_timings.txt` (per-sample wall-clock time)
- **Stage 2 Timing**: `outputs/designA/benchmark/stage2_timings.txt` (per-sample wall-clock time)
- **Combined Timing**: `outputs/designA/benchmark/combined_timings.txt` (summary statistics)

### Interface Contracts

**Data Loader → Stage 1 Module:**

- **Input**: Image tensor [3, 224, 224, 3] (3 views, RGB), metadata dictionary (camera poses, labels)
- **Output Format**: Coarse mesh tensor [2466, 3] (vertex positions)
- **Contract**: Images normalized to [0, 1] float32; camera poses pre-loaded and accessible

**Stage 1 Module → Artifact Manager:**

- **Input**: Coarse mesh tensor [2466, 3]
- **Output**: Saved file `{sample_id}_predict.xyz`
- **Contract**: File must be readable by Stage 2; format must match Stage 2 expectations

**Stage 2 Module → Mesh Export:**

- **Input**: Refined mesh tensor [2466, 3]
- **Output Format**: OBJ file with faces
- **Contract**: Face connectivity available from template; output file parseable by standard viewers

---

## 4.2.4 Model Architecture Specification (Pixel2Mesh++)

### High-Level Architecture

Pixel2Mesh++ is a two-stage graph neural network for template-based 3D mesh deformation. The system follows the encoder-refiner architecture described in the ICCV 2019 paper "Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation" (Wen et al., 2019).

**Encoder**: Extracts perceptual features from multi-view RGB images using a backbone network (pretrained VGG-16 or similar convolutional feature extractor [as per official Pixel2Mesh++ baseline]). Feature maps are generated at multiple scales (224×224, 112×112, 56×56) to capture both global and local geometry cues.

**Mesh Deformation Backbone**: A fixed-topology ellipsoid template is iteratively deformed using graph convolutional operations on the vertex/edge graph structure. Each deformation step projects learned features onto the mesh surface and performs localized refinement.

### Stage 1: Coarse Mesh Generation (MVP2M)

**Purpose**: Generate an initial coarse reconstruction by deforming a template ellipsoid.

**Input**: 3-view RGB images [3, 224, 224, 3]

**Architecture**:
- Encode multi-view images into perceptual feature maps
- Initialize mesh as unit ellipsoid (2466 vertices)
- Apply graph convolution + feature projection (1–2 rounds of refinement)
- Output coarse mesh vertex positions [2466, 3]

**Pretrained Checkpoint**: `data/p2mpp_models/coarse_mvp2m/models/meshnetmvp2m.ckpt-50` (epoch 50, trained on ShapeNet)

**Assumptions**:
- Ellipsoid initialization is adequate for all object categories
- 2466-vertex template topology is universal (fixed across all samples)
- View indices 0, 6, 7 are sufficient to reconstruct object geometry

### Stage 2: Mesh Refinement (P2MPP)

**Purpose**: Refine coarse mesh via perceptual feature projection and additional graph convolutions.

**Input**: Coarse mesh [2466, 3] from Stage 1, 3-view RGB images [3, 224, 224, 3]

**Architecture**:
- Project multi-scale image features onto mesh surface via bilinear sampling
- Perform graph convolution with feature-enhanced kernels (3–4 rounds of refinement)
- Localized graph projection layer that samples perceptual features at projected vertex positions
- Output refined mesh vertex positions [2466, 3]

**Key Operations**:
- Bilinear sampling from feature maps at 3 scales: [224, 224], [112, 112], [56, 56]
- Coordinate clamping to prevent out-of-bounds access (critical fix: coordinates clamped to valid indices after scaling)
- Graph convolution on mesh laplacian
- Vertex position updates via learned deformation offsets

**Pretrained Checkpoint**: `data/p2mpp_models/refine_p2mpp/models/meshnet.ckpt-10` (epoch 10, fine-tuned on ShapeNet)

**Assumptions**:
- Bilinear feature sampling is differentiable and stable (requires clamping for GPU implementations)
- Graph structure remains fixed (no topological changes)
- Refinement improves all objects in the same template space

### Mesh Template and Topology

**Fixed Template**: Ellipsoid with 2466 vertices and fixed connectivity (faces).

**Topology Preservation**: All outputs maintain identical vertex count and face list. No mesh decimation, subdivision, or remeshing is performed.

**Output Validity Constraints**:
- No NaN or Inf values in vertex positions
- Manifold mesh structure preserved (one closed surface per object)
- Self-intersection allowed (mesh may fold on itself during early iterations)

---

## 4.2.5 Module-Level Design (Implementation Breakdown)

The system is decomposed into six key modules, each with defined responsibilities, interfaces, and error handling:

### Module Breakdown and Responsibilities

| **Module** | **Responsibilities** | **Inputs** | **Outputs** | **Key Constraints** |
|------------|---------------------|-----------|------------|-------------------|
| **Data Loader** | Load PNG images, parse .dat metadata, validate input format | Sample ID, image paths, metadata paths | Image tensor [3, 224, 224, 3], camera poses dict | All images must exist; resolution must be 224×224; .dat format fixed |
| **Stage 1 Inference** | Execute MVP2M model, forward pass, vertex position computation | Image tensor, metadata (camera poses) | Coarse mesh [2466, 3] as float32 | Checkpoint must be loaded; TF/PyTorch graph must be built correctly |
| **Artifact Manager** | Name, save, and index intermediate outputs (*.xyz files) | Stage 1 mesh, sample ID | .xyz file on disk, file path record | File paths must follow convention; no overwrites without explicit flag |
| **Stage 2 Inference** | Execute P2MPP model, load Stage 1 output, forward pass, refinement | Image tensor, Stage 1 mesh (.xyz), metadata | Refined mesh [2466, 3] as float32 | Stage 1 file must exist and be readable; checkpoint pre-loaded |
| **Mesh Export** | Convert vertex tensor to OBJ format with face connectivity | Refined mesh [2466, 3], face list (from template) | .obj file on disk | OBJ format must be standard-compliant; faces must reference valid vertex indices |
| **Timing/Profiler** | Measure wall-clock per-stage time, compute statistics, log to disk | Start/end timestamps per stage | Timing CSV/text files, summary statistics | Timing accuracy ≥ millisecond precision; no blocking I/O during measurement |

### Module Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input: Sample {ID}                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Data Loader     │
                    │ - Load images    │
                    │ - Parse metadata │
                    └────────┬─────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
    ┌─────────────────────────┐      ┌──────────────────────┐
    │ Stage 1 Inference       │      │ Timing/Profiler      │
    │ - MVP2M forward pass    │      │ - Record start_time  │
    │ - Coarse mesh (2466×3)  │      └──────────────────────┘
    └────────┬────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Artifact Manager         │
    │ - Save *_predict.xyz     │
    │ - Index output           │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ Stage 2 Inference            │
    │ - Load Stage 1 output        │
    │ - P2MPP forward pass         │
    │ - Refined mesh (2466×3)      │
    └────────┬─────────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Mesh Export              │
    │ - Generate OBJ file      │
    │ - Write to disk          │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Timing/Profiler          │
    │ - Record end_time        │
    │ - Compute latency        │
    └────────┬─────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Output: {sample_id}_predict.xyz, {sample_id}_predict.obj,       │
│         timing_log (stage1_time, stage2_time, total_time)       │
└─────────────────────────────────────────────────────────────────┘
```

### Error Handling

Each module implements fault detection and logging:

- **Data Loader**: Raises exception if image file not found, .dat not parseable, or image resolution incorrect. Logs error with sample ID and path.
- **Stage 1/2 Inference**: Catches NaN/Inf in output tensors; logs warning; optionally skips sample or saves invalid flag.
- **Artifact Manager**: Validates file write success; logs path; detects naming conflicts.
- **Mesh Export**: Validates face indices; detects self-intersections (optional warning); ensures OBJ parseable.
- **Timing/Profiler**: Detects timing anomalies (≥ 3σ outliers); logs with sample ID for post-hoc analysis.

---

## 4.2.6 Design Alternatives Specification (A vs B vs C)

### Cross-Design Comparison Table

| **Dimension** | **Design A (Baseline)** | **Design B (GPU-Accelerated)** | **Design C (FaceScape)** |
|---------------|------------------------|------------------------------|------------------------|
| **Framework** | TensorFlow 1.15.0 (static graph) | PyTorch 2.0.1 (dynamic graphs) | PyTorch 2.0.1 (same as B) |
| **Compute Target** | CPU only (Intel i5-1335U, 12 cores) | GPU (NVIDIA RTX 4070, 7168 CUDA cores) | GPU (RTX 4070) |
| **Hardware RAM/VRAM** | ~3.7 GB system RAM | 12.4 GB GPU VRAM | 12.4 GB GPU VRAM |
| **Compute Capability / Arch** | CPU native; no CUDA | Ada Lovelace (compute 8.9) | Ada Lovelace (compute 8.9) |
| **Dataset** | ShapeNet (35-sample fixed subset) | ShapeNet (35-sample fixed subset, identical to A) | FaceScape (TBD: split/scale) |
| **Model Weights** | Official pretrained (ShapeNet) | Official pretrained (ShapeNet) | Pretrained (ShapeNet) or retrained (FaceScape TBD) |
| **Docker Image** | p2mpp:cpu (Ubuntu 18.04.3, Python 3.6.8) | p2mpp-pytorch:gpu (Ubuntu 20.04, Python 3.10, CUDA 11.7) | p2mpp-pytorch:gpu (same as B) |
| **Python Version** | 3.6.8 | 3.10 | 3.10 |
| **Inference Latency (35 samples)** | 6.96s ± 0.12s (~199ms/sample) | 0.28s ± 0.01s (~8ms/sample) | TBD (depends on preprocessing) |
| **Throughput** | ~5 samples/second | ~125 samples/second | TBD |
| **Speedup vs A** | 1.0× (baseline) | 24.85× | TBD |
| **Expected Quality** | Ground truth (baseline) | Identical to A (35/35 verified) | TBD (domain-dependent) |
| **Reproducibility** | Full (CPU deterministic) | High (GPU variance < 1e-5) | TBD |
| **Status** | ✅ Complete, verified | ✅ Complete, verified | 🔄 Planned (not executed) |

### Design Rationale

**Design A** establishes the functional and performance baseline, answering: "Does the official Pixel2Mesh++ implementation work on modern hardware?" This design is essential for validation and provides ground truth for quality comparisons.

**Design B** answers: "Can we achieve substantial speedup through modern hardware acceleration (GPU) while maintaining output quality?" By keeping the dataset and evaluation protocol identical to Design A, we isolate the effect of computational acceleration from data effects.

**Design C** addresses generalization: "Does GPU-accelerated Pixel2Mesh++ work on domain-shifted data (human faces)?" This design will reveal whether the template-based approach and pretrained weights transfer to a different semantic domain.

---

## 4.2.7 Reproducibility and Deployment Specification

### Docker-Based Reproducibility Strategy

Both Design A and Design B employ Docker containerization to ensure reproducibility across different host systems.

**Design A (CPU Baseline):**

```dockerfile
Image: p2mpp:cpu
Base: ubuntu:18.04 (Ubuntu 18.04.3 LTS)
Python: 3.6.8
TensorFlow: 1.15.0 (CPU-only)
Key libraries: numpy 1.17.3, opencv-python 4.5.5, networkx, scipy
```

**Design B (GPU-Accelerated):**

```dockerfile
Image: p2mpp-pytorch:gpu
Base: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime (or equivalent)
Python: 3.10
PyTorch: 2.0.1
CUDA: 11.7, cuDNN: 8
Key libraries: torch-vision, torch3d 0.7.4, scipy, scikit-image, matplotlib
```

### Dependency Pinning

All dependencies are version-pinned in `requirements.txt` (per-design) to enable exact reproducibility:

- **Design A**: `requirements_cpu.txt` with frozen TensorFlow 1.15.0 and compatible libraries
- **Design B**: `requirements_gpu.txt` with frozen PyTorch 2.0.1, CUDA 11.7, and compatible libraries

Version conflicts are resolved before Docker build; no runtime package installations are performed.

### Hardware Assumptions and Constraints

**Design A Assumptions:**
- Multi-core CPU (6+ cores) for reasonable latency
- Minimum 4 GB RAM available for inference
- No GPU required; CPU-only computation
- Standard Linux kernel (Docker-compatible)

**Design B Assumptions:**
- NVIDIA GPU with compute capability ≥ 8.0 (Ada or newer preferred; older architectures may face cuSolver compatibility issues)
- Minimum 4 GB GPU VRAM; 6+ GB recommended for batch processing
- NVIDIA Docker runtime installed on host (nvidia-docker or docker with --gpus flag)
- NVIDIA driver ≥ 450 for CUDA 11.7 compatibility

### Deployment and Setup Documentation

Complete setup documentation is provided separately:

- **Design A**: `env/cpu/QUICKSTART.md` with Docker build and run commands
- **Design B**: `env/gpu/SETUP_GUIDE.md` with GPU setup, verification, and troubleshooting
- **Common**: `README.md` with high-level overview and hyperlinks to design-specific guides

---

## 4.2.8 Verification Hooks and Test Plan Mapping

### Fixed Test Harness

The evaluation uses a fixed 35-sample test subset (defined in Section 4.2.3) to enable reproducible benchmarking and quality assessment.

**Test Harness Configuration:**

- **Sample List File**: `designA/designA_eval_list.txt` (35 samples, newline-separated)
- **Category Distribution**: As specified in Section 4.2.3 table
- **Output Root**: `outputs/designA/eval_meshes/` (Design A), `outputs/designB/eval_meshes/` (Design B)

### Output Artifact Verification

Each generated artifact is checked for validity before quality assessment:

- **File Existence**: All expected `.xyz` and `.obj` files present
- **File Non-Emptiness**: File size > 0; no empty/truncated outputs
- **Format Validity**: XYZ files parse as float32 arrays [2466, 3]; OBJ files parse as standard Wavefront OBJ
- **Geometric Validity**: No NaN, Inf, or degenerate values in vertex positions
- **Vertex Count**: Exactly 2466 vertices per mesh (topology preserved)
- **Face Connectivity**: All face indices valid (reference vertices 0..2465)

### Stage-Wise Timing Integration Points

Timing measurements are collected at module boundaries:

1. **Pre-Stage 1**: Record wall-clock time before data loading
2. **Post-Stage 1**: Record time after coarse mesh generation; log stage 1 elapsed time
3. **Post-Stage 2**: Record time after refinement; log stage 2 elapsed time
4. **Post-Export**: Record final time; compute total end-to-end latency

**Timing Logs Output:**

- `outputs/designA/benchmark/stage1_timings.txt`: Per-sample Stage 1 latency (35 rows)
- `outputs/designA/benchmark/stage2_timings.txt`: Per-sample Stage 2 latency (35 rows)
- `outputs/designA/benchmark/combined_timings.txt`: Summary (total time, average, std dev)

### Quality Equivalence Checks

**Designs A and B are expected to produce numerically identical or near-identical outputs.** Quality is verified through:

1. **Qualitative Visual Inspection**: Sample outputs from Design A and B are visually compared side-by-side. Meshes should appear indistinguishable to human observers.

2. **Structural Validation**: Both designs produce identical vertex counts (2466) and face connectivity. This is automatically checked.

3. **Numerical Precision Check**: For floating-point comparisons, vertex positions are compared with tolerance (default: absolute error < 0.001 in meters). GPU variance is expected due to CUDA non-determinism; this tolerance accounts for it.

4. **Mesh Validity Consistency**: Both designs produce valid, non-degenerate meshes with no self-intersections or topology errors.

**Outcome Reporting**: Design A and B both achieve **35/35 successful completion** and **35/35 quality match** (verified).

---

## 4.2.9 Design C Preliminary Specification (FaceScape Plan)

### Scope and Objectives

Design C extends the GPU-accelerated pipeline (Design B) to a new semantic domain: high-resolution 3D human face scans from the FaceScape dataset. This design answers whether Pixel2Mesh++, originally trained on rigid objects (ShapeNet), can generalize to human face reconstruction without retraining.

**Research Questions:**
- Does the ShapeNet-pretrained model produce reasonable face meshes on FaceScape inputs?
- If pretrained weights are insufficient, how much retraining is needed for acceptable quality?
- What preprocessing steps are necessary to adapt FaceScape images/metadata to the Pixel2Mesh++ input format?

### FaceScape Dataset and Preprocessing

**FaceScape Overview** [placeholder: exact dataset structure TBD from official FaceScape resources]:

- **Source**: Large-scale 3D human face database with high-quality scans (multi-view images, registered meshes, expression/identity variations)
- **Typical Input**: Multi-view RGB images of human faces (e.g., 20+ views per subject, variable resolution) + face landmarks/masks
- **Challenge**: Pixel2Mesh++ expects fixed 3 views at specific angles; FaceScape may have different view configurations

**Preprocessing Requirements** (Conceptual):

1. **View Selection**: Extract 3 canonical views per subject (e.g., frontal + 2 diagonal) that approximate the ShapeNet view convention (indices 0, 6, 7 equivalent).

2. **Image Cropping/Alignment**: Normalize face pose using facial landmarks. Crop to focus on face region; resize to 224×224.

3. **Background Removal**: FaceScape images may have non-black backgrounds. Apply face mask or background removal to isolate face geometry (optional, depends on model robustness).

4. **Intensity Normalization**: Normalize image intensity to match ShapeNet preprocessing (gamma correction, per-channel normalization [to be filled from official preprocessing]).

5. **Metadata Creation**: Generate `.dat` files compatible with existing data loader, including estimated camera intrinsics/extrinsics for each view.

### Model Retraining Strategy (if necessary)

If pretrained ShapeNet weights prove inadequate for face reconstruction:

**Option 1: Fine-Tuning**
- Keep Stage 1 (coarse) weights; retrain Stage 2 (refinement) on FaceScape data
- Expected benefit: Refiner learns face-specific geometry patterns
- Data requirement: [To be filled from FaceScape size]

**Option 2: Full Retraining**
- Retrain both Stage 1 and Stage 2 from scratch (or ImageNet initialization) on FaceScape
- Expected benefit: Model fully adapts to face template and topology
- Data requirement: Large labeled FaceScape training split

**Option 3: Domain Adaptation**
- Use ShapeNet-pretrained weights with minimal fine-tuning (few epochs) on FaceScape
- Expected benefit: Balance transfer learning and domain adaptation
- Data requirement: Small labeled FaceScape set for validation

### Evaluation Plan for Design C

**Test/Validation Split**:
- Reserve [TBD: X%] of FaceScape samples for validation during retraining (if applicable)
- Reserve [TBD: Y%] for final evaluation (same size as 35-sample ShapeNet subset for fair comparison)

**Metrics**:
- Mesh validity (same checks as Designs A/B)
- Inference latency (expected: similar to Design B if no architectural changes)
- Qualitative quality (visual inspection of face reconstruction fidelity, landmark alignment)
- Comparison to ground-truth FaceScape meshes (if available; compute Chamfer distance or similar metric [To be filled from evaluation methodology])

### Risks and Mitigation

| **Risk** | **Impact** | **Mitigation** |
|----------|-----------|----------------|
| **Domain Gap**: Pretrained ShapeNet weights produce invalid face meshes | High | Start with qualitative inspection on 5–10 samples; assess feasibility before full retraining |
| **Template Mismatch**: Face topology differs from object template (e.g., different vertex density in eyes/mouth) | Medium | Retrain Stage 1 to deform template to face shape; accept topology variations if necessary |
| **Data Preprocessing**: FaceScape image format/resolution differ from ShapeNet | Medium | Carefully align preprocessing pipeline; create test script to validate 5 preprocessed samples before full pipeline |
| **Computational Cost**: Retraining Stage 1/2 on large FaceScape dataset may exceed available GPU VRAM | Medium | Implement gradient checkpointing; use smaller batch sizes; consider multi-GPU if available |
| **Licensing/Access**: FaceScape dataset may have restricted access or license terms | High | Verify dataset availability and licensing before implementation; plan alternative if necessary |

### Design C Output Specification

If executed, Design C will produce:

- Preprocessed FaceScape test subset (images + metadata) in `data/facespace_subset/`
- Inference outputs: `outputs/designC/eval_meshes/{face_id}_predict.xyz` and `{face_id}_predict.obj`
- Timing logs: `outputs/designC/benchmark/` (similar structure to Design A/B)
- Retraining logs (if applicable): `outputs/designC/training_logs/` with loss curves and checkpoint files
- Quality assessment report: `docs/designC_evaluation_report.md` documenting domain adaptation success and lessons learned

---

## Summary

This chapter has specified the system design and model architecture for Pixel2Mesh++ inference across three design alternatives. Designs A and B share identical model logic but differ in framework and compute target, enabling fair performance comparison while maintaining output quality equivalence. Design C, planned as a forward extension, defines a specification for domain generalization to human faces. The modular design, comprehensive data interfaces, and verification hooks enable reproducible benchmarking and support straightforward extension to new datasets and implementations.

