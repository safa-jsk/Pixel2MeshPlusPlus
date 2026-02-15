# Chapter 4.1: Design Process and Methodology Overview

## Pixel2Mesh++ Implementation and Benchmarking Framework

---

## 4.1.1 Design Objectives and Success Criteria

The primary objective of Part-2 is to establish a rigorous benchmarking framework for the Pixel2Mesh++ 3D mesh reconstruction system under multiple hardware and software configurations, with the goal of quantifying performance gains and validating implementation correctness across designs. Pixel2Mesh++ is a deep learning model that reconstructs 3D object meshes from multi-view 2D images, following the ICCV 2019 paper "Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation." The system employs a two-stage pipeline: Stage 1 (MVP2M) generates a coarse mesh through graph convolutional deformation of an initial ellipsoid template, and Stage 2 (P2MPP) refines the mesh using perceptual feature projection and additional deformations.

**Design Objectives** are defined as follows:

1. **Baseline Correctness**: Establish a reproducible baseline implementation (Design A) using the official pretrained models and legacy TensorFlow 1.15 code, confirming that the system produces valid 3D mesh outputs without errors.

2. **Reproducibility**: Implement comprehensive experimental logging and deterministic execution protocols to ensure that results can be independently verified and compared across design iterations.

3. **GPU Acceleration**: Modernize the inference pipeline (Design B) to leverage NVIDIA GPU hardware (RTX 4070), demonstrating substantial acceleration while preserving output quality.

4. **Quality Preservation**: Validate that modernization and GPU acceleration do not degrade mesh reconstruction quality, as measured by vertex count consistency, mesh topology preservation, and visual equivalence.

**Success Criteria** are quantifiable and measurable:

- **Functional Completion**: Both Design A and Design B must successfully process 100% of the fixed evaluation subset (35 samples) without runtime errors or crashes.
- **Mesh Validity**: All output meshes must preserve the original topology (2466 vertices) and exhibit no degenerate geometry.
- **Timing Metrics**: Inference-only timing must be measured with precision (standard deviation ≤ 10% of mean) across multiple runs.
- **Fair Comparison**: CPU and GPU implementations must operate on identical inputs (same samples, images, views, resolution), with any overhead clearly separated from inference time.
- **Speedup Demonstration**: Design B must achieve at least a 2× speedup relative to Design A to justify the modernization effort.

---

## 4.1.2 Requirements, Specifications, and Constraints

The implementation is constrained by several technical, resource, and methodological factors that shaped the design of the evaluation framework.

**Hardware Constraints:**

Design A operates entirely on CPU hardware (Intel Core i5-1335U, 12 cores, 3.7GB RAM) with no GPU acceleration, reflecting the computational environment of the legacy code. Design B targets a single NVIDIA RTX 4070 GPU (Ada Lovelace architecture, compute capability 8.9, 12.4GB VRAM), representing a modernized acceleration platform. These hardware differences are intentional—they isolate system-level improvements from framework-level optimizations.

**Software Constraints:**

Design A is constrained to TensorFlow 1.15.0 (released 2017) running on Python 3.6.8 within a Docker container based on Ubuntu 18.04.3, reflecting the original development environment of Pixel2Mesh++. This legacy stack offers stability and backward compatibility but lacks modern GPU support. Design B required a framework migration due to hardware-library incompatibility (detailed in Section 4.1.7), ultimately selecting PyTorch 2.0.1 with CUDA 11.7 on Python 3.10. This modernization enables efficient GPU utilization while necessitating code porting.

**Dataset Constraints:**

Both Design A and Design B use an identical fixed evaluation subset of 35 samples drawn from the official ShapeNet test split. The subset spans six object categories (Airplane, Car, Chair, Table, Loudspeaker, Lamp) with a total of 8, 6, 8, 6, 4, and 3 samples respectively. Each sample is represented by three 224×224 RGB images (views at camera indices 0, 6, and 7) from ShapeNetRendering. This fixed subset ensures reproducibility and eliminates confounding factors (sample variance, data augmentation) from the performance comparison. Design C, planned but not yet executed, will incorporate the FaceScape dataset to investigate generalization to human face reconstruction—a distinct domain that challenges the template-based deformation approach.

**Reporting and Reproducibility Constraints:**

Due to the iterative nature of implementation, experiments were conducted across several weeks with multiple code refinements (bug fixes, layer clamping corrections, file naming standardization). To maintain scientific rigor, all results are reported from the final, validated implementations. Experimental logs, hardware specifications, and timing data are archived to enable reproduction.

---

## 4.1.3 Alternative Design Solutions and Rationale

The evaluation compares three design alternatives, each isolating a specific variable:

| **Aspect**    | **Design A**                   | **Design B**                  | **Design C** (Planned)       |
| ------------- | ------------------------------ | ----------------------------- | ---------------------------- |
| **Framework** | TensorFlow 1.15.0 (CPU)        | PyTorch 2.0.1 (CUDA 11.7)     | PyTorch 2.0.1 (CUDA 11.7)    |
| **Hardware**  | Intel i5-1335U CPU             | NVIDIA RTX 4070 GPU           | NVIDIA RTX 4070 GPU          |
| **Dataset**   | ShapeNet (35 samples, 6 cat.)  | ShapeNet (35 samples, 6 cat.) | FaceScape (pending)          |
| **Pipeline**  | Official legacy code (2-stage) | Modernized PyTorch (2-stage)  | Modernized PyTorch + retrain |
| **Status**    | ✅ Complete & Verified         | ✅ Complete & Verified        | 🔄 Planned (not executed)    |

**Design A (Baseline):**

Design A establishes a functional baseline using the official Pixel2Mesh++ implementation with pretrained weights on ShapeNet. This design serves as the reference point for all comparisons. It answers the question: "Does the legacy system work correctly on modern hardware?" Despite being CPU-bound and slow, Design A is valuable because it (1) validates the experimental setup using well-established code, (2) establishes the ground truth for output quality, and (3) provides the performance baseline for speedup calculations.

**Design B (GPU Modernization):**

Design B isolates the effect of GPU acceleration and framework modernization while holding the dataset constant. By comparing Design B to Design A on identical inputs, we can quantify the computational improvement achieved through GPU parallelization, CUDA kernel optimization, and PyTorch's efficient memory management. Design B investigates whether moving from legacy CPU-based TensorFlow to modern GPU-based PyTorch yields substantial practical speedup on real inference workloads.

**Design C (Dataset Generalization):**

Design C extends the evaluation to a different domain—human face reconstruction using the FaceScape dataset—while maintaining the same GPU/PyTorch implementation as Design B. This design isolates the effect of dataset shift on model performance, inference speed, and reconstruction quality. Because FaceScape represents a domain distinct from ShapeNet objects (with different geometry, topology constraints, and preprocessing), Design C will reveal whether GPU-accelerated Pixel2Mesh++ generalizes beyond the original training domain. However, Design C is not yet executed due to time constraints and pending FaceScape dataset setup.

**Rationale for Multi-Design Approach:**

Each design answers a distinct research question:

- **Design A + B comparison**: How much performance improvement can be achieved through hardware acceleration and modernization?
- **Design B + C comparison** (future): How robust is the accelerated pipeline to distribution shift and domain transfer?
- **A + B + C together**: What is the interplay between dataset choice, acceleration strategy, and practical inference latency?

This factorial design ensures that results are not confounded by simultaneous changes in multiple variables.

---

## 4.1.4 System Workflow and Experimental Methodology

The inference pipeline follows a standard two-stage mesh reconstruction workflow:

```
Preprocessing (load .dat files, extract images)
           ↓
    Stage 1: Coarse Mesh Generation (MVP2M)
           ↓
    Intermediate: Save coarse mesh as .xyz file
           ↓
    Stage 2: Mesh Refinement (P2MPP)
           ↓
    Output: Refined mesh as .obj file
           ↓
    Evaluation: Validate mesh and record timing
```

**Dataset Selection Methodology:**

The evaluation subset comprises 35 samples from six ShapeNet categories: Airplane (8), Car (6), Chair (8), Table (6), Loudspeaker (4), Lamp (3). This selection was made to (1) ensure diversity across object types (symmetric objects, articulated structures, smooth surfaces), (2) balance computational cost against statistical robustness, and (3) remain tractable for manual quality inspection. The subset is held constant across all designs to eliminate between-design variance due to sample selection.

**Input Specification:**

Each sample is provided as a preprocessed .dat file containing camera calibration matrices and ground truth labels, plus three rendered views (indices 0, 6, 7) from ShapeNetRendering. All views are 224×224 RGB images, matching the input specification of the original Pixel2Mesh++ model. This fixed input format is identical across Design A and Design B.

**Measurement Methodology:**

Timing measurements are collected at two levels of granularity:

1. **End-to-End Pipeline Timing**: Measures total elapsed time from loading the .dat file to writing the final .obj file, including all overhead (data loading, image I/O, file writes). For Design A, this includes Stage 1 total time (6.56s) and Stage 2 total time (101.95s), yielding a combined end-to-end total of 108.51s for 35 samples, or 3.10 seconds per sample on average.

2. **Inference-Only Timing**: Measures the wall-clock time spent in the neural network forward passes, excluding I/O and file operations. For Design A, inference-only timing is 6.96s ± 0.12s for 35 samples, corresponding to ~199 milliseconds per sample. For Design B, inference-only timing is 0.28s ± 0.01s for 35 samples, corresponding to ~8 milliseconds per sample.

Both timing views are essential: end-to-end timing is relevant for production deployment, while inference-only timing isolates the neural network performance from engineering overhead and reveals the true computational speedup.

**Fairness Controls:**

To ensure valid comparison between Design A and Design B, the following factors are held constant:

- Same fixed dataset (35 samples, 6 categories)
- Same input resolution (224×224 RGB, 3 views per sample)
- Same model architecture (two-stage pipeline, 2466-vertex output)
- Same output format (.obj mesh files)
- Same evaluation protocol (success/failure classification, timing measurement)

Any timing differences between designs are thus attributable to (a) hardware capabilities (CPU vs GPU) and (b) implementation optimizations (TensorFlow vs PyTorch), not to differences in input data or model structure.

---

## 4.1.5 Functional Verification ("Simulation") and Validation Strategy

In the context of this thesis, "functional verification" refers to comprehensive testing of implementation correctness without ground truth labels (which are unavailable for test data). This strategy validates that the implementation executes correctly, produces geometrically valid outputs, and preserves reconstruction quality through code modernization.

**Verification Outcomes:**

Both Design A and Design B achieved 35/35 successful completion on the evaluation subset:

- **Design A**: 35 samples processed without error, 35 valid mesh outputs generated, 35/35 meshes have expected topology (2466 vertices, consistent face structure).
- **Design B**: 35 samples processed without error, 35 valid mesh outputs generated, 35/35 meshes numerically equivalent to Design A outputs (floating-point precision < 0.001).

**Quality Match**: Visual inspection and numerical validation confirmed that Design B outputs are qualitatively indistinguishable from Design A outputs. Mesh vertex counts, face topology, and surface geometry are identical between the two designs, confirming that GPU acceleration and framework migration introduced no quality degradation.

**Correctness Checks:**

The following checks were applied to all 105 output files (35 samples × 3 stages/outputs):

- **File Existence**: All expected .xyz and .obj files were created.
- **File Non-Emptiness**: All output files contain valid data (non-zero size, parseable format).
- **Vertex Count**: All refined meshes contain exactly 2466 vertices (matching the template topology).
- **Geometric Validity**: No NaN, Inf, or degenerate triangle values in vertex positions or connectivity.
- **Reproducibility**: Multiple runs of Design A and Design B produced identical outputs (deterministic execution on CPU; GPU results consistent to floating-point precision).

These checks confirm that both implementations faithfully execute the Pixel2Mesh++ algorithm as specified and produce valid 3D mesh reconstructions.

---

## 4.1.6 Performance Profiling and Bottleneck Identification

Understanding the performance bottlenecks is essential for justifying the GPU acceleration strategy. The two-stage pipeline exhibits highly asymmetric computational cost.

**Stage-Wise Timing (Design A):**

| **Stage**              | **Total Time** | **Per-Sample** | **% of Pipeline** |
| ---------------------- | -------------- | -------------- | ----------------- |
| Stage 1 (Coarse MVP2M) | 6.56s          | 0.187s         | 6.0%              |
| Stage 2 (Refine P2MPP) | 101.95s        | 2.913s         | 94.0%             |
| **Combined**           | **108.51s**    | **3.10s**      | **100%**          |

Stage 2 (mesh refinement via graph convolutions and perceptual feature projection) dominates the runtime, consuming ~94% of the total pipeline time. This bottleneck arises because Stage 2 performs expensive operations:

- Multiple rounds of graph convolution on the 2466-vertex mesh
- Bilinear sampling from multi-scale image feature maps (224×224, 112×112, 56×56)
- Vertex-wise projection using learned perceptual features

Stage 1 (coarse generation) is comparatively fast because it performs fewer graph convolution rounds and no perceptual feature projection.

**Two Views of Performance:**

The distinction between end-to-end timing and inference-only timing is important:

- **End-to-End (3.10s/sample)**: Reflects practical deployment latency, including image I/O, data preprocessing, and file writes. This metric is relevant for real-world applications.

- **Inference-Only (199ms/sample for Design A, 8ms/sample for Design B)**: Isolates the neural network computation from engineering overhead. This metric reveals the true computational speedup (24.85×) and is necessary for understanding how GPU acceleration affects the algorithmic core.

The ratio between these two views (end-to-end ≈ 16× longer than inference-only for Design A) indicates that I/O and preprocessing overhead is substantial. However, this overhead is not the focus of the modernization effort; instead, we focus on accelerating the inference-critical Stage 2, where GPU parallelization offers the greatest benefit.

---

## 4.1.7 Design B Modernization Strategy (GPU/CUDA)

Design B's modernization from CPU-based TensorFlow to GPU-based PyTorch involves a strategic shift in where computation occurs and how it is scheduled.

**Conceptual Modernization Steps:**

The following steps outline the modernization strategy:

1. **Move Computation to GPU**: Transfer tensor operations (matrix multiplications, convolutions, sampling) from CPU to GPU hardware, where massive parallelism (7168 CUDA cores on RTX 4070) accelerates these operations.

2. **Reduce CPU-GPU Synchronization**: Minimize host-device data transfer by keeping intermediate tensors on GPU throughout the pipeline and only transferring final outputs to host memory.

3. **Leverage Hardware-Optimized Libraries**: Use PyTorch's native CUDA kernels and PyTorch3D's specialized 3D operations rather than manually implementing or relying on generic frameworks.

4. **Enable Batch Processing**: Structure computation to process multiple samples in a single GPU kernel invocation where possible, amortizing kernel launch overhead.

**Framework Migration Rationale:**

An initial attempt to migrate from TensorFlow 1.15 to TensorFlow 2.x encountered a critical blocker: the **cuSolver initialization failure**. TensorFlow 2.4–2.10 Docker images were compiled with cuSolver libraries targeting older GPU architectures; the RTX 4070 (Ada Lovelace, released 2023) is not supported by these libraries, causing all GPU linear algebra operations to silently fall back to CPU execution. This nullified GPU acceleration and left the system 6% faster than Design A—a negligible improvement insufficient to justify the modernization.

**Decision to Migrate to PyTorch:**

Rather than battle TensorFlow's cuSolver incompatibility, we selected PyTorch 2.0.1, which offers native support for Ada Lovelace GPUs and does not require external cuSolver. PyTorch's architecture allows direct use of NVIDIA's optimized CUDA kernels without the abstraction overhead of TensorFlow's graph compilation. This decision trade-off replaces in-place TensorFlow code modifications with a complete framework port; however, the result—24.85× speedup—justifies the engineering effort.

---

## 4.1.8 Limitations and Next Steps (Design C Plan)

**Limitations of the Current Work (Design A and B):**

1. **Single Dataset**: Results are based only on ShapeNet, a synthetic dataset of rigid objects. Generalization to real-world captured geometry or other object categories is unknown.

2. **No Model Retraining**: Design B uses the same pretrained weights as Design A; no GPU-optimized training is performed. This limits our ability to assess whether GPU acceleration improves the training pipeline.

3. **Single GPU Type**: Performance is measured only on RTX 4070. Behavior on other GPU architectures (A100, V100, mobile GPUs) is unexplored.

4. **Fixed Subset**: The evaluation uses only 35 samples. Statistical confidence would improve with a larger test set.

**Design C Plan (Human Face Reconstruction via FaceScape):**

Design C is planned as a forward extension to evaluate GPU-accelerated Pixel2Mesh++ on a domain-shifted dataset. FaceScape is a large-scale 3D human face dataset with high-resolution scans and detailed topology, representing a distribution distinct from ShapeNet objects.

**Expected Differences from ShapeNet:**

- Higher geometric detail and finer surface features
- Different template-fitting challenges (facial topology is more constrained than object topology)
- Potential requirement for template retraining or domain adaptation
- Different input preprocessing (face landmark alignment, etc.)

**Design C Methodology (Outline):**

1. **Dataset Mapping**: Identify or create a FaceScape test subset with corresponding multi-view renders, following the preprocessing pipeline of Design B.

2. **Inference Protocol**: Execute the pretrained Pixel2Mesh++ model on FaceScape test samples (same Stage 1 + Stage 2 pipeline as Design B).

3. **Measurement**: Collect inference timing and qualitative quality assessment (mesh validity, visual plausibility, convergence issues).

4. **Risks and Mitigations**:
   - **Risk**: Pretrained weights may be poorly adapted to face geometry, leading to invalid meshes or slow convergence.
   - **Mitigation**: Implement mesh validity checks; if necessary, retrain Stage 1 and/or Stage 2 on FaceScape data.
   - **Risk**: FaceScape preprocessing may differ from ShapeNet, affecting input distribution.
   - **Mitigation**: Carefully align FaceScape input pipeline to ShapeNet preprocessing (image normalization, camera calibration, view selection).

**Expected Outcomes of Design C:**

If Design C execution is completed, it will answer:

- Does GPU acceleration remain effective on face-domain models?
- Does mesh reconstruction quality degrade on out-of-distribution data?
- Are template-based models (Pixel2Mesh++) suitable for face reconstruction, or do they require architectural modifications?

Answers to these questions will inform the scope and recommendations of the thesis.

---

## Summary

This chapter has outlined the systematic design process and methodology for benchmarking Pixel2Mesh++ across three design alternatives (A, B, C). Design A and Design B have been fully implemented and verified, demonstrating a 24.85× inference speedup through GPU acceleration while maintaining output quality on 35 ShapeNet samples. The two-stage pipeline architecture, fixed evaluation subset, and comprehensive verification strategy provide a rigorous foundation for comparing implementations and isolating the sources of performance improvement. Design C, planned as a domain-shift experiment using FaceScape, will extend these results to evaluate generalization and identify remaining research challenges.
