# Design A Quality Metrics Summary

## Evaluation Configuration

| Parameter       | Value                 |
| --------------- | --------------------- |
| **Date**        | January 31, 2026      |
| **Samples**     | 1,023                 |
| **Threshold τ** | 0.0001                |
| **Framework**   | TensorFlow 1.15 (CPU) |

---

## Overall Metrics

| Metric               | Mean       | Std Dev     | Description                     |
| -------------------- | ---------- | ----------- | ------------------------------- |
| **Chamfer Distance** | 0.00040363 | ±0.00049255 | Bidirectional surface distance  |
| **F1@τ**             | 66.64%     | -           | F1-score at threshold τ=0.0001  |
| **F1@2τ**            | 80.45%     | -           | F1-score at threshold 2τ=0.0002 |

---

## Per-Category Breakdown

### Chamfer Distance (Lower is Better)

| Rank | Category  | ID       | CD Value | Relative |
| ---- | --------- | -------- | -------- | -------- |
| 1    | **Plane** | 02691156 | 0.000225 | Best     |
| 2    | Car       | 02958343 | 0.000256 | 1.14×    |
| 3    | Chair     | 03001627 | 0.000402 | 1.79×    |
| 3    | Table     | 04379243 | 0.000402 | 1.79×    |
| 5    | Lamp      | 03636649 | 0.000567 | 2.52×    |
| 6    | Speaker   | 03691459 | 0.000581 | 2.58×    |

### F1@τ Score (Higher is Better)

| Rank | Category  | ID       | F1@τ   | F1@2τ  |
| ---- | --------- | -------- | ------ | ------ |
| 1    | **Plane** | 02691156 | 82.36% | 90.24% |
| 2    | Table     | 04379243 | 69.85% | 83.03% |
| 3    | Car       | 02958343 | 66.22% | 82.42% |
| 4    | Lamp      | 03636649 | 65.13% | 77.40% |
| 5    | Chair     | 03001627 | 61.26% | 77.25% |
| 6    | Speaker   | 03691459 | 54.58% | 72.01% |

---

## Sample Distribution

```
Category Distribution (n=1023):
├── 02691156 (Plane):   ████████████████████ 173 (16.9%)
├── 02958343 (Car):     ████████████████████ 172 (16.8%)
├── 03001627 (Chair):   ████████████████████ 172 (16.8%)
├── 03636649 (Lamp):    ███████████████████  166 (16.2%)
├── 03691459 (Speaker): ███████████████████  168 (16.4%)
└── 04379243 (Table):   ████████████████████ 172 (16.8%)
```

---

## Metric Definitions

### Chamfer Distance (CD)

The bidirectional Chamfer Distance measures the average closest point distance between predicted mesh $\hat{S}$ and ground truth mesh $S$:

$$CD(\hat{S}, S) = \frac{1}{|\hat{S}|}\sum_{x \in \hat{S}} \min_{y \in S} \|x - y\|_2^2 + \frac{1}{|S|}\sum_{y \in S} \min_{x \in \hat{S}} \|y - x\|_2^2$$

### F1-Score at Threshold τ

F1-score computed using precision and recall at distance threshold τ:

- **Precision@τ**: Fraction of predicted points within τ of ground truth
- **Recall@τ**: Fraction of ground truth points within τ of prediction

$$F1@\tau = 2 \cdot \frac{Precision@\tau \cdot Recall@\tau}{Precision@\tau + Recall@\tau}$$

---

## Statistical Analysis

### Chamfer Distance Distribution

| Statistic | Value      |
| --------- | ---------- |
| Mean      | 0.00040363 |
| Std Dev   | 0.00049255 |
| Median    | ~0.00025   |
| Min       | 0.00004450 |
| Max       | ~0.0015    |

### Key Observations

1. **Plane category** consistently achieves the best reconstruction quality
2. **Speaker category** shows the most challenging reconstruction
3. **Table and Chair** have similar CD but different F1 scores
4. **F1@2τ improvement** averages ~14 percentage points over F1@τ

---

## Comparison Ready

These baseline metrics are ready for comparison with:

- **Design B**: PyTorch GPU-accelerated implementation
- **Design C**: Optimized hybrid approach

Expected improvements with GPU acceleration:

- Timing: 5-10× speedup
- Quality: Maintained or improved

---

## Files Generated

| File                         | Description                         |
| ---------------------------- | ----------------------------------- |
| `metrics_results.csv`        | Full per-sample metrics (1025 rows) |
| `metrics_summary.txt`        | Text format summary                 |
| `DesignA_Metrics_Summary.md` | This document                       |

---

_Last Updated: February 3, 2026_
