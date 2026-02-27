#!/bin/bash
# Design A - Sequential 2-Stage Evaluation Runner
# Runs Stage 1 (coarse) then Stage 2 (refined) separately to avoid graph conflicts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DESIGN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS="$PROJECT_ROOT/artifacts"

echo "=========================================="
echo "Design A - Sequential 2-Stage Evaluation"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "$DESIGN_DIR/designA_eval_list.txt" ]; then
    echo "❌ Error: designA_eval_list.txt not found in $DESIGN_DIR"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/data/p2mppdata/test" ]; then
    echo "❌ Error: $PROJECT_ROOT/data/p2mppdata/test/ not found"
    exit 1
fi

if [ ! -d "$ARTIFACTS/checkpoints/tf/coarse_mvp2m/models" ]; then
    echo "❌ Error: Stage 1 model checkpoint not found"
    exit 1
fi

if [ ! -d "$ARTIFACTS/checkpoints/tf/refine_p2mpp/models" ]; then
    echo "❌ Error: Stage 2 model checkpoint not found"
    exit 1
fi

echo "✓ All prerequisites found"
echo ""

cd "$SCRIPT_DIR"

# Stage 1: Coarse MVP2M
echo "==========================================  "
echo "Stage 1: Coarse MVP2M Inference"
echo "=========================================="
python eval_designA_stage1.py \
    --eval-list "$DESIGN_DIR/designA_eval_list.txt" \
    --output-dir "$ARTIFACTS/outputs/designA/eval_meshes"

if [ $? -ne 0 ]; then
    echo "❌ Stage 1 failed!"
    exit 1
fi

echo ""
echo "✓ Stage 1 complete"
echo ""

# Stage 2: Refined P2MPP
echo "=========================================="
echo "Stage 2: Refined P2MPP Inference"
echo "=========================================="
python eval_designA_stage2.py \
    --eval-list "$DESIGN_DIR/designA_eval_list.txt" \
    --coarse-dir "$ARTIFACTS/outputs/designA/eval_meshes" \
    --output-dir "$ARTIFACTS/outputs/designA/eval_meshes"

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 failed!"
    exit 1
fi

echo ""
echo "✓ Stage 2 complete"
echo ""

# Stage 3: Compute Quality Metrics
echo "=========================================="
echo "Stage 3: Computing Quality Metrics"
echo "  (Chamfer Distance, F1@tau, F1@2tau)"
echo "=========================================="
python compute_metrics.py \
    --mesh-dir "$ARTIFACTS/outputs/designA/eval_meshes" \
    --tau 0.0001

if [ $? -ne 0 ]; then
    echo "⚠️  Metrics computation failed (non-critical)"
fi

echo ""

# Display summary
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "📊 Results:"

if [ -f "$ARTIFACTS/outputs/designA/benchmark/combined_timings.txt" ]; then
    cat "$ARTIFACTS/outputs/designA/benchmark/combined_timings.txt"
    echo ""
else
    echo "⚠️  Combined timing stats not found"
    echo ""
fi

if [ -f "$ARTIFACTS/outputs/designA/benchmark/metrics_summary.txt" ]; then
    echo "Quality Metrics:"
    cat "$ARTIFACTS/outputs/designA/benchmark/metrics_summary.txt"
    echo ""
else
    echo "⚠️  Metrics summary not found"
    echo ""
fi

echo "📁 Output files:"
echo "  Meshes:  $ARTIFACTS/outputs/designA/eval_meshes/"
echo "  Timing:  $ARTIFACTS/outputs/designA/benchmark/"
echo "  Metrics: $ARTIFACTS/outputs/designA/benchmark/metrics_results.csv"
echo ""

# Count generated files
num_ground=$(ls -1 "$ARTIFACTS/outputs/designA/eval_meshes/"*_ground.xyz 2>/dev/null | wc -l)
num_predict=$(ls -1 "$ARTIFACTS/outputs/designA/eval_meshes/"*_predict.xyz 2>/dev/null | wc -l)
num_obj=$(ls -1 "$ARTIFACTS/outputs/designA/eval_meshes/"*_predict.obj 2>/dev/null | wc -l)

echo "  Generated: ${num_ground} ground truth, ${num_predict} predictions (.xyz), ${num_obj} meshes (.obj)"
echo ""
