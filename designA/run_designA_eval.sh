#!/bin/bash
# Design A - Sequential 2-Stage Evaluation Runner
# Runs Stage 1 (coarse) then Stage 2 (refined) separately to avoid graph conflicts

echo "=========================================="
echo "Design A - Sequential 2-Stage Evaluation"
echo "=========================================="

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "designA_eval_list.txt" ]; then
    echo "❌ Error: designA_eval_list.txt not found"
    exit 1
fi

if [ ! -d "../data/p2mppdata/test" ]; then
    echo "❌ Error: ../data/p2mppdata/test/ not found"
    exit 1
fi

if [ ! -d "../results/coarse_mvp2m/models" ]; then
    echo "❌ Error: Stage 1 model checkpoint not found"
    exit 1
fi

if [ ! -d "../results/refine_p2mpp/models" ]; then
    echo "❌ Error: Stage 2 model checkpoint not found"
    exit 1
fi

echo "✓ All prerequisites found"
echo ""

# Stage 1: Coarse MVP2M
echo "==========================================  "
echo "Stage 1: Coarse MVP2M Inference"
echo "=========================================="
python eval_designA_stage1.py \
    --eval-list designA_eval_list.txt \
    --output-dir ../outputs/designA/eval_meshes

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
    --eval-list designA_eval_list.txt \
    --coarse-dir ../outputs/designA/eval_meshes \
    --output-dir ../outputs/designA/eval_meshes

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 failed!"
    exit 1
fi

echo ""
echo "✓ Stage 2 complete"
echo ""

# Display summary
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "📊 Results:"

if [ -f "../outputs/designA/benchmark/combined_timings.txt" ]; then
    cat ../outputs/designA/benchmark/combined_timings.txt
    echo ""
else
    echo "⚠️  Combined timing stats not found"
    echo ""
fi

echo "📁 Output files:"
echo "  Meshes: ../outputs/designA/eval_meshes/"
echo "  Timing: ../outputs/designA/benchmark/"
echo ""

# Count generated files
num_ground=$(ls -1 ../outputs/designA/eval_meshes/*_ground.xyz 2>/dev/null | wc -l)
num_predict=$(ls -1 ../outputs/designA/eval_meshes/*_predict.xyz 2>/dev/null | wc -l)
num_obj=$(ls -1 ../outputs/designA/eval_meshes/*_predict.obj 2>/dev/null | wc -l)

echo "  Generated: ${num_ground} ground truth, ${num_predict} predictions (.xyz), ${num_obj} meshes (.obj)"
echo ""
