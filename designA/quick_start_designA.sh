#!/bin/bash
# Quick Start: Design A Evaluation
# Run this inside Docker container

echo "╔══════════════════════════════════════════════════════════╗"
echo "║      Design A Baseline Evaluation - Quick Start         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  • Run 2-stage inference on 35 samples (sequential)"
echo "  • Stage 1: Coarse MVP2M"
echo "  • Stage 2: Refined P2MPP"
echo "  • Generate .obj meshes for visualization"
echo "  • Collect timing statistics"
echo "  • Compute quality metrics:"
echo "      - Chamfer Distance (CD)"
echo "      - F1@tau (tau=0.0001)"
echo "      - F1@2tau (tau=0.0002)"
echo "  • Save results to outputs/designA/"
echo ""
echo "Estimated time: 6-10 minutes on CPU"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."
echo ""

# Run evaluation (sequential stages)
bash run_designA_eval.sh

# Show results
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Quick Results                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [ -f "../outputs/designA/benchmark/combined_timings.txt" ]; then
    echo "📊 Performance Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cat ../outputs/designA/benchmark/combined_timings.txt
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "⚠️  Timing stats not found"
fi

echo ""
echo "📁 Output Files:"
echo "  Meshes:   ../outputs/designA/eval_meshes/"
echo "  Timing:   ../outputs/designA/benchmark/timing_results_detailed.csv"
echo "  Metrics:  ../outputs/designA/benchmark/metrics_results.csv"
echo "  Summary:  ../outputs/designA/benchmark/summary_stats.txt"
echo ""
echo "🔍 To view a mesh:"
echo "  1. Copy .obj file to host machine"
echo "  2. Open in MeshLab or Blender"
echo ""
echo "Example:"
echo "  ls ../outputs/designA/eval_meshes/*.obj | head -1"
echo ""
