#!/bin/bash
# Design B - Full benchmark (speed + metrics)
# Usage: bash benchmark.sh [--samples N]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACTS="$PROJECT_ROOT/artifacts"

echo "=== Design B: PyTorch GPU Benchmark (CAMFM) ==="
echo "Project Root: $PROJECT_ROOT"
echo ""

# Check GPU
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "--- Speed-only run ---"
cd "$SCRIPT_DIR"
python infer_speed.py "$@"

echo ""
echo "--- With metrics ---"
python infer_with_metrics.py "$@"

echo ""
echo "=== Benchmark complete ==="
echo "Outputs: $ARTIFACTS/outputs/"
