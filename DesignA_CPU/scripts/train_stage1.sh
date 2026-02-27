#!/bin/bash
# Design A CPU - Train Stage 1 (Coarse MVP2M)
# Usage: bash train_stage1.sh [--epochs N]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TF_SCRIPTS="$PROJECT_ROOT/src/p2mpp/tf/scripts"
CONFIG="$PROJECT_ROOT/configs/designA/mvp2m.yaml"

echo "=== Design A CPU: Train Stage 1 (Coarse MVP2M) ==="
echo "Config: $CONFIG"
echo ""

cd "$PROJECT_ROOT"
python "$TF_SCRIPTS/train_mvp2m.py" --options "$CONFIG" "$@"
