#!/bin/bash
# Design A CPU - Train Stage 2 (Refined P2MPP)
# Usage: bash train_stage2.sh [--epochs N]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TF_SCRIPTS="$PROJECT_ROOT/src/p2mpp/tf/scripts"
CONFIG="$PROJECT_ROOT/configs/designA/p2mpp.yaml"

echo "=== Design A CPU: Train Stage 2 (Refined P2MPP) ==="
echo "Config: $CONFIG"
echo ""

cd "$PROJECT_ROOT"
python "$TF_SCRIPTS/train_p2mpp.py" --options "$CONFIG" "$@"
