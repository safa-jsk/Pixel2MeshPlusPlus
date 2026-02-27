#!/bin/bash
# Design A CPU - Demo (single image 3D reconstruction)
# Usage: bash demo.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TF_SCRIPTS="$PROJECT_ROOT/src/p2mpp/tf/scripts"
CONFIG="$PROJECT_ROOT/configs/designA/p2mpp.yaml"

echo "=== Design A CPU: Demo ==="
echo ""

cd "$PROJECT_ROOT"
python "$TF_SCRIPTS/demo.py" --options "$CONFIG" "$@"
