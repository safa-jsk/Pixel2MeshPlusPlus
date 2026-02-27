#!/bin/bash
# Design A CPU - Evaluation (wraps run_designA_eval.sh)
# Usage: bash eval.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec bash run_designA_eval.sh
