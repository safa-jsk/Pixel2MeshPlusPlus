#!/bin/bash
# Design A GPU - Evaluation (wraps run_eval_gpu.sh)
# Usage: bash eval_gpu.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec bash run_eval_gpu.sh
