#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# Self-contained: calls code in eval/ and writes scoring/scores.json.
#
# Usage:
#   bash scripts/evaluate.sh [results_dir] [experiment_name]

set -euo pipefail

cd /home/user
mkdir -p scoring

RESULTS_DIR="${1:-/home/user/checkpoints}"
EXPERIMENT="${2:-watermark_detection_mistral}"

echo "Running evaluation for experiment: ${EXPERIMENT}"
echo "Results directory: ${RESULTS_DIR}"

python3 eval/directory_evaluate.py \
    --results-dir "${RESULTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --output /home/user/scoring/scores.json

python3 validate.py --compare
