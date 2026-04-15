#!/bin/bash
# Evaluate the held-out test slice of pre-generated method outputs.
#
# Usage:
#   bash scripts/evaluate_test.sh [results_dir] [experiment_name] [method_name]

set -euo pipefail

cd /home/user
mkdir -p scoring

RESULTS_DIR="${1:-/home/user/checkpoints}"
EXPERIMENT="${2:-watermark_detection_mistral}"
METHOD_NAME="${3:-gaussmark_200_tokens}"

echo "Running test-slice evaluation for experiment: ${EXPERIMENT}"
echo "Results directory: ${RESULTS_DIR}"

python3 eval/test/evaluate_slice.py \
    --results-dir "${RESULTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --method-name "${METHOD_NAME}" \
    --output /home/user/scoring/scores_test.json

python3 validate.py --compare
