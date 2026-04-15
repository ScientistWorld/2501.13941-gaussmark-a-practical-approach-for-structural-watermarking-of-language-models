#!/bin/bash
# Evaluate the visible train slice of pre-generated method outputs.
#
# Usage:
#   bash scripts/evaluate_train.sh [results_dir] [experiment_name] [method_name]

set -euo pipefail

cd /home/user
mkdir -p scoring

RESULTS_DIR="${1:-/home/user/checkpoints}"
EXPERIMENT="${2:-watermark_detection_mistral}"
METHOD_NAME="${3:-gaussmark_200_tokens}"

echo "Running train-slice evaluation for experiment: ${EXPERIMENT}"
echo "Results directory: ${RESULTS_DIR}"

python3 eval/train/evaluate_slice.py \
    --results-dir "${RESULTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --method-name "${METHOD_NAME}" \
    --output /home/user/scoring/scores_train.json

python3 validate.py --compare
