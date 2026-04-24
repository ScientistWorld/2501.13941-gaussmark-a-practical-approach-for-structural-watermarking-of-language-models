#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# Self-contained: calls code in eval/ and writes scoring/scores.json.
#
# Usage:
#   bash scripts/evaluate.sh [results_dir] [experiment_name] [method_name] [watermarked_file] [unwatermarked_file]

set -euo pipefail

cd /home/user
mkdir -p scoring

RESULTS_DIR="${1:-/home/user/checkpoints}"
EXPERIMENT="${2:-watermark_detection_mistral}"
METHOD_NAME="${3:-}"
WATERMARKED_FILE="${4:-}"
UNWATERMARKED_FILE="${5:-}"

echo "Running evaluation for experiment: ${EXPERIMENT}"
echo "Results directory: ${RESULTS_DIR}"

CMD=(
    python3 eval/directory_evaluate.py
    --results-dir "${RESULTS_DIR}"
    --experiment "${EXPERIMENT}"
    --output /home/user/scoring/scores.json
)

if [ -n "${METHOD_NAME}" ]; then
    CMD+=(--method-name "${METHOD_NAME}")
fi
if [ -n "${WATERMARKED_FILE}" ]; then
    CMD+=(--watermarked-file "${WATERMARKED_FILE}")
fi
if [ -n "${UNWATERMARKED_FILE}" ]; then
    CMD+=(--unwatermarked-file "${UNWATERMARKED_FILE}")
fi

"${CMD[@]}"

python3 validate.py --compare
