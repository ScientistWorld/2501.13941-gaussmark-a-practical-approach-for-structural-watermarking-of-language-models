#!/bin/bash
# Evaluate the held-out test slice of pre-generated method outputs.
#
# Usage:
#   bash scripts/evaluate_test.sh [results_dir] [experiment_name]
#
# Iterates over all methods found in the results directory.
# For each method, runs eval/test/evaluate_slice.py and accumulates results.

set -euo pipefail

cd /home/user
mkdir -p scoring

RESULTS_DIR="${1:-/home/user/checkpoints}"
EXPERIMENT="${2:-watermark_detection_mistral}"

echo "Running test-slice evaluation for experiment: ${EXPERIMENT}"
echo "Results directory: ${RESULTS_DIR}"

# Enumerate proposed benchmark methods from scoring/scores.json so the split
# matches the scored benchmark contract instead of every historical artifact.
mapfile -t METHOD_NAMES < <(
    EXPERIMENT_NAME="${EXPERIMENT}" python3 - <<'PY'
import json
import os

with open("/home/user/scoring/scores.json") as f:
    data = json.load(f)

experiment = data.get("experiments", {}).get(os.environ["EXPERIMENT_NAME"], {})
for method_name, metrics in experiment.get("results", {}).items():
    if isinstance(metrics, dict) and method_name != "unwatermarked":
        print(method_name)
PY
)

if [ ${#METHOD_NAMES[@]} -eq 0 ]; then
    echo "No proposed methods found for ${EXPERIMENT} in scoring/scores.json"
    exit 1
fi

echo "Found ${#METHOD_NAMES[@]} benchmark method(s) to process"

# Accumulate all method scores into a temporary file, then merge
ACCUMULATED=""
for METHOD_NAME in "${METHOD_NAMES[@]}"; do
    echo "  Processing benchmark method: ${METHOD_NAME}"

    tmp_out=$(mktemp)
    python3 eval/test/evaluate_slice.py \
        --results-dir "${RESULTS_DIR}" \
        --experiment "${EXPERIMENT}" \
        --method-name "${METHOD_NAME}" \
        --output "${tmp_out}"

    if [ -z "${ACCUMULATED}" ]; then
        ACCUMULATED="${tmp_out}"
    else
        merged_out=$(mktemp)
        python3 - "${ACCUMULATED}" "${tmp_out}" <<'PY' > "${merged_out}"
import json
import sys

with open(sys.argv[1]) as f:
    accumulated = json.load(f)
with open(sys.argv[2]) as f:
    update = json.load(f)

for exp, exp_data in update.get("experiments", {}).items():
    accumulated.setdefault("experiments", {}).setdefault(exp, {"results": {}})
    accumulated["experiments"][exp].setdefault("results", {}).update(
        exp_data.get("results", {})
    )

for exp, audit_data in update.get("_audit", {}).items():
    accumulated.setdefault("_audit", {}).setdefault(exp, {}).update(audit_data)

print(json.dumps(accumulated, indent=2))
PY
        mv "${merged_out}" "${ACCUMULATED}"
        rm "${tmp_out}"
    fi
done

# Copy accumulated results to final output
if [ -n "${ACCUMULATED}" ]; then
    cp "${ACCUMULATED}" /home/user/scoring/scores_test.json
    rm "${ACCUMULATED}"
fi
