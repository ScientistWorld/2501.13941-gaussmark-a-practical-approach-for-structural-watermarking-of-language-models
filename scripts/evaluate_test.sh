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

# Enumerate all watermarked detection result files (excluding baselines, robustness, unwatermarked)
mapfile -t WM_FILES < <(
    find "${RESULTS_DIR}" -maxdepth 1 -name "*_detection_full.json" \
        -o -name "*watermark*_detection_full.json" 2>/dev/null | \
    grep -v -i "unwatermarked\|baseline\|robustness" | sort
)

if [ ${#WM_FILES[@]} -eq 0 ]; then
    echo "No watermarked detection result files found in ${RESULTS_DIR}"
    exit 1
fi

echo "Found ${#WM_FILES[@]} detection file(s) to process"

# Accumulate all method scores into a temporary file, then merge
ACCUMULATED=""
for wm_file in "${WM_FILES[@]}"; do
    # Derive method name from filename: strip path, _detection_full.json suffix
    fname=$(basename "${wm_file}")
    METHOD_NAME="${fname%_detection_full.json}"
    METHOD_NAME="${METHOD_NAME%_watermarked_detection_full}"

    echo "  Processing method: ${METHOD_NAME} (from ${fname})"

    tmp_out=$(mktemp)
    python3 eval/test/evaluate_slice.py \
        --results-dir "${RESULTS_DIR}" \
        --experiment "${EXPERIMENT}" \
        --method-name "${METHOD_NAME}" \
        --watermarked-file "${wm_file}" \
        --output "${tmp_out}"

    if [ -z "${ACCUMULATED}" ]; then
        ACCUMULATED="${tmp_out}"
    else
        # Merge into accumulated file using Python
        python3 - <<EOF
import json, sys
a = json.load(open("${ACCUMULATED}"))
b = json.load(open("${tmp_out}"))
# Merge b into a at experiment level
for exp, exp_data in b.get("experiments", {}).items():
    if exp not in a.get("experiments", {}):
        a["experiments"][exp] = {"results": {}}
    if "results" in exp_data:
        a["experiments"][exp]["results"].update(exp_data["results"])
print(json.dumps(a, indent=2))
EOF
        > "${ACCUMULATED}"
        rm "${tmp_out}"
    fi
done

# Copy accumulated results to final output
if [ -n "${ACCUMULATED}" ]; then
    cp "${ACCUMULATED}" /home/user/scoring/scores_test.json
    rm "${ACCUMULATED}"
fi

python3 validate.py --compare
