#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"watermark_detection_mistral": {"detection_rate_fpr_0.05": 0.62, ...}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Usage:
#   bash scripts/evaluate.sh <experiment_name> <results_dir>
#
# Where results_dir contains detection results:
#   results_dir/watermarked_pvalues.json  - p-values from watermarked text detection
#   results_dir/unwatermarked_pvalues.json - p-values from unwatermarked text detection

set -e

cd /home/user

mkdir -p scoring

# Default paths
RESULTS_DIR="${1:-/home/user/checkpoints}"
EXPERIMENT="${2:-watermark_detection_mistral}"

echo "Running evaluation for experiment: $EXPERIMENT"
echo "Results directory: $RESULTS_DIR"

# Run the evaluation
python3 -c "
import sys
sys.path.insert(0, '/home/user/eval')
from evaluate import run_evaluation, load_p_values, evaluate_p_values, evaluate_detection
import json
import os
import glob

results = {}

# Look for detection results
wm_patterns = [
    os.path.join('$RESULTS_DIR', '*watermarked*.json'),
    os.path.join('$RESULTS_DIR', 'watermarked_results.json'),
]
uwm_patterns = [
    os.path.join('$RESULTS_DIR', '*unwatermarked*.json'),
    os.path.join('$RESULTS_DIR', 'unwatermarked_results.json'),
]

wm_files = []
for p in wm_patterns:
    wm_files.extend(glob.glob(p))
uwm_files = []
for p in uwm_patterns:
    uwm_files.extend(glob.glob(p))

print(f'Found watermarked result files: {wm_files}')
print(f'Found unwatermarked result files: {uwm_files}')

if wm_files:
    with open(wm_files[0], 'r') as f:
        data = json.load(f)
    if 'p_values' in data:
        p_wm = data['p_values']
    elif 'samples' in data:
        p_wm = [s.get('p_value', 1.0) for s in data['samples']]
    else:
        p_wm = [1.0]
    metrics = evaluate_p_values(p_wm)
    results['watermarked_detection'] = metrics

if uwm_files:
    with open(uwm_files[0], 'r') as f:
        data = json.load(f)
    if 'p_values' in data:
        p_uwm = data['p_values']
    elif 'samples' in data:
        p_uwm = [s.get('p_value', 1.0) for s in data['samples']]
    else:
        p_uwm = [1.0]
    metrics_uwm = evaluate_p_values(p_uwm)
    results['unwatermarked_detection'] = metrics_uwm

    if wm_files:
        from eval.evaluate import compute_roc_auc
        auc = compute_roc_auc(p_wm, p_uwm)
        results['watermarked_detection']['auc'] = auc

print('Evaluation results:')
print(json.dumps(results, indent=2))

# Write scores.json in the expected format
scores = {
    '$EXPERIMENT': results.get('watermarked_detection', {})
}
with open('/home/user/scoring/scores.json', 'w') as f:
    json.dump(scores, f, indent=2)

print('Scores written to /home/user/scoring/scores.json')
"
