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
# a JSON dict matching the reference.json experiment structure:
#   {"watermark_detection_mistral": {"results": {"gaussmark_200_tokens": {...}, "unwatermarked": {...}}}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Usage:
#   bash scripts/evaluate.sh <experiment_name> <results_dir>
#
# Default: evaluates watermark_detection_mistral experiment.

set -e

cd /home/user

mkdir -p scoring

RESULTS_DIR="${1:-/home/user/checkpoints}"
EXPERIMENT="${2:-watermark_detection_mistral}"

echo "Running evaluation for experiment: $EXPERIMENT"
echo "Results directory: $RESULTS_DIR"

# Run the evaluation
python3 -c "
import sys
sys.path.insert(0, '/home/user/eval')
from evaluate import evaluate_p_values
import json
import os
import glob

scores = {}
scores['$EXPERIMENT'] = {'results': {}}

# Look for detection results files
# Watermarked detection (detection_full.json or detection_metrics.json)
wm_candidates = glob.glob(os.path.join('$RESULTS_DIR', 'watermarked_*_detection_full.json'))
wm_candidates += glob.glob(os.path.join('$RESULTS_DIR', 'watermarked_*_detection_metrics.json'))
uwm_candidates = glob.glob(os.path.join('$RESULTS_DIR', 'unwatermarked_*_detection_full.json'))
uwm_candidates += glob.glob(os.path.join('$RESULTS_DIR', 'unwatermarked_*_detection_metrics.json'))
uwm_candidates += glob.glob(os.path.join('$RESULTS_DIR', '*unwatermarked*_detection_full.json'))

wm_candidates = sorted(set(wm_candidates))
uwm_candidates = sorted(set(uwm_candidates))

print(f'Watermarked detection files: {wm_candidates}')
print(f'Unwatermarked detection files: {uwm_candidates}')

# Process watermarked results
if wm_candidates:
    wm_file = wm_candidates[0]
    with open(wm_file, 'r') as f:
        wm_data = json.load(f)

    p_wm = wm_data.get('p_values', [])
    if not p_wm:
        # Try from metrics
        for k, v in wm_data.items():
            if isinstance(v, dict) and 'p_values' in v:
                p_wm = v['p_values']
                break

    if p_wm:
        metrics = evaluate_p_values(p_wm)
        # Use method name based on generation parameters
        # The watermarked file contains detection at 200 tokens
        scores['$EXPERIMENT']['results']['gaussmark_200_tokens'] = {
            'detection_rate_fpr_0.05': metrics.get('detection_rate_fpr_0.05', 0.0),
            'detection_rate_fpr_0.01': metrics.get('detection_rate_fpr_0.01', 0.0),
            'median_p_value': metrics.get('median_p_value', 0.5),
        }
        # AUC requires unwatermarked comparison
        print(f'Watermarked: detection_rate_fpr_0.05={metrics.get(\"detection_rate_fpr_0.05\", 0):.4f}, '
              f'median_p_value={metrics.get(\"median_p_value\", 0):.4f}')

# Process unwatermarked results (for FPR / baseline)
if uwm_candidates:
    uwm_file = uwm_candidates[0]
    with open(uwm_file, 'r') as f:
        uwm_data = json.load(f)

    p_uwm = uwm_data.get('p_values', [])
    if not p_uwm:
        for k, v in uwm_data.items():
            if isinstance(v, dict) and 'p_values' in v:
                p_uwm = v['p_values']
                break

    if p_uwm:
        metrics_uwm = evaluate_p_values(p_uwm)
        scores['$EXPERIMENT']['results']['unwatermarked'] = {
            'detection_rate_fpr_0.05': metrics_uwm.get('detection_rate_fpr_0.05', 0.0),
            'detection_rate_fpr_0.01': metrics_uwm.get('detection_rate_fpr_0.01', 0.0),
            'median_p_value': metrics_uwm.get('median_p_value', 0.5),
        }
        print(f'Unwatermarked: detection_rate_fpr_0.05={metrics_uwm.get(\"detection_rate_fpr_0.05\", 0):.4f}, '
              f'median_p_value={metrics_uwm.get(\"median_p_value\", 0):.4f}')

        # Compute AUC if we have both
        if p_wm and p_uwm:
            from eval.evaluate import compute_roc_auc
            try:
                auc = compute_roc_auc(p_wm, p_uwm)
                scores['$EXPERIMENT']['results']['gaussmark_200_tokens']['auc'] = round(auc, 4)
                print(f'AUC: {auc:.4f}')
            except:
                pass

print('\\nEvaluation results:')
print(json.dumps(scores, indent=2))

with open('/home/user/scoring/scores.json', 'w') as f:
    json.dump(scores, f, indent=2)
print('Scores written to /home/user/scoring/scores.json')
"
