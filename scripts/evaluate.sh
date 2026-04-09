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
# a JSON dict matching the reference.json experiment structure.
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
import json
import os
import glob

scores = {}
scores['$EXPERIMENT'] = {'results': {}}

# Find watermarked detection results (ends with _detection_full.json or _detection_metrics.json)
# from the watermarked generation output file.
wm_detect_files = sorted(glob.glob(os.path.join('$RESULTS_DIR', 'watermarked_*_detection_full.json')))
wm_metrics_files = sorted(glob.glob(os.path.join('$RESULTS_DIR', 'watermarked_*_detection_metrics.json')))

# Find unwatermarked detection results
uwm_detect_files = sorted(glob.glob(os.path.join('$RESULTS_DIR', 'unwatermarked_*_detection_full.json')))
uwm_metrics_files = sorted(glob.glob(os.path.join('$RESULTS_DIR', 'unwatermarked_*_detection_metrics.json')))

print(f'Watermarked detection files: {wm_detect_files}')
print(f'Unwatermarked detection files: {uwm_detect_files}')

# Load p-values from detection results
def load_p_values(file_list):
    '''Load p_values from the first available detection file.'''
    p_values = None
    for fpath in file_list:
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            data = json.load(f)
        # p_values may be at top level or nested
        if 'p_values' in data:
            return data['p_values']
        for k, v in data.items():
            if isinstance(v, dict) and 'p_values' in v:
                return v['p_values']
    return None

# Load watermarked p-values
p_wm = load_p_values(wm_detect_files + wm_metrics_files)
p_uwm = load_p_values(uwm_detect_files + uwm_metrics_files)

print(f'Watermarked p-values loaded: {len(p_wm) if p_wm else 0}')
print(f'Unwatermarked p-values loaded: {len(p_uwm) if p_uwm else 0}')

# Compute metrics from p-values
def compute_metrics(p_values):
    '''Compute detection metrics from a list of p-values.'''
    import numpy as np
    if not p_values:
        return {}
    p = np.array(p_values)
    return {
        'detection_rate_fpr_0.05': float(np.mean(p < 0.05)),
        'detection_rate_fpr_0.01': float(np.mean(p < 0.01)),
        'median_p_value': float(np.median(p)),
        'mean_p_value': float(np.mean(p)),
        'std_p_value': float(np.std(p)),
    }

if p_wm:
    metrics_wm = compute_metrics(p_wm)
    scores['$EXPERIMENT']['results']['gaussmark_200_tokens'] = metrics_wm
    print(f'Watermarked: detection_rate_fpr_0.05={metrics_wm.get(\"detection_rate_fpr_0.05\", 0):.4f}, '
          f'median_p_value={metrics_wm.get(\"median_p_value\", 0):.4f}')

if p_uwm:
    metrics_uwm = compute_metrics(p_uwm)
    scores['$EXPERIMENT']['results']['unwatermarked'] = metrics_uwm
    print(f'Unwatermarked: detection_rate_fpr_0.05={metrics_uwm.get(\"detection_rate_fpr_0.05\", 0):.4f}, '
          f'median_p_value={metrics_uwm.get(\"median_p_value\", 0):.4f}')

    # Compute AUC if we have both watermarked and unwatermarked p-values
    if p_wm:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        labels = np.concatenate([np.ones(len(p_wm)), np.zeros(len(p_uwm))])
        scores_arr = np.concatenate([p_wm, p_uwm])
        # Lower p-value = more likely to be watermarked
        try:
            auc = roc_auc_score(labels, -scores_arr)
            scores['$EXPERIMENT']['results']['gaussmark_200_tokens']['auc'] = round(float(auc), 4)
            print(f'AUC: {auc:.4f}')
        except Exception as e:
            print(f'AUC computation failed: {e}')

print()
print('Evaluation results:')
print(json.dumps(scores, indent=2))

with open('/home/user/scoring/scores.json', 'w') as f:
    json.dump(scores, f, indent=2)
print('Scores written to /home/user/scoring/scores.json')
"
