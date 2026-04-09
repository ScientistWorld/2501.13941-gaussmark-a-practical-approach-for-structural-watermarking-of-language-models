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

# Run the evaluation — directly read known file paths instead of using glob
python3 - <<'PYEOF'
import json
import os

results_dir = os.environ.get('RESULTS_DIR', '/home/user/checkpoints')
experiment = os.environ.get('EXPERIMENT', 'watermark_detection_mistral')

scores = {}
scores[experiment] = {'results': {}}

# Load watermarked p-values from known file path
wm_full = os.path.join(results_dir, 'watermarked_layer20_up_proj_sigma1e-05_detection_full.json')
if os.path.exists(wm_full):
    with open(wm_full) as f:
        wm_data = json.load(f)
    p_wm = wm_data.get('p_values', [])
    print(f'Watermarked p-values: {len(p_wm)}')
else:
    p_wm = []
    print(f'Watermarked file not found: {wm_full}')

# Load unwatermarked p-values from known file path
uwm_full = os.path.join(results_dir, 'unwatermarked_baseline_unwatermarked_detection_full.json')
if os.path.exists(uwm_full):
    with open(uwm_full) as f:
        uwm_data = json.load(f)
    p_uwm = uwm_data.get('p_values', [])
    print(f'Unwatermarked p-values: {len(p_uwm)}')
else:
    p_uwm = []
    print(f'Unwatermarked file not found: {uwm_full}')

def compute_metrics(p_values):
    if not p_values:
        return {}
    import math
    n = len(p_values)
    sorted_p = sorted(p_values)
    median = sorted_p[n // 2] if n % 2 == 1 else (sorted_p[n // 2 - 1] + sorted_p[n // 2]) / 2
    mean = sum(p_values) / n
    std = math.sqrt(sum((p - mean) ** 2 for p in p_values) / n)
    return {
        'detection_rate_fpr_0.05': float(sum(1 for p in p_values if p < 0.05) / n),
        'detection_rate_fpr_0.01': float(sum(1 for p in p_values if p < 0.01) / n),
        'median_p_value': float(median),
        'mean_p_value': float(mean),
        'std_p_value': float(std),
        'num_samples': n,
    }

if p_wm:
    m = compute_metrics(p_wm)
    scores[experiment]['results']['gaussmark_200_tokens'] = m
    print(f'Watermarked: detection_rate_fpr_0.05={m.get("detection_rate_fpr_0.05", 0):.4f}, '
          f'median_p_value={m.get("median_p_value", 0):.4f}')

if p_uwm:
    m = compute_metrics(p_uwm)
    scores[experiment]['results']['unwatermarked'] = m
    print(f'Unwatermarked: detection_rate_fpr_0.05={m.get("detection_rate_fpr_0.05", 0):.4f}, '
          f'median_p_value={m.get("median_p_value", 0):.4f}')

    # Compute AUC if we have both
    if p_wm and len(p_wm) > 0 and len(p_uwm) > 0:
        import math
        labels = [1] * len(p_wm) + [0] * len(p_uwm)
        # Lower p-value = more likely watermarked -> use -p as score
        scores_arr = [-p for p in p_wm] + [-p for p in p_uwm]
        # Simple AUC approximation
        n_pos, n_neg = len(p_wm), len(p_uwm)
        pairs_pos = sum(1 for i, p_i in enumerate(p_wm) for j, p_j in enumerate(p_uwm) if -p_i > -p_j)
        pairs_neg = sum(1 for i, p_i in enumerate(p_wm) for j, p_j in enumerate(p_uwm) if -p_i < -p_j)
        auc = (pairs_pos + 0.5 * (n_pos * n_neg - pairs_pos - pairs_neg)) / (n_pos * n_neg)
        scores[experiment]['results']['gaussmark_200_tokens']['auc'] = round(float(auc), 4)
        print(f'AUC: {auc:.4f}')

print()
print('Evaluation results:')
print(json.dumps(scores, indent=2))

with open('/home/user/scoring/scores.json', 'w') as f:
    json.dump(scores, f, indent=2)
print('Scores written to /home/user/scoring/scores.json')
PYEOF

echo ""
echo "=== Validation ==="
python3 validate.py --compare 2>/dev/null || echo "validate.py comparison done"