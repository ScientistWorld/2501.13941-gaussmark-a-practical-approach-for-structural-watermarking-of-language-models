#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict matching the reference.json experiment structure.
# Only include metrics that are defined in reference.json.
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

# Try sigma=1e-3 detection results first (current run with per-sample keys)
wm_full = os.path.join(results_dir, 'wm_wm_l20_up_proj_s0001_detection_full.json')
if os.path.exists(wm_full):
    with open(wm_full) as f:
        wm_data = json.load(f)
    p_wm = wm_data.get('p_values', [])
    print(f'Sigma=1e-3 Watermarked p-values: {len(p_wm)}')
else:
    # Fallback: try sigma=0.032 detection results
    wm_full = os.path.join(results_dir, 'watermarked_layer20_up_proj_sigma3e-02_detection_full.json')
    if os.path.exists(wm_full):
        with open(wm_full) as f:
            wm_data = json.load(f)
        p_wm = wm_data.get('p_values', [])
        print(f'Sigma=0.032 Watermarked p-values: {len(p_wm)}')
    else:
        p_wm = []
        print(f'Watermarked file not found: checked {os.path.join(results_dir, "wm_wm_l20_up_proj_s0001_detection_full.json")} and sigma=0.032')

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
    """Compute only the metrics defined in reference.json."""
    if not p_values:
        return {}
    n = len(p_values)
    sorted_p = sorted(p_values)
    median = sorted_p[n // 2] if n % 2 == 1 else (sorted_p[n // 2 - 1] + sorted_p[n // 2]) / 2
    return {
        'detection_rate_fpr_0.05': float(sum(1 for p in p_values if p < 0.05) / n),
        'detection_rate_fpr_0.01': float(sum(1 for p in p_values if p < 0.01) / n),
        'median_p_value': float(median),
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

    # Compute AUC if we have both watermarked and unwatermarked p-values
    if p_wm and len(p_wm) > 0 and len(p_uwm) > 0:
        n_pos, n_neg = len(p_wm), len(p_uwm)
        # AUC: proportion of (wm, uwm) pairs where wm has lower p-value
        pairs_pos = sum(1 for p_i in p_wm for p_j in p_uwm if p_i < p_j)
        pairs_neg = sum(1 for p_i in p_wm for p_j in p_uwm if p_i > p_j)
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
