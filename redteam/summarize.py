"""Cohort-normalized rank following the audit's stated scoring rule.

No absolute rank is definable before the future attempt cohort is known. This
file computes an explicit audit cohort score and also gives raw metric vectors.
Signs of coefficients are ignored; higher_is_better controls direction. Missing
metrics are skipped to match the ranking contract. Nonfinite scores are listed.
"""
import math
import statistics

from common import AUDIT, EXPERIMENT, METHOD, read, write

reference = read(AUDIT/'original/scoring/scores_test.json')['experiments'][EXPERIMENT]['results'][METHOD]
config = read(AUDIT/'original/scoring/reference.json')['experiments'][EXPERIMENT]
attempts = {}
for path in sorted(AUDIT.glob('*/scores_test.json')):
    data = read(path)
    attempts[path.parent.name] = data.get('experiments', {}).get(EXPERIMENT, {}).get('results', {}).get(METHOD, {})
scales = {}
for metric in config['metrics']:
    values = [reference[metric]] if metric in reference else []
    values += [row[metric] for row in attempts.values()
               if metric in row and isinstance(row[metric], (float, int)) and math.isfinite(row[metric])]
    if values:
        scales[metric] = statistics.pstdev(values)
summary = {'reference_test': reference, 'normalization': 'population std of all finite audited attempts plus original test reference; absent metrics skipped',
           'metric_scales': scales, 'experiment_weight': config['weight'], 'attempts': {}}
for name, row in attempts.items():
    contributions = {}
    nonfinite = []
    missing = []
    for metric, spec in config['metrics'].items():
        if metric not in row or metric not in reference:
            missing.append(metric)
            continue
        if not math.isfinite(row[metric]):
            nonfinite.append(metric)
            continue
        scale = scales.get(metric, 0)
        if scale > 0:
            sign = 1 if spec['higher_is_better'] else -1
            contributions[metric] = config['weight'] * abs(spec['coefficient']) * sign * (row[metric]-reference[metric])/scale
    summary['attempts'][name] = {'metrics': row, 'rank_delta_finite_terms': sum(contributions.values()),
        'contributions': contributions, 'nonfinite': nonfinite, 'missing': missing,
        'status': read(AUDIT/name/'status.json')}
write(AUDIT/'summary.json', summary)
lines = ['| Run | Δ rank (reference 0) | TPR .05 | TPR .01 | FPR .05 | AUC |',
         '|---|---:|---:|---:|---:|---:|']
for name, item in summary['attempts'].items():
    def fmt(metric):
        v = item['metrics'].get(metric)
        return 'missing' if v is None else f'{v:.6g}'
    lines.append(f"| {name} | {item['rank_delta_finite_terms']:.6f} | {fmt('detection_rate_fpr_0.05')} | {fmt('detection_rate_fpr_0.01')} | {fmt('unwatermarked_detection_rate_fpr_0.05')} | {fmt('auc')} |")
(AUDIT/'table.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
