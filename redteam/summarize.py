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
statuses = {}
for path in sorted(AUDIT.glob('*/scores_test.json')):
    data = read(path)
    attempts[path.parent.name] = data.get('experiments', {}).get(EXPERIMENT, {}).get('results', {}).get(METHOD, {})
    statuses[path.parent.name] = read(path.parent/'status.json')
scales = {}
for metric in config['metrics']:
    values = [reference[metric]] if metric in reference else []
    values += [row[metric] for name, row in attempts.items()
               if statuses[name]['test'] == 0 and metric in row
               and isinstance(row[metric], (float, int)) and math.isfinite(row[metric])]
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
    valid = statuses[name]['test'] == 0 and not nonfinite
    summary['attempts'][name] = {'metrics': row,
        'rank_delta_finite_terms': sum(contributions.values()) if valid else None,
        'contributions': contributions, 'nonfinite': nonfinite, 'missing': missing,
        'status': statuses[name], 'test_success': valid,
        'score_file_may_be_stale': statuses[name]['test'] != 0}
seeds = {name: item for name, item in summary['attempts'].items()
         if name.startswith('reference_seed_') and item['test_success']}
summary['seed_spread'] = {'completed': list(seeds), 'metrics': {}}
for metric in reference:
    values = [item['metrics'][metric] for item in seeds.values() if metric in item['metrics']]
    if values:
        summary['seed_spread']['metrics'][metric] = {
            'min': min(values), 'max': max(values), 'range': max(values)-min(values),
            'population_std': statistics.pstdev(values)}
ranks = [item['rank_delta_finite_terms'] for item in seeds.values()]
if ranks:
    summary['seed_spread']['rank'] = {'min': min(ranks), 'max': max(ranks),
                                    'range': max(ranks)-min(ranks)}
write(AUDIT/'summary.json', summary)
lines = ['| Run | Δ rank (reference 0) | TPR .05 | TPR .01 | FPR .05 | AUC |',
         '|---|---:|---:|---:|---:|---:|']
for name, item in summary['attempts'].items():
    def fmt(metric):
        v = item['metrics'].get(metric)
        return 'missing' if v is None else f'{v:.6g}'
    rank = item['rank_delta_finite_terms']
    rank_text = 'failed / invalid' if rank is None else f'{rank:.6f}'
    lines.append(f"| {name} | {rank_text} | {fmt('detection_rate_fpr_0.05')} | {fmt('detection_rate_fpr_0.01')} | {fmt('unwatermarked_detection_rate_fpr_0.05')} | {fmt('auc')} |")
(AUDIT/'table.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
