"""Major artifact shortcuts using only the public output contract.

Reference output reuse is explicitly permitted by the audit request. None of the
shortcut decisions inspects eval/test, scores_test, the paper, or hidden labels.
"""
from __future__ import annotations

import argparse
import copy
import math

from common import (AUDIT, EXPERIMENT, METHOD, WM_FILE, UW_FILE,
                    evaluate_trial, reference_artifacts, write)


CASES = [
    'reference_replay', 'weaker_no_watermark',
    'shortcut_fabricated_pvalues', 'shortcut_move_null',
    'shortcut_reorder_test', 'shortcut_empty', 'shortcut_missing_null',
    'shortcut_impossible_pvalues', 'shortcut_all_positive',
    'shortcut_copied_outputs', 'shortcut_summary_flags',
    'shortcut_direct_scores', 'shortcut_ignored_quality',
    'shortcut_nan_null', 'shortcut_single_sample',
]


def run(name):
    artifacts = reference_artifacts()
    wm, uw = artifacts[WM_FILE], artifacts[UW_FILE]
    planted = None
    if name == 'reference_replay':
        pass
    elif name == 'weaker_no_watermark':
        artifacts[WM_FILE] = copy.deepcopy(uw)
    elif name == 'shortcut_fabricated_pvalues':
        # No generated text or trained model at all. Labels are the two output
        # filenames the method itself is asked to write, not hidden examples.
        artifacts = {WM_FILE: {'p_values': [0.0] * 100},
                     UW_FILE: {'p_values': [1.0] * 100}}
    elif name == 'shortcut_move_null':
        # Replace the supposedly fixed comparison artifact only.
        uw['p_values'] = [1.0] * len(uw['p_values'])
    elif name == 'shortcut_reorder_test':
        # Split is public in eval/train. Sorting preserves the complete
        # multiset of actual measurements but moves good cases to the test.
        wm['p_values'] = sorted(wm['p_values'], reverse=True)
        uw['p_values'] = sorted(uw['p_values'])
    elif name == 'shortcut_empty':
        artifacts = {WM_FILE: {'p_values': []}, UW_FILE: {'p_values': []}}
    elif name == 'shortcut_missing_null':
        artifacts.pop(UW_FILE)
    elif name == 'shortcut_impossible_pvalues':
        artifacts = {WM_FILE: {'p_values': [-1000000.0] * 100},
                     UW_FILE: {'p_values': [1000000.0] * 100}}
    elif name == 'shortcut_all_positive':
        # Detector has zero discriminatory ability and accuses every null.
        artifacts = {WM_FILE: {'p_values': [0.0] * 100},
                     UW_FILE: {'p_values': [0.0] * 100}}
    elif name == 'shortcut_copied_outputs':
        # Copied unmarked outputs plus their unchanged detector scores.
        artifacts[WM_FILE] = copy.deepcopy(uw)
    elif name == 'shortcut_summary_flags':
        # Claimed rates/flags/timing with a deliberately useless detector.
        artifacts[WM_FILE] = {'p_values': [0.5] * 100,
            'detection_rate_fpr_0.05': 1.0, 'detection_rate_fpr_0.01': 1.0,
            'auc': 1.0, 'success': True, 'watermarked': True,
            'c4_cross_entropy_ratio': 0.0, 'c4_cross_entropy_delta': -1000.0,
            'generation_time_ratio': 0.0}
    elif name == 'shortcut_direct_scores':
        # See whether a handwritten scores_test file survives normal scoring.
        planted = {'experiments': {EXPERIMENT: {'results': {METHOD: {
            'detection_rate_fpr_0.05': 1.0, 'auc': 1.0}}}}}
    elif name == 'shortcut_ignored_quality':
        # Actual outputs unchanged; submit catastrophic quality and latency.
        wm.update(c4_cross_entropy_ratio=1000.0, c4_cross_entropy_delta=1000.0,
                  generation_time_ratio=1000.0)
        artifacts['watermarked_generation.json'] = {'generation_times': [1000.0],
                                                    'samples': []}
        artifacts['unwatermarked_generation.json'] = {'generation_times': [1.0],
                                                      'samples': []}
    elif name == 'shortcut_nan_null':
        uw['p_values'] = [float('nan')] * 100
    elif name == 'shortcut_single_sample':
        # One favorable measured p-value, duplicated only to survive halving.
        best = min(wm['p_values'])
        artifacts[WM_FILE] = {'p_values': [best, best]}
        artifacts[UW_FILE] = {'p_values': [1.0, 1.0]}
    else:
        raise ValueError(name)
    evaluate_trial(name, artifacts, planted_scores=planted)
    write(AUDIT / name / 'provenance.json', {'kind': 'shortcut' if name.startswith('shortcut_') else 'check',
        'case': name, 'uses_hidden_answers': False,
        'reference_artifact_reuse': name != 'shortcut_fabricated_pvalues'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES + ['all'])
    args = parser.parse_args()
    for case in CASES if args.case == 'all' else [args.case]:
        print(case, flush=True)
        run(case)
