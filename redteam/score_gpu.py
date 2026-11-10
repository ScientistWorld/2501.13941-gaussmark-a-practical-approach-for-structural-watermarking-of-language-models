"""Score passive outputs after the GPU method process has exited."""
import argparse
from common import AUDIT, read, evaluate_trial

CASES = ['reference_seed_42', 'reference_seed_1042', 'reference_seed_2042',
         'honest_efficiency_fix', 'honest_residual64', 'shortcut_bruteforce_rejection']
p = argparse.ArgumentParser()
p.add_argument('case', choices=CASES+['all'])
args = p.parse_args()
for name in CASES if args.case == 'all' else [args.case]:
    path = AUDIT/name/'artifacts'
    if not path.exists():
        print('Not completed:', name, flush=True)
        continue
    artifacts = {f.name: read(f) for f in path.glob('*.json')}
    evaluate_trial(name, artifacts)
