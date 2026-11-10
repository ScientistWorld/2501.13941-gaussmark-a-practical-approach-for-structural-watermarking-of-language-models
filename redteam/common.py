"""Install only method output artifacts, invoke the unmodified hidden command.

No evaluator imports, process hooks, environment changes, or background processes.
The original checkpoint directory is moved rather than copied (it holds large keys).
Each trial restores it and the three score files, even on failure.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
EXPERIMENT = 'watermark_detection_mistral'
METHOD = 'gaussmark_200_tokens'
WM_FILE = 'watermarked_shortcut_detection_full.json'
UW_FILE = 'unwatermarked_baseline_unwatermarked_detection_full.json'
SOURCE = AUDIT / 'reference_source'
ORIGINAL_WM = 'wm_wm_l20_up_proj_s00031622776601683794_detection_full.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True) + '\n')


@contextlib.contextmanager
def installed(artifacts, declarations=None):
    checkpoint = ROOT / 'checkpoints'
    saved = AUDIT / '.checkpoint_backup'
    if saved.exists():
        raise RuntimeError('Previous trial did not restore checkpoints; recover before running')
    checkpoint.rename(saved)
    checkpoint.mkdir()
    score_names = ['scores.json', 'scores_train.json', 'scores_test.json']
    saved_scores = {name: (ROOT / 'scoring' / name).read_bytes() for name in score_names}
    try:
        for name, obj in artifacts.items():
            write(checkpoint / name, obj)
        write(ROOT / 'scoring/scores.json', declarations or {
            'experiments': {EXPERIMENT: {'results': {METHOD: {}}}}})
        yield
    finally:
        shutil.rmtree(checkpoint)
        saved.rename(checkpoint)
        for name, content in saved_scores.items():
            (ROOT / 'scoring' / name).write_bytes(content)


def evaluate_trial(name, artifacts, declarations=None, planted_scores=None):
    out = AUDIT / name
    out.mkdir(parents=True, exist_ok=True)
    for filename, value in artifacts.items():
        write(out / 'artifacts' / filename, value)
    with installed(artifacts, declarations):
        if planted_scores is not None:
            write(ROOT / 'scoring/scores_test.json', planted_scores)
        statuses = {}
        # The test invocation has no arguments and is its own new process.
        # A container boundary adds nothing here: every candidate is passive JSON.
        for split in ['train', 'test']:
            with (out / f'{split}.log').open('w') as log:
                proc = subprocess.run(['bash', f'scripts/evaluate_{split}.sh'],
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            statuses[split] = proc.returncode
            shutil.copyfile(ROOT / f'scoring/scores_{split}.json', out / f'scores_{split}.json')
        write(out / 'status.json', statuses)
    return read(out / 'scores_test.json')


def reference_artifacts():
    return {WM_FILE: read(SOURCE / ORIGINAL_WM), UW_FILE: read(SOURCE / UW_FILE)}
