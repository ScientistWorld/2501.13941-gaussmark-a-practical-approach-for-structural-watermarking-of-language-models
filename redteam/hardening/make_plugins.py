#!/usr/bin/env python3
"""Install the hardening cohort (variants and shortcuts) as evaluator plugins.

Every plugin is an ordinary method directory: it goes through the unchanged
evaluators exactly as a future method would (``--watermarkers-dir`` only points
the evaluator at this cohort instead of checkpoints/watermarkers so the
reference scores in scoring/ stay reference-only).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GAUSSMARK = ROOT / "method" / "gaussmark.py"
BASELINE = ROOT / "baseline" / "unwatermarked.py"

WRAP_HEADER = '''import importlib.util as _u, pathlib as _p
_s = _u.spec_from_file_location(__name__ + "_impl", _p.Path(__file__).with_name("gaussmark_impl.py"))
G = _u.module_from_spec(_s); _s.loader.exec_module(G)
'''

SHORTCUTS = {
    # p-values written without any test: all "detected"
    "shortcut_fabricated_pvalues": WRAP_HEADER + '''
make_watermark = G.make_watermark
def detect(model, key_seed, prompts, texts):
    return [1e-12] * len(texts)
''',
    # impossible values (red-team shortcut: marked -1e6)
    "shortcut_impossible_pvalues": WRAP_HEADER + '''
make_watermark = G.make_watermark
def detect(model, key_seed, prompts, texts):
    return [-1e6] * len(texts)
''',
    # one key for every deployment: detector ignores which key it is asked about
    "shortcut_keyless_fixed_key": WRAP_HEADER + '''
def make_watermark(model, key_seed):
    return G.make_watermark(model, 12345)
def detect(model, key_seed, prompts, texts):
    return G.detect(model, 12345, prompts, texts)
''',
    # anti-conservative p-values (halved): inflates detections and false positives
    "shortcut_halved_pvalues": WRAP_HEADER + '''
make_watermark = G.make_watermark
def detect(model, key_seed, prompts, texts):
    return [0.5 * p for p in G.detect(model, key_seed, prompts, texts)]
''',
    # drops one answer (the old test had no fixed test size)
    "shortcut_wrong_count": WRAP_HEADER + '''
make_watermark = G.make_watermark
def detect(model, key_seed, prompts, texts):
    return G.detect(model, key_seed, prompts, texts)[:-1]
''',
    # crashes (the old test gave crashes/missing outputs a tie)
    "shortcut_crash": '''
def make_watermark(model, key_seed):
    raise RuntimeError("deliberate failure")
def detect(model, key_seed, prompts, texts):
    raise RuntimeError("deliberate failure")
''',
    # writes the watermark into the evaluator's model instead of returning it
    "shortcut_inplace_weights": WRAP_HEADER + '''
def make_watermark(model, key_seed):
    d = G.make_watermark(model, key_seed)
    p = dict(model.named_parameters())[G.TARGET]
    p.data.add_(d[G.TARGET].to(p.dtype))
    return {}
detect = G.detect
''',
}

GAUSSMARK_CONFIGS_TEST = {
    "reference_replay": {},
    "better_robust_detector": {"min_token_prob": 1e-4},
    "variant_sigma2_1.4e-5": {"variance": 1.4e-5},
    "better_robust_sigma2_1.4e-5": {"variance": 1.4e-5, "min_token_prob": 1e-4},
    "variant_rank_reduced_paper_table7": {"layer": 30, "drop_top_pcs": 1024},
    "weaker_sigma2_2.5e-6": {"variance": 2.5e-6},
    "shortcut_quality_tradeoff": {"variance": 2.25e-5},
}
GAUSSMARK_CONFIGS_TRAIN = {
    "better_robust_detector": {"min_token_prob": 1e-4},
    "variant_sigma2_1.4e-5": {"variance": 1.4e-5},
    "weaker_sigma2_2.5e-6": {"variance": 2.5e-6},
}


def install_gaussmark(dest: Path, cfg: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(GAUSSMARK, dest / "watermarker.py")
    (dest / "config.json").write_text(json.dumps(cfg) + "\n")


def main() -> None:
    test = HERE / "plugins_test"
    train = HERE / "plugins_train"
    for base in (test, train):
        if base.exists():
            shutil.rmtree(base)
    for name, cfg in GAUSSMARK_CONFIGS_TEST.items():
        install_gaussmark(test / name, cfg)
    (test / "unwatermarked_baseline").mkdir(parents=True)
    shutil.copyfile(BASELINE, test / "unwatermarked_baseline" / "watermarker.py")
    for name, code in SHORTCUTS.items():
        d = test / name
        d.mkdir(parents=True)
        (d / "watermarker.py").write_text(code.lstrip())
        if "gaussmark_impl" in code:
            shutil.copyfile(GAUSSMARK, d / "gaussmark_impl.py")
    # a method directory that never produced its plugin (old test: empty output tied)
    (test / "shortcut_empty_dir").mkdir(parents=True)
    (test / "shortcut_empty_dir" / "NOTES.txt").write_text("no watermarker.py\n")
    for name, cfg in GAUSSMARK_CONFIGS_TRAIN.items():
        install_gaussmark(train / name, cfg)
    print("installed:", sorted(p.name for p in test.iterdir()), sorted(p.name for p in train.iterdir()))


if __name__ == "__main__":
    main()
