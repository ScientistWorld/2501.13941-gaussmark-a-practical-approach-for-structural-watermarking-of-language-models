#!/bin/bash
# Hidden-test evaluation. Runs with no arguments and scores EVERY method on disk:
# each directory under checkpoints/watermarkers/ (one watermarker.py per method).
#
# The evaluator in eval/test/ loads the fixed model, runs each method's
# make_watermark() and detect() on the held-out prompts, keys and null texts
# that only eval/test/ holds, recomputes every metric, applies the validity and
# quality gates, and writes scoring/scores_test.json (rewritten from scratch on
# every run; a method that fails gets the worst value of every metric).
#
# Requires one CUDA GPU (the method is executed on hidden inputs).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p scoring
python3 eval/test/evaluate.py
