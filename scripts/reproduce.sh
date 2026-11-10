#!/bin/bash
# End-to-end reproduction (one GPU): install the baseline and GaussMark, then
# score them on the public slice and the hidden test.
#
# Assumes scripts/download.sh has been run on a node with internet access.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 eval/train/build_train_set.py      # no-op when eval/train/data/train_set.json exists
python3 eval/test/build_test_set.py        # no-op when eval/test/data/test_set.json exists
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh
python3 validate.py --compare --post-design
