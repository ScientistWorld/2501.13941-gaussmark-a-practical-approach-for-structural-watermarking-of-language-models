#!/bin/bash
# Public development evaluation. Runs with no arguments and scores every method
# under checkpoints/watermarkers/ on the public train slice (eval/train/data),
# writing scoring/scores_train.json. Same protocol, metrics and gates as the
# hidden test, on 200 public C4 documents and 4 public key seeds.
#
# Requires one CUDA GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p scoring
python3 eval/train/evaluate.py
