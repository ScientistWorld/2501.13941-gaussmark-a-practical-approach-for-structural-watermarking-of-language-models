#!/bin/bash
# Standard evaluation entry point: scores every method under
# checkpoints/watermarkers/ on the public slice and writes scoring/scores.json,
# then validates the workspace. Requires one CUDA GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p scoring
python3 eval/train/evaluate.py --output scoring/scores.json
python3 validate.py --compare || true
