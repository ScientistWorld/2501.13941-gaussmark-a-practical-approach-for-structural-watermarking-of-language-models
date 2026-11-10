#!/bin/bash
# Install the paper's method (GaussMark, Mistral-7B settings of Table 2:
# layer 20 up_proj, sigma^2 = 1e-5) as checkpoints/watermarkers/gaussmark_200_tokens/.
# GaussMark needs no training: the evaluators run its Generate/Detect directly.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 method/export_watermarker.py --name gaussmark_200_tokens
