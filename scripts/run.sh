#!/bin/bash
# GaussMark reproduction pipeline job
set -e

cd /home/user

echo "=== GaussMark GPU Job ==="
date

# Check GPU
nvidia-smi

# Reuse the completed core-claim generation artifacts and test the paper's
# token-corruption robustness setting on a bounded sample.
python3 /home/user/method/run_robustness.py --fraction 0.25 --max-samples 40

# Keep the already-scored core-claim contract valid while the robustness artifact
# remains a separate candidate for the next milestone.
python3 /home/user/validate.py --compare

echo ""
echo "=== GPU Job Complete ==="
date
