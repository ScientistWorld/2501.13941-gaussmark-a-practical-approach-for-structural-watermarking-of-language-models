#!/bin/bash
# Install the unwatermarked baseline (no weight perturbation; same detector run
# against an unused key) as checkpoints/watermarkers/unwatermarked/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p checkpoints/watermarkers/unwatermarked
cp baseline/unwatermarked.py checkpoints/watermarkers/unwatermarked/watermarker.py
echo "installed checkpoints/watermarkers/unwatermarked"
