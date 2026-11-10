#!/bin/bash
set -euo pipefail
cd /home/user
python3 redteam/gpu_checks.py shortcut_quality_tradeoff
python3 redteam/score_gpu.py shortcut_quality_tradeoff
