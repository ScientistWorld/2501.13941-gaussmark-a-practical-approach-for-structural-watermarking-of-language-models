#!/bin/bash
set -euo pipefail
cd /home/user
python3 redteam/artifact_shortcuts.py all > redteam/artifact_bundle.log 2>&1
gpu_status=0
python3 redteam/gpu_checks.py all > redteam/gpu_bundle.log 2>&1 || gpu_status=$?
python3 redteam/score_gpu.py all > redteam/gpu_score.log 2>&1
python3 redteam/summarize.py
echo "GPU check exit status: ${gpu_status}"
exit "${gpu_status}"
