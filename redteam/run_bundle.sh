#!/bin/bash
set -euo pipefail
cd /home/user
python3 redteam/artifact_shortcuts.py all > redteam/artifact_bundle.log 2>&1
python3 redteam/gpu_checks.py reference_seed_42 --tokenizer-only > redteam/tokenizer_validation.log 2>&1
gpu_status=0
# End each model process before scoring its passive artifacts. Save a completed
# test result after every case so a scheduler timeout does not erase progress.
for case in reference_seed_42 honest_efficiency_fix reference_seed_1042 reference_seed_2042 shortcut_quality_tradeoff shortcut_bruteforce_rejection; do
    python3 redteam/gpu_checks.py "$case" > "redteam/${case}/gpu.log" 2>&1 || gpu_status=$?
    python3 redteam/score_gpu.py "$case" > "redteam/${case}/score.log" 2>&1
    python3 redteam/summarize.py > redteam/summary.log
done
python3 redteam/summarize.py
echo "GPU check exit status: ${gpu_status}"
exit "${gpu_status}"
