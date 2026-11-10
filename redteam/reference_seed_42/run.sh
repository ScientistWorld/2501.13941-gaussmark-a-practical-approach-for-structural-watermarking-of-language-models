#!/bin/bash
set -euo pipefail
cd /home/user
python3 redteam/gpu_checks.py reference_seed_42
python3 redteam/score_gpu.py reference_seed_42
