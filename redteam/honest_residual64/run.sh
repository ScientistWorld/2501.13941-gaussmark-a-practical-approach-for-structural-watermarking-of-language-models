#!/bin/bash
set -euo pipefail
cd /home/user
python3 redteam/gpu_checks.py honest_residual64
python3 redteam/score_gpu.py honest_residual64
