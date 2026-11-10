#!/bin/bash
# Job entry point (1 GPU): reproduce the reference end to end.
# Builds the evaluation slices if missing, installs the unwatermarked baseline
# and GaussMark, scores them on the public slice (scores_train.json, scores.json)
# and on the hidden test (scores_test.json), then validates.
# The hardening verification (variants and shortcuts) is
# redteam/hardening/run_hardening.sh.
set -euo pipefail
cd /home/user
bash scripts/reproduce.sh
