#!/bin/bash
set -euo pipefail

cd /home/user

echo "=== GaussMark Split Evaluation Job ==="
date

rm -f /home/user/scoring/scores_train.json /home/user/scoring/scores_test.json

bash /home/user/scripts/evaluate_train.sh /home/user/checkpoints
bash /home/user/scripts/evaluate_test.sh /home/user/checkpoints

python3 /home/user/validate.py --compare --post-design

echo ""
echo "=== Split Evaluation Complete ==="
date
