#!/bin/bash
set -euo pipefail

cd /home/user

echo "=== GaussMark Packaging Smoke Test ==="
date

bash /home/user/scripts/evaluate.sh /home/user/checkpoints watermark_detection_mistral gaussmark_200_tokens
python3 /home/user/validate.py --compare --post-design

echo ""
echo "=== Packaging Smoke Test Complete ==="
date
