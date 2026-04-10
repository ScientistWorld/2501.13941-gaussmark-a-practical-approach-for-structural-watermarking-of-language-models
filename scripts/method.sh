#!/bin/bash
# Run the paper's proposed method (GaussMark).
#
# This script generates watermarked text and runs detection.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

MODEL_PATH="${GAUSSMARK_MODEL_PATH:-$(python3 - <<'PY'
from method.paths import default_model_path
print(default_model_path())
PY
)}"
DATA_PATH="${GAUSSMARK_DATA_PATH:-$(python3 - <<'PY'
from method.paths import default_data_path
print(default_data_path())
PY
)}"
OUTPUT_DIR="/home/user/checkpoints"

echo "=== GaussMark Method ==="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"

# Step 1: Run the end-to-end GaussMark pipeline
echo ""
echo "Step 1: Running generation, detection, and quality check..."
python3 method/run_pipeline.py

# Step 2: Evaluate
echo ""
echo "Step 2: Evaluating..."
bash scripts/evaluate.sh "$OUTPUT_DIR" watermark_detection_mistral

echo ""
echo "=== GaussMark Method Complete ==="
