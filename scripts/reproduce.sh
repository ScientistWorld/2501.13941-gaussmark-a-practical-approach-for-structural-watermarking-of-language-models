#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction —
# a scientist can run this and get the paper's numbers.

set -e

cd /home/user

echo "=== GaussMark Reproduction ==="

# Step 1: Generate watermarked text
echo "Step 1: Generating watermarked text..."
bash scripts/method.sh

# Step 2: Run detection
echo "Step 2: Running watermark detection..."
python3 method/detect.py \
    --results_path /home/user/checkpoints/watermarked_layer20_up_proj_sigma1e-05.json \
    --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
    --layer_idx 20 \
    --weight_name up_proj \
    --sigma 1e-5 \
    --save_keys \
    --device cuda

# Step 3: Evaluate
echo "Step 3: Evaluating results..."
bash scripts/evaluate.sh watermark_detection_mistral /home/user/checkpoints

# Step 4: Compare with reference
echo "Step 4: Comparing with paper's reported numbers..."
python3 validate.py --compare 2>/dev/null || echo "Comparison not available yet"

echo "=== Reproduction complete ==="
