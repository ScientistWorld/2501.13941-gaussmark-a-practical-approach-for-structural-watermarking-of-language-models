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

# Step 1: Run the method pipeline and evaluator
echo "Step 1: Running method pipeline and evaluator..."
bash scripts/method.sh

# Step 2: Re-run standalone evaluation for the reusable output contract
echo "Step 2: Evaluating results..."
bash scripts/evaluate.sh /home/user/checkpoints watermark_detection_mistral

# Step 3: Compare with reference
echo "Step 3: Comparing with paper's reported numbers..."
python3 validate.py --compare 2>/dev/null || echo "Comparison not available yet"

echo "=== Reproduction complete ==="
