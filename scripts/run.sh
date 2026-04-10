#!/bin/bash
# GaussMark reproduction pipeline job
set -e

cd /home/user

echo "=== GaussMark GPU Job ==="
date

# Check GPU
nvidia-smi

# Run the pipeline (skips generation if already done, runs detection)
python3 /home/user/method/run_pipeline.py

echo ""
echo "=== GPU Job Complete ==="
date
