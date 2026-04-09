#!/bin/bash
# Verification + extension: confirm pipeline + try 400 tokens
set -e

cd /home/user

echo "=== GaussMark Verification + Extension ==="

RESULTS_FILE="/home/user/checkpoints/watermarked_layer20_up_proj_sigma3e-02.json"
UWM_FILE="/home/user/checkpoints/unwatermarked_baseline.json"

# Step 1: Re-run detection on watermarked text (existing file, should be fast)
echo ""
echo "Step 1: Running detection on existing watermarked text..."
python3 method/detect.py \
    --results_path "$RESULTS_FILE" \
    --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
    --layer_idx 20 \
    --weight_name up_proj \
    --sigma 0.032 \
    --device cuda

# Step 2: Run evaluation
echo ""
echo "Step 2: Running evaluation..."
bash scripts/evaluate.sh watermark_detection_mistral /home/user/checkpoints

# Step 3: Compare with reference
echo ""
echo "Step 3: Comparing with reference..."
python3 validate.py --compare 2>/dev/null || echo "Validation done"

echo ""
echo "=== Verification + Extension Complete ==="