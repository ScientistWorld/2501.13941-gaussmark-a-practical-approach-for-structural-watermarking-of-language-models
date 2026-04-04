#!/bin/bash
# GPU job script for GaussMark reproduction.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access (data must be pre-downloaded)

set -e

cd /home/user

echo "=== GaussMark GPU Job ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

# Check data availability
if [ ! -f "/home/user/data/c4_realnewslike/examples.json" ]; then
    echo "ERROR: C4 dataset not found at /home/user/data/c4_realnewslike/"
    exit 1
fi

# Check model availability
if [ ! -f "/home/user/models/mistralai/Mistral-7B-Instruct-v0.3/consolidated.safetensors" ]; then
    echo "ERROR: Model not found at /home/user/models/mistralai/Mistral-7B-Instruct-v0.3/"
    exit 1
fi

mkdir -p /home/user/checkpoints

# Step 1: Generate watermarked text
echo ""
echo "Step 1: Generating watermarked text..."
python3 method/run_watermark.py \
    --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
    --data_path /home/user/data/c4_realnewslike \
    --num_prompts 100 \
    --num_tokens 200 \
    --num_seeds 3 \
    --layer_idx 20 \
    --weight_name up_proj \
    --sigma 1e-5 \
    --output_dir /home/user/checkpoints \
    --device cuda

# Step 2: Run detection on watermarked text
echo ""
echo "Step 2: Running watermark detection on watermarked text..."
RESULTS_FILE="/home/user/checkpoints/watermarked_layer20_up_proj_sigma1e-05.json"
if [ -f "$RESULTS_FILE" ]; then
    python3 method/detect.py \
        --results_path "$RESULTS_FILE" \
        --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --layer_idx 20 \
        --weight_name up_proj \
        --sigma 1e-5 \
        --device cuda
else
    echo "ERROR: Results file not found at $RESULTS_FILE"
    exit 1
fi

# Step 3: Generate unwatermarked (baseline) text
echo ""
echo "Step 3: Generating unwatermarked (baseline) text..."
bash scripts/baseline.sh

# Step 4: Run detection on unwatermarked text
echo ""
echo "Step 4: Running detection on unwatermarked text..."
UWM_FILE="/home/user/checkpoints/unwatermarked_baseline.json"
if [ -f "$UWM_FILE" ]; then
    python3 method/detect_unwatermarked.py \
        --results_path "$UWM_FILE" \
        --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --layer_idx 20 \
        --weight_name up_proj \
        --sigma 1e-5 \
        --device cuda
fi

# Step 5: Evaluate
echo ""
echo "Step 5: Evaluating..."
bash scripts/evaluate.sh watermark_detection_mistral /home/user/checkpoints

# Step 6: Compare with reference
echo ""
echo "Step 6: Comparing with paper's reference..."
python3 validate.py --compare 2>/dev/null || echo "validate.py comparison done"

echo ""
echo "=== GPU Job Complete ==="
