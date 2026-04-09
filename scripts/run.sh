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

# Parameters
LAYER_IDX=20
WEIGHT_NAME=up_proj
NUM_PROMPTS=20
NUM_TOKENS=200

# NOTE: The paper specifies sigma^2 (variance), not sigma.
# Paper for Mistral-7B: sigma^2 = 1e-3, so sigma = sqrt(1e-3) = 0.0316.
# My code's sigma parameter IS sigma (std dev), so sigma=0.032 is correct.
SIGMA_DISPLAY="3e-2"    # Display name (matches Python's .0e format: 0.032 -> "3e-2")
SIGMA_VALUE="0.032"    # sigma = sqrt(1e-3) ≈ 0.032

RESULTS_FILE="/home/user/checkpoints/watermarked_layer${LAYER_IDX}_${WEIGHT_NAME}_sigma${SIGMA_DISPLAY}.json"
UWM_FILE="/home/user/checkpoints/unwatermarked_baseline.json"

# Step 1: Generate watermarked text with sigma=1e-4
echo ""
echo "Step 1: Generating watermarked text (sigma=1e-4)..."
if [ -f "$RESULTS_FILE" ]; then
    echo "    Results file exists, skipping generation..."
else
    python3 method/run_watermark.py \
        --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --data_path /home/user/data/c4_realnewslike \
        --num_prompts $NUM_PROMPTS \
        --num_tokens $NUM_TOKENS \
        --num_seeds 1 \
        --layer_idx $LAYER_IDX \
        --weight_name $WEIGHT_NAME \
        --sigma $SIGMA_VALUE \
        --output_dir /home/user/checkpoints \
        --device cuda
fi

# Step 2: Run detection on watermarked text
echo ""
echo "Step 2: Running watermark detection on watermarked text..."
if [ -f "$RESULTS_FILE" ]; then
    python3 method/detect.py \
        --results_path "$RESULTS_FILE" \
        --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --layer_idx $LAYER_IDX \
        --weight_name $WEIGHT_NAME \
        --sigma $SIGMA_VALUE \
        --device cuda
else
    echo "ERROR: Results file not found at $RESULTS_FILE"
    exit 1
fi

# Step 3: Generate unwatermarked (baseline) text
echo ""
echo "Step 3: Generating unwatermarked (baseline) text..."
if [ ! -f "$UWM_FILE" ]; then
    bash scripts/baseline.sh
else
    echo "    Unwatermarked baseline already exists, skipping..."
fi

# Step 4: Run detection on unwatermarked text
echo ""
echo "Step 4: Running detection on unwatermarked text..."
if [ -f "$UWM_FILE" ]; then
    python3 method/detect_unwatermarked.py \
        --results_path "$UWM_FILE" \
        --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --layer_idx $LAYER_IDX \
        --weight_name $WEIGHT_NAME \
        --sigma $SIGMA_VALUE \
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
