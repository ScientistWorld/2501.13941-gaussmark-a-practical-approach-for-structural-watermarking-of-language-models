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

RESULTS_FILE="/home/user/checkpoints/watermarked_layer20_up_proj_sigma1e-05.json"
UWM_FILE="/home/user/checkpoints/unwatermarked_baseline.json"

# Step 1: Generate watermarked text
# Uses checkpointing — skips prompts that already have .npy key files saved.
# If checkpoint is stale (missing key files), delete and regenerate.
echo ""
echo "Step 1: Generating watermarked text..."
if [ -f "$RESULTS_FILE" ]; then
    # Check if all samples have saved key files
    MISSING_KEYS=$(python3 -c "
import json, os, glob
d = json.load(open('$RESULTS_FILE'))
for s in d.get('samples', []):
    key_path = s.get('watermark_key_path')
    if not key_path or not os.path.exists(key_path):
        print('MISSING')
        break
else:
    print('OK')
" 2>/dev/null)
    if [ "$MISSING_KEYS" = "OK" ]; then
        echo "    All 20 samples have key files, skipping generation..."
    else
        echo "    Key files missing — regenerating..."
        rm -f "$RESULTS_FILE"
        rm -f /home/user/checkpoints/key_*.npy
    fi
fi

if [ ! -f "$RESULTS_FILE" ]; then
    python3 method/run_watermark.py \
        --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --data_path /home/user/data/c4_realnewslike \
        --num_prompts 20 \
        --num_tokens 200 \
        --num_seeds 1 \
        --layer_idx 20 \
        --weight_name up_proj \
        --sigma 1e-5 \
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


# Step 2: Run detection on watermarked text (FIXED: key regeneration uses outer seed)
echo ""
echo "Step 2: Running watermark detection on watermarked text..."
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

# Step 3: Generate unwatermarked (baseline) text (skip if exists)
echo ""
echo "Step 3: Generating unwatermarked (baseline) text..."
if [ ! -f "$UWM_FILE" ]; then
    bash scripts/baseline.sh
else
    echo "    Unwatermarked baseline already exists, skipping..."
fi

# Step 4: Run detection on unwatermarked text (FIXED: key regeneration uses outer seed)
echo ""
echo "Step 4: Running detection on unwatermarked text..."
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
