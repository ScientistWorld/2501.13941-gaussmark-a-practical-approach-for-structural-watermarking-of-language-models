#!/bin/bash
# Run the paper's proposed method (GaussMark).
#
# This script generates watermarked text and runs detection.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

MODEL_PATH="/home/user/models/mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH="/home/user/data/c4_realnewslike"
OUTPUT_DIR="/home/user/checkpoints"

echo "=== GaussMark Method ==="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"

# Step 1: Generate watermarked text
echo ""
echo "Step 1: Generating watermarked text..."
python3 method/run_watermark.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --num_prompts 20 \
    --num_tokens 200 \
    --num_seeds 1 \
    --layer_idx 20 \
    --weight_name up_proj \
    --sigma 0.032 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

# Step 2: Run detection
echo ""
echo "Step 2: Running watermark detection..."
RESULTS_FILE="$OUTPUT_DIR/watermarked_layer20_up_proj_sigma3e-02.json"
if [ -f "$RESULTS_FILE" ]; then
    python3 method/detect.py \
        --results_path "$RESULTS_FILE" \
        --model_path "$MODEL_PATH" \
        --layer_idx 20 \
        --weight_name up_proj \
        --sigma 0.032 \
        --device cuda
else
    echo "Warning: Results file not found at $RESULTS_FILE"
fi

# Step 3: Evaluate
echo ""
echo "Step 3: Evaluating..."
bash scripts/evaluate.sh watermark_detection_mistral "$OUTPUT_DIR"

echo ""
echo "=== GaussMark Method Complete ==="
