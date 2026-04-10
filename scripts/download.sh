#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Estimated total download size: ~30 GB (model ~28GB, C4 ~100MB)
# Estimated disk usage after extraction: ~30 GB

set -e

SHARED_C4_DIR="/home/user/shared/datasets/c4_realnewslike"
LEGACY_C4_DIR="/home/user/data/c4_realnewslike"
SHARED_MODEL_DIR="/home/user/shared/models/mistralai/Mistral-7B-Instruct-v0.3"
LEGACY_MODEL_DIR="/home/user/models/mistralai/Mistral-7B-Instruct-v0.3"

C4_DIR="${GAUSSMARK_DATA_PATH:-$SHARED_C4_DIR}"
MODEL_DIR="${GAUSSMARK_MODEL_PATH:-$SHARED_MODEL_DIR}"

mkdir -p "$(dirname "$C4_DIR")" "$(dirname "$MODEL_DIR")"

if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "Mistral model already exists at $MODEL_DIR, skipping."
elif [ -f "$LEGACY_MODEL_DIR/config.json" ] && ls "$LEGACY_MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "Mistral model already exists at legacy path $LEGACY_MODEL_DIR."
    echo "Set GAUSSMARK_MODEL_PATH=$LEGACY_MODEL_DIR or copy it to $MODEL_DIR if shared storage is required."
else
    echo "Downloading Mistral-7B-Instruct-v0.3 model to $MODEL_DIR..."
    mkdir -p "$MODEL_DIR"
    huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
        --local-dir "$MODEL_DIR"
fi

# Check if C4 dataset already exists
if [ -f "$C4_DIR/examples.json" ]; then
    echo "C4 dataset already exists at $C4_DIR, skipping."
elif [ -f "$LEGACY_C4_DIR/examples.json" ]; then
    echo "C4 dataset already exists at legacy path $LEGACY_C4_DIR."
    echo "Set GAUSSMARK_DATA_PATH=$LEGACY_C4_DIR or copy it to $C4_DIR if shared storage is required."
else
    echo "Downloading C4 realnewslike dataset (2000 examples) to $C4_DIR..."
    mkdir -p "$C4_DIR"

    # Method 1: Try streaming load (uses less memory)
    echo "Attempting streaming download..."
    if python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('allenai/c4', 'realnewslike', split='train[:2000]', streaming=True)
data = [ex for _, ex in zip(range(2000), ds)]
from datasets import Dataset
ds2 = Dataset.from_list(data)
ds2.save_to_disk('$C4_DIR')
with open('$C4_DIR/examples.json', 'w') as f:
    json.dump(data, f)
print(f'Saved {len(ds2)} examples')
" 2>/dev/null; then
        echo "C4 dataset downloaded successfully (streaming)."
    else
        echo "Streaming failed, trying direct download..."
        # Method 2: Direct load (may use more memory)
        if python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('allenai/c4', 'realnewslike', split='train[:2000]')
ds.save_to_disk('$C4_DIR')
with open('$C4_DIR/examples.json', 'w') as f:
    json.dump(list(ds), f)
print(f'Saved {len(ds)} examples')
" 2>/dev/null; then
            echo "C4 dataset downloaded successfully (direct)."
        else
            echo "Direct C4 download also failed. Refusing to create a synthetic fallback for the primary reproduction." >&2
            exit 1
        fi
    fi
fi

echo "Download complete."
