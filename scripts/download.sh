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

C4_DIR="/home/user/data/c4_realnewslike"
MODEL_DIR="/home/user/models/mistralai/Mistral-7B-Instruct-v0.3"

if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "Mistral model already exists, skipping."
else
    echo "Downloading Mistral-7B-Instruct-v0.3 model..."
    mkdir -p "$MODEL_DIR"
    huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
        --local-dir "$MODEL_DIR"
fi

# Check if C4 dataset already exists
if [ -f "$C4_DIR/dataset_info.json" ] || [ -f "$C4_DIR/.arrow" ]; then
    echo "C4 dataset already exists, skipping."
else
    echo "Downloading C4 realnewslike dataset (2000 examples)..."

    # Method 1: Try streaming load (uses less memory)
    echo "Attempting streaming download..."
    if python3 -c "
from datasets import load_dataset
ds = load_dataset('allenai/c4', 'realnewslike', split='train[:2000]', streaming=True)
data = [ex for _, ex in zip(range(2000), ds)]
from datasets import Dataset
ds2 = Dataset.from_list(data)
ds2.save_to_disk('/home/user/data/c4_realnewslike')
print(f'Saved {len(ds2)} examples')
" 2>/dev/null; then
        echo "C4 dataset downloaded successfully (streaming)."
    else
        echo "Streaming failed, trying direct download..."
        # Method 2: Direct load (may use more memory)
        if python3 -c "
from datasets import load_dataset
ds = load_dataset('allenai/c4', 'realnewslike', split='train[:2000]')
ds.save_to_disk('/home/user/data/c4_realnewslike')
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
