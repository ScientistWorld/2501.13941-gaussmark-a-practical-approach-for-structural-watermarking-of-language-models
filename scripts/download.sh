#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Estimated total download size: ~15 GB (model ~14.5GB, C4 ~60MB)
# Estimated disk usage after extraction: ~15 GB

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_ROOT="$REPO_ROOT/data/downloads"

C4_DIR="${GAUSSMARK_DATA_PATH:-$DOWNLOAD_ROOT/c4_realnewslike}"
MODEL_DIR="${GAUSSMARK_MODEL_PATH:-$DOWNLOAD_ROOT/models/mistralai/Mistral-7B-Instruct-v0.3}"

mkdir -p "$(dirname "$C4_DIR")" "$(dirname "$MODEL_DIR")"

if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "Mistral model already exists at $MODEL_DIR, skipping."
else
    echo "Downloading Mistral-7B-Instruct-v0.3 model to $MODEL_DIR..."
    mkdir -p "$MODEL_DIR"
    huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
        --local-dir "$MODEL_DIR"
fi

# Check if C4 dataset already exists
if [ -f "$C4_DIR/examples.json" ]; then
    echo "C4 dataset already exists at $C4_DIR, skipping."
else
    echo "Downloading C4 realnewslike shards to $C4_DIR..."
    mkdir -p "$C4_DIR"

    for shard in 00000 00001; do
        file="c4-train.${shard}-of-00512.json.gz"
        url="https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/${file}"
        if [ ! -f "$C4_DIR/$file" ]; then
            curl -L --fail --retry 3 --continue-at - \
                -o "$C4_DIR/$file" \
                "$url"
        fi
    done

    echo "Extracting the first 2000 C4 examples to examples.json..."
    C4_DIR="$C4_DIR" python3 - <<'PY'
import gzip
import json
import os
from pathlib import Path

c4_dir = Path(os.environ["C4_DIR"])
examples = []
for shard in sorted(c4_dir.glob("c4-train.*-of-00512.json.gz")):
    with gzip.open(shard, "rt", encoding="utf-8") as handle:
        for line in handle:
            examples.append(json.loads(line))
            if len(examples) >= 2000:
                break
    if len(examples) >= 2000:
        break

if len(examples) < 2000:
    raise RuntimeError(f"Expected 2000 C4 examples, found {len(examples)}")

with open(c4_dir / "examples.json", "w", encoding="utf-8") as handle:
    json.dump(examples, handle)

print(f"Saved {len(examples)} examples")
PY
fi

echo "Download complete."
