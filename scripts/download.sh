#!/bin/bash
# Download everything this environment needs. Run on a node with internet
# access BEFORE building the container or submitting jobs. Idempotent.
#
# Estimated download: ~14.5 GB (Mistral-7B-Instruct-v0.3 bf16 weights)
#                     + ~30 MB (C4 realnewslike train shard 00000)
#                     + ~15 MB (C4 realnewslike validation shard; source of the
#                       hidden test slice, kept under eval/test/raw/)
# The Mistral repository is gated on Hugging Face: accept its terms and export
# HF_TOKEN (or run `huggingface-cli login`) first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${GAUSSMARK_MODEL_PATH:-$ROOT/data/downloads/models/mistralai/Mistral-7B-Instruct-v0.3}"
C4_DIR="$ROOT/data/downloads/c4_realnewslike"
TEST_RAW="$ROOT/eval/test/raw"
mkdir -p "$MODEL_DIR" "$C4_DIR" "$TEST_RAW"

if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model-00003-of-00003.safetensors" ] \
   && [ -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "Mistral model already present at $MODEL_DIR"
else
    if command -v hf >/dev/null 2>&1; then CLI=(hf download); else CLI=(huggingface-cli download); fi
    "${CLI[@]}" mistralai/Mistral-7B-Instruct-v0.3 --local-dir "$MODEL_DIR" \
        --include "config.json" "generation_config.json" "model.safetensors.index.json" \
                  "model-0000*-of-00003.safetensors" "tokenizer.json" "tokenizer_config.json" \
                  "special_tokens_map.json" "tokenizer.model"
fi

fetch() {  # url dest
    if [ -s "$2" ] && gzip -t "$2" 2>/dev/null; then
        echo "present: $2"
    else
        curl -L --fail --retry 3 -o "$2" "$1"
    fi
}
fetch "https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/c4-train.00000-of-00512.json.gz" \
      "$C4_DIR/c4-train.00000-of-00512.json.gz"
fetch "https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/c4-validation.00000-of-00001.json.gz" \
      "$TEST_RAW/c4-realnewslike-validation.00000-of-00001.json.gz"

echo "Download complete."
