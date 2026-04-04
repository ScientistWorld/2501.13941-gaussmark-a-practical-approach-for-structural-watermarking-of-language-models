#!/bin/bash
# Run baseline (unwatermarked) generation for comparison.
#
# The baseline is the unwatermarked model - text generated without any watermark.
# Detection on baseline text should give ~5% false positives at FPR=0.05.
#
# Usage:
#   bash scripts/baseline.sh

set -e

cd /home/user

MODEL_PATH="/home/user/models/mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH="/home/user/data/c4_realnewslike"
OUTPUT_DIR="/home/user/checkpoints"

echo "=== Baseline (Unwatermarked) ==="

# Generate unwatermarked text
echo "Generating unwatermarked text..."
python3 -c "
import sys
sys.path.insert(0, 'method')
from gaussmark import GaussMark
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_from_disk
import json
import time
import numpy as np

# Load model
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_PATH',
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained('$MODEL_PATH')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
model.tokenizer = tokenizer

# Load prompts
print('Loading C4 prompts...')
ds = load_from_disk('$DATA_PATH')
prompts = [' '.join(ex['text'].split()[:200]) for ex in ds[:100]]

# Generate WITHOUT watermark (just use the original model)
results = {'config': {'watermarked': False, 'num_prompts': len(prompts)}, 'samples': []}

for i, prompt in enumerate(prompts[:100]):
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(
            enc['input_ids'].to(model.device),
            attention_mask=enc.get('attention_mask', None),
            max_new_tokens=200,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
        )
    gen_ids = out[0][enc['input_ids'].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    results['samples'].append({
        'prompt_idx': i,
        'prompt_text': prompt[:500],
        'generated_text': gen_text,
        'generated_ids': gen_ids.cpu().tolist(),
        'watermarked': False,
    })

# Save
output_file = '$OUTPUT_DIR/unwatermarked_baseline.json'
with open(output_file, 'w') as f:
    json.dump(results, f)
print(f'Saved {len(results[\"samples\"])} unwatermarked samples to {output_file}')
"

echo "Baseline generation complete."
echo "Now run detection to verify false positive rate is ~5%."
