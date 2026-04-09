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
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_PATH',
    torch_dtype=torch.bfloat16,
    device_map='cuda',
)
tokenizer = AutoTokenizer.from_pretrained('$MODEL_PATH')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
model.tokenizer = tokenizer

# Load C4 prompts from JSON
examples_path = os.path.join('$DATA_PATH', 'examples.json')
with open(examples_path, 'r') as f:
    texts = json.load(f)
prompts = [' '.join(str(t).split()[:200]) for t in texts[:100]]
print(f'Loaded {len(prompts)} prompts')

# Generate WITHOUT watermark (just use the original model)
results = {'config': {'watermarked': False, 'num_prompts': len(prompts)}, 'samples': []}

for i, prompt in enumerate(prompts):
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(
            enc['input_ids'].to(model.device),
            attention_mask=enc.get('attention_mask'),
            max_new_tokens=200,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = out[0][enc['input_ids'].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    results['samples'].append({
        'prompt_idx': i,
        'seed': 42 + i,  # Same scheme as watermarked: seed = 42 + prompt_idx
        'prompt_text': prompt[:500],
        'generated_text': gen_text,
        'generated_ids': gen_ids.cpu().tolist(),
        'watermarked': False,
    })
    if (i + 1) % 10 == 0:
        print(f'Generated {i+1}/{len(prompts)}')

# Save
output_file = '$OUTPUT_DIR/unwatermarked_baseline.json'
with open(output_file, 'w') as f:
    json.dump(results, f)
print(f'Saved {len(results[\"samples\"])} unwatermarked samples to {output_file}')
"

echo "Baseline generation complete."
