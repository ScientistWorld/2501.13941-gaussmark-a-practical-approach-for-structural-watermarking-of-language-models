#!/usr/bin/env python3
"""
Run GaussMark watermarking experiments.

This script generates watermarked text from the C4 dataset using GaussMark
and saves the results for later detection evaluation.

Usage:
    python method/run_watermark.py --model_path /path/to/model --num_prompts 100 --num_tokens 200
"""

import argparse
import json
import os
import sys
import time
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np

# Add method to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gaussmark import GaussMark, compute_p_value


DEFAULT_SIGMA = math.sqrt(1e-5)


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer


def load_c4_prompts(data_path: str, num_prompts: int = 100):
    """Load C4 prompts, truncating to 200 tokens as in the paper."""
    import json

    print(f"Loading C4 dataset from {data_path}...")
    # Try examples.json first (list of text strings)
    examples_path = os.path.join(data_path, "examples.json")
    if os.path.exists(examples_path):
        with open(examples_path, "r") as f:
            texts = json.load(f)
        prompts = []
        for i, text in enumerate(texts):
            if i >= num_prompts:
                break
            # Truncate to first 200 words (paper uses first 200 tokens as prompt)
            truncated = " ".join(str(text).split()[:200])
            prompts.append(truncated)
    else:
        raise FileNotFoundError(f"No examples.json found in {data_path}")

    print(f"Loaded {len(prompts)} prompts")
    return prompts


def run_experiment(
    model,
    tokenizer,
    prompts: list,
    layer_idx: int = 20,
    weight_name: str = "up_proj",
    sigma: float = DEFAULT_SIGMA,
    num_tokens: int = 200,
    num_seeds: int = 3,
    device: str = "cuda",
    output_dir: str = "/home/user/checkpoints",
):
    """Run watermarking experiments with checkpointing (save after each seed)."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize GaussMark
    wm = GaussMark(
        model=model,
        layer_idx=layer_idx,
        weight_name=weight_name,
        sigma=sigma,
        device=device,
    )

    output_path = os.path.join(output_dir, f"watermarked_layer{layer_idx}_{weight_name}_sigma{sigma:.0e}.json")

    # Load existing checkpoint if available (for resumability)
    if os.path.exists(output_path):
        print(f"[+] Loading existing checkpoint: {output_path}")
        with open(output_path, "r") as f:
            all_results = json.load(f)
        print(f"    Already have {len(all_results['samples'])} samples")
    else:
        all_results = {
            "config": {
                "layer_idx": layer_idx,
                "weight_name": weight_name,
                "sigma": sigma,
                "num_tokens": num_tokens,
                "num_seeds": num_seeds,
                "num_prompts": len(prompts),
            },
            "samples": [],
            "generation_times": [],
        }

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx
        print(f"\n=== Seed {seed_idx} (seed={seed}) ===")
        np.random.seed(seed)
        torch.manual_seed(seed)

        for i, prompt in enumerate(prompts):
            if i % 5 == 0:
                print(f"  Seed {seed_idx}: {i}/{len(prompts)}")

            # Skip if already generated
            existing = [s for s in all_results["samples"]
                      if s.get("prompt_idx") == i and s.get("seed") == seed + i]
            if existing:
                continue

            # Tokenize prompt
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")

            gen_start = time.time()
            generated_text, watermark_key, metadata = wm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                seed=seed + i,
            )
            gen_time = time.time() - gen_start

            generated_ids = metadata["generated_ids"]

            sample = {
                "prompt_idx": i,
                "seed": seed + i,
                "prompt_text": prompt[:500],
                # Don't store full generated_text to save memory
                "generated_ids": generated_ids,
                "watermark_key_shape": list(watermark_key.shape),
                "watermark_key_dtype": str(watermark_key.dtype),
                "gen_time": gen_time,
                "num_generated_tokens": len(generated_ids),
            }
            # Save watermark key to .npy file for reliable detection
            key_path = os.path.join(output_dir, f"key_layer{layer_idx}_{weight_name}_sigma{sigma:.0e}_seed{seed + i}.npy")
            np.save(key_path, watermark_key)
            sample["watermark_key_path"] = key_path
            all_results["samples"].append(sample)
            all_results["generation_times"].append(gen_time)

        # Checkpoint after each seed
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"    Checkpoint saved ({len(all_results['samples'])} samples)")

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    import statistics
    times = all_results["generation_times"]
    print(f"\nGeneration time stats:")
    print(f"  Mean: {statistics.mean(times):.3f}s")
    print(f"  Median: {statistics.median(times):.3f}s")
    print(f"  Total: {sum(times):.1f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run GaussMark watermarking experiments")
    sys.path.insert(0, str(Path(__file__).parent))
    from paths import default_data_path, default_model_path

    parser.add_argument("--model_path", type=str, default=default_model_path())
    parser.add_argument("--data_path", type=str, default=default_data_path())
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--num_tokens", type=int, default=200)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--weight_name", type=str, default="up_proj")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--output_dir", type=str, default="/home/user/checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    prompts = load_c4_prompts(args.data_path, args.num_prompts)

    # Attach tokenizer to model for generation
    model.tokenizer = tokenizer

    results = run_experiment(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        layer_idx=args.layer_idx,
        weight_name=args.weight_name,
        sigma=args.sigma,
        num_tokens=args.num_tokens,
        num_seeds=args.num_seeds,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
