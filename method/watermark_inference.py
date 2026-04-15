#!/usr/bin/env python3
"""
Generate watermarked text using GaussMark.Generate on C4 prompts.

This script implements the inference-time watermarking described in the paper:
- Loads Mistral-7B-Instruct from the specified path
- Loads the first 100 prompts from the C4 realnewslike dataset
- Truncates each prompt to 200 tokens (as in the paper)
- Runs GaussMark.Generate with sigma=sqrt(1e-5), because Table 2 reports
  variance sigma^2=1e-5 for Mistral
- Saves generated text and watermark keys for later detection

Usage:
    python method/watermark_inference.py \
        --model_path /home/user/data/downloads/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --data_path /home/user/data/downloads/c4_realnewslike \
        --output_path /home/user/checkpoints/watermarked_outputs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import statistics
import math
from pathlib import Path

import torch
import numpy as np

# HuggingFace datasets
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

sys.path.insert(0, str(Path(__file__).parent))
from gaussmark import GaussMark
from paths import default_data_path, default_model_path


# Default paths
DEFAULT_MODEL_PATH = default_model_path()
DEFAULT_DATA_PATH = default_data_path()
DEFAULT_OUTPUT_PATH = "/home/user/checkpoints/watermarked_outputs.json"
DEFAULT_OUTPUT_DIR = "/home/user/checkpoints"
DEFAULT_SIGMA = math.sqrt(1e-5)


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load the HuggingFace model and tokenizer.

    Args:
        model_path: Path to the model directory.
        device:     Device to load the model on.
        torch_dtype: Data type for model weights.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"[+] Loading model from {model_path} ...")
    t0 = time.time()

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model.eval()

    print(f"[+] Model loaded in {time.time() - t0:.1f}s. "
          f"Device: {next(model.parameters()).device}")
    print(f"[+] Model dtype: {torch_dtype}, "
          f"Layers: {config.num_hidden_layers}, "
          f"Hidden size: {config.hidden_size}")

    return model, tokenizer


def load_c4_prompts(
    data_path: str,
    num_prompts: int = 100,
    max_prompt_tokens: int = 200,
) -> list:
    """
    Load prompts from the C4 realnewslike dataset.

    Each prompt is created by truncating the text to max_prompt_tokens tokens,
    as described in the paper (Appendix A).

    Args:
        data_path:        Path to the C4 dataset directory.
        num_prompts:      Number of prompts to load. Default: 100.
        max_prompt_tokens: Maximum tokens per prompt. Default: 200.

    Returns:
        List of prompt strings.
    """
    print(f"[+] Loading C4 prompts from {data_path} ...")
    t0 = time.time()

    if load_dataset is not None:
        try:
            # Try loading as a HuggingFace dataset directory
            ds = load_dataset("json", data_dir=data_path, split="train")
        except Exception:
            try:
                # Try loading from disk
                ds = load_dataset(data_path, split="train")
            except Exception:
                ds = None
    else:
        ds = None

    if ds is None:
        # Fallback: load JSON lines directly
        json_path = Path(data_path)
        if json_path.is_dir():
            json_files = list(json_path.glob("*.json")) + list(json_path.glob("*.jsonl"))
            if json_files:
                json_path = json_files[0]
            else:
                raise FileNotFoundError(f"No JSON files found in {data_path}")

        print(f"[+] Loading from {json_path} ...")
        with open(json_path, "r") as f:
            lines = f.readlines()

        prompts = []
        for line in lines[:num_prompts]:
            obj = json.loads(line)
            text = obj.get("text", "")
            # Truncate to ~max_prompt_tokens tokens
            words = text.split()
            truncated = " ".join(words[:max_prompt_tokens])
            prompts.append(truncated)
    else:
        prompts = []
        for i, example in enumerate(ds):
            if i >= num_prompts:
                break
            text = example.get("text", "")
            words = text.split()
            truncated = " ".join(words[:max_prompt_tokens])
            prompts.append(truncated)

    print(f"[+] Loaded {len(prompts)} prompts in {time.time() - t0:.1f}s")
    return prompts


def run_watermark_generation(
    model,
    tokenizer,
    prompts: list,
    layer_idx: int = 20,
    weight_name: str = "up_proj",
    sigma: float = DEFAULT_SIGMA,
    num_tokens: int = 200,
    num_seeds: int = 3,
    seed_base: int = 42,
    device: str = "cuda",
) -> dict:
    """
    Run GaussMark.Generate on a list of prompts.

    For each prompt and each seed:
      1. Sample watermark key xi ~ N(0, sigma^2 * I)
      2. Perturb the target MLP layer
      3. Generate text with the watermarked model
      4. Restore original weights

    Args:
        model:        The language model.
        tokenizer:    The tokenizer.
        prompts:      List of input prompt strings.
        layer_idx:    Transformer layer to watermark. Default: 20.
        weight_name:  MLP weight name to perturb. Default: 'up_proj'.
        sigma:        Gaussian noise standard deviation. Default: sqrt(1e-5).
        num_tokens:   Number of tokens to generate. Default: 200.
        num_seeds:    Number of seeds (watermark keys) per prompt. Default: 3.
        seed_base:    Base seed for reproducibility. Default: 42.
        device:       Device for computation.

    Returns:
        Dict with config, samples, and timing statistics.
    """
    print(f"\n[+] Initializing GaussMark:")
    print(f"    layer_idx={layer_idx}, weight_name={weight_name}, sigma={sigma}")
    print(f"    num_tokens={num_tokens}, num_seeds={num_seeds}")
    print(f"    num_prompts={len(prompts)}")

    wm = GaussMark(
        model=model,
        layer_idx=layer_idx,
        weight_name=weight_name,
        sigma=sigma,
        device=device,
    )

    # Attach tokenizer to model for generation
    if not hasattr(model, "tokenizer"):
        model.tokenizer = tokenizer

    results = {
        "config": {
            "layer_idx": layer_idx,
            "weight_name": weight_name,
            "sigma": sigma,
            "sigma_sq": float(sigma ** 2),
            "num_tokens": num_tokens,
            "num_seeds": num_seeds,
            "num_prompts": len(prompts),
            "total_samples": len(prompts) * num_seeds,
        },
        "samples": [],
        "generation_times": [],
    }

    sample_idx = 0
    for seed_idx in range(num_seeds):
        seed = seed_base + seed_idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n[+] Seed {seed_idx + 1}/{num_seeds} (seed={seed})")
        rng = np.random.RandomState(seed)

        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                print(f"  Seed {seed}: {i}/{len(prompts)}")
            # Tokenize prompt
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")

            # Track per-sample seed for key regeneration
            sample_seed = seed + i

            # Generate watermarked text
            gen_start = time.time()
            generated_text, watermark_key, metadata = wm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                seed=sample_seed,
            )
            gen_time = time.time() - gen_start
            results["generation_times"].append(gen_time)

            generated_ids = metadata["generated_ids"]

            # Save the watermark key to a .npy file (too large for JSON)
            keys_dir = Path(results.get("_keys_dir", DEFAULT_OUTPUT_DIR)) / "watermark_keys"
            keys_dir.mkdir(parents=True, exist_ok=True)
            key_path = keys_dir / f"key_{sample_idx:04d}_seed{seed}.npy"
            np.save(key_path, watermark_key)

            sample = {
                "sample_idx": sample_idx,
                "prompt_idx": i,
                "seed": seed,
                "sample_seed": sample_seed,
                "prompt_text": prompt[:1000],  # Store truncated for space
                "generated_text": generated_text,
                "generated_ids": generated_ids,
                "watermark_key_shape": list(watermark_key.shape),
                "watermark_key_path": str(key_path),
                "gen_time": gen_time,
                "num_generated_tokens": len(generated_ids),
                **metadata,
            }
            results["samples"].append(sample)
            sample_idx += 1

    # Summary statistics
    times = results["generation_times"]
    results["timing_stats"] = {
        "mean_gen_time": float(statistics.mean(times)),
        "median_gen_time": float(statistics.median(times)),
        "std_gen_time": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        "min_gen_time": float(min(times)),
        "max_gen_time": float(max(times)),
        "total_gen_time": float(sum(times)),
        "total_samples": len(times),
    }

    return results


def save_results(results: dict, output_path: str) -> None:
    """Save results to JSON, with key paths tracked in the JSON."""
    output_dir = os.path.dirname(output_path) or "."
    keys_dir = os.path.join(output_dir, "watermark_keys")
    results["_keys_dir"] = keys_dir
    results["_keys_glob"] = "watermark_keys/key_*.npy"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(keys_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Results saved to {output_path}")
    print(f"[+] Watermark keys saved to {keys_dir}/")

    # Print timing summary
    stats = results["timing_stats"]
    print(f"\n[+] Generation timing summary:")
    print(f"    Total samples:   {stats['total_samples']}")
    print(f"    Total time:     {stats['total_gen_time']:.1f}s")
    print(f"    Mean time:      {stats['mean_gen_time']:.3f}s")
    print(f"    Median time:    {stats['median_gen_time']:.3f}s")
    print(f"    Std dev:        {stats['std_gen_time']:.3f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate watermarked text using GaussMark on C4 prompts."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the Mistral-7B model directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the C4 realnewslike dataset directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the generated outputs JSON",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=100,
        help="Number of C4 prompts to process (default: 100)",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=200,
        help="Number of tokens to generate per prompt (default: 200)",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="Number of watermark keys per prompt (default: 3, as in paper)",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=20,
        help="Transformer layer to watermark (default: 20, per paper Table 2)",
    )
    parser.add_argument(
        "--weight_name",
        type=str,
        default="up_proj",
        help="MLP weight name to perturb (default: 'up_proj', per paper Table 2)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help="Gaussian noise std dev (default: sqrt(1e-5), because Table 2 reports Mistral variance)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    model.tokenizer = tokenizer

    # Load C4 prompts
    prompts = load_c4_prompts(
        args.data_path,
        num_prompts=args.num_prompts,
        max_prompt_tokens=200,  # Always 200 tokens as per paper
    )

    # Generate watermarked text
    total_start = time.time()
    results = run_watermark_generation(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        layer_idx=args.layer_idx,
        weight_name=args.weight_name,
        sigma=args.sigma,
        num_tokens=args.num_tokens,
        num_seeds=args.num_seeds,
        device=args.device,
    )
    total_time = time.time() - total_start
    print(f"\n[+] Total wall-clock time: {total_time:.1f}s")
    results["wallclock_time"] = total_time

    # Save results
    save_results(results, args.output_path)

    print("\n[+] Done. Run detection with:")
    print(f"    python method/detect.py --results_path {args.output_path}")


if __name__ == "__main__":
    main()
