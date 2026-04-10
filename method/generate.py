#!/usr/bin/env python3
"""
GaussMark.Generate: Command-line interface for watermarked text generation.

This script provides a simple CLI for generating watermarked text using the
GaussMark watermarking scheme. It wraps GaussMark.Generate from gaussmark.py.

Usage:
    # Basic usage with defaults (Mistral-7B, layer 20, up_proj, sigma=sqrt(1e-5))
    python method/generate.py --prompt "Once upon a time"

    # With custom parameters
    python method/generate.py \
        --model_path /home/user/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --prompt "The future of AI is" \
        --sigma 0.0031622776601683794 \
        --num_tokens 100 \
        --layer 20 \
        --weight_name up_proj \
        --seed 42

    # Greedy decoding (no sampling)
    python method/generate.py --prompt "Hello world" --no_sample

    # Save watermark key to file
    python method/generate.py --prompt "Test" --save_key /tmp/watermark_key.npy
"""

from __future__ import annotations

import argparse
import sys
import time
import math
from pathlib import Path

import torch
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

sys.path.insert(0, str(Path(__file__).parent))
from gaussmark import GaussMark


DEFAULT_MODEL_PATH = "/home/user/models/mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_SIGMA = math.sqrt(1e-5)


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load the model and tokenizer."""
    print(f"[+] Loading model from {model_path} ...", file=sys.stderr)
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model.eval()
    print(f"[+] Model loaded in {time.time() - t0:.1f}s "
          f"on {next(model.parameters()).device}".format(), file=sys.stderr)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate watermarked text using GaussMark.Generate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python method/generate.py --prompt "The history of artificial intelligence began"
  python method/generate.py --prompt "In a galaxy far far away" --num_tokens 150 --seed 123
  python method/generate.py --prompt "Write a poem about LLMs" --no_sample --temperature 1.0
        """,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt text",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help="Gaussian noise standard deviation (variance = sigma^2). "
             "Default: sqrt(1e-5), because Table 2 reports Mistral-7B variance",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate. Default: 200",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="0-indexed transformer layer to watermark. Default: 20 (per paper Table 2)",
    )
    parser.add_argument(
        "--weight_name",
        type=str,
        default="up_proj",
        choices=["up_proj", "down_proj", "gate_proj"],
        help="MLP weight matrix to perturb. Default: 'up_proj' (per paper Table 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default: 1.0",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold. Default: 0.9",
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use. Default: cuda",
    )
    parser.add_argument(
        "--save_key",
        type=str,
        default=None,
        help="Path to save the watermark key as a .npy file",
    )
    parser.add_argument(
        "--save_text",
        type=str,
        default=None,
        help="Path to save the generated text to a .txt file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information (watermark key shape, timing, etc.)",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    model.tokenizer = tokenizer

    # Initialize GaussMark
    wm = GaussMark(
        model=model,
        layer_idx=args.layer,
        weight_name=args.weight_name,
        sigma=args.sigma,
        device=args.device,
    )

    if args.verbose:
        print(f"[+] GaussMark config:", file=sys.stderr)
        print(f"    layer_idx:     {args.layer}", file=sys.stderr)
        print(f"    weight_name:   {args.weight_name}", file=sys.stderr)
        print(f"    sigma:         {args.sigma}", file=sys.stderr)
        print(f"    key shape:     {wm.target_shape}", file=sys.stderr)
        print(f"    key dim:       {wm.target_shape[0] * wm.target_shape[1]:,} params", file=sys.stderr)
        print(f"    num_tokens:    {args.num_tokens}", file=sys.stderr)
        print(f"    temperature:   {args.temperature}", file=sys.stderr)
        print(f"    top_p:         {args.top_p}", file=sys.stderr)
        print(f"    sample:        {not args.no_sample}", file=sys.stderr)
        if args.seed is not None:
            print(f"    seed:          {args.seed}", file=sys.stderr)

    # Tokenize prompt
    enc = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")

    if args.verbose:
        print(f"[+] Prompt tokens: {input_ids.shape[1]}", file=sys.stderr)

    # Generate
    gen_start = time.time()
    generated_text, watermark_key, metadata = wm.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.num_tokens,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    gen_time = time.time() - gen_start

    # Output the generated text (clean output for piping)
    print(generated_text)

    # Verbose info goes to stderr
    if args.verbose:
        print(f"\n[+] Generation took {gen_time:.3f}s", file=sys.stderr)
        print(f"[+] Generated {metadata['num_generated_tokens']} tokens", file=sys.stderr)
        print(f"[+] Watermark key shape: {watermark_key.shape}", file=sys.stderr)
        print(f"[+] Key norm: {np.linalg.norm(watermark_key):.6e}", file=sys.stderr)
        print(f"[+] Key mean: {watermark_key.mean():.6e}, std: {watermark_key.std():.6e}",
              file=sys.stderr)

    # Save watermark key if requested
    if args.save_key:
        np.save(args.save_key, watermark_key)
        if args.verbose:
            print(f"[+] Watermark key saved to {args.save_key}", file=sys.stderr)

    # Save generated text if requested
    if args.save_text:
        with open(args.save_text, "w") as f:
            f.write(generated_text)
        if args.verbose:
            print(f"[+] Generated text saved to {args.save_text}", file=sys.stderr)

    # Also print metadata to stderr in verbose mode
    if args.verbose:
        print(f"\n[+] Full metadata:", file=sys.stderr)
        for k, v in metadata.items():
            print(f"    {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
