#!/usr/bin/env python3
"""
Detect watermarked text using GaussMark.Detect.

This script loads pre-generated watermarked samples (with their watermark keys)
and runs the GaussMark.Detect procedure to compute p-values and detection rates.

Usage:
    python method/detect.py --results_path /path/to/watermarked_outputs.json

The script:
  1. Loads the original (unwatermarked) model
  2. For each sample, loads the watermark key and generated text
  3. Runs GaussMark.Detect to compute the test statistic psi
  4. Reports detection rates at FPR=0.05 and FPR=0.01, median p-value, and timing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import statistics
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

sys.path.insert(0, str(Path(__file__).parent))
from gaussmark import GaussMark, compute_detection_metrics


# Default paths
DEFAULT_MODEL_PATH = "/home/user/models/mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_RESULTS_PATH = "/home/user/checkpoints/watermarked_outputs.json"


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    Load the HuggingFace model and tokenizer.

    Args:
        model_path:  Path to the model directory.
        device:      Device to load on.
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
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model.eval()
    print(f"[+] Model loaded in {time.time() - t0:.1f}s. "
          f"Device: {next(model.parameters()).device}")

    return model, tokenizer


def load_samples(results_path: str) -> tuple:
    """
    Load generated samples from the JSON results file.

    Args:
        results_path: Path to the watermarked outputs JSON.

    Returns:
        Tuple of (config dict, list of sample dicts).
    """
    print(f"[+] Loading samples from {results_path} ...")
    with open(results_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    samples = data["samples"]
    print(f"[+] Loaded {len(samples)} samples")
    print(f"    Config: layer={config['layer_idx']}, "
          f"weight={config['weight_name']}, sigma={config['sigma']}")
    return config, samples


def load_watermark_key(sample: dict, config: dict) -> np.ndarray:
    """
    Load the watermark key for a sample.

    The key is stored as a .npy file. If the path is not available in the sample,
    regenerate it from the seed.

    Args:
        sample:  Sample dict from the results JSON.
        config:  Config dict from the results JSON.

    Returns:
        Watermark key as numpy array.
    """
    # Try to load from saved .npy file
    key_path = sample.get("watermark_key_path")
    if key_path and os.path.exists(key_path):
        return np.load(key_path)

    # Fallback: regenerate from seed
    seed = sample.get("seed")
    if seed is not None:
        rng = np.random.RandomState(int(seed))
    else:
        rng = np.random.RandomState(42)

    shape = tuple(sample.get("watermark_key_shape", config.get("watermark_key_shape", [])))
    sigma = config.get("sigma", 1e-5)
    return rng.normal(0, sigma, shape).astype(np.float32)


def run_detection(
    model,
    tokenizer,
    samples: list,
    config: dict,
    layer_idx: int,
    weight_name: str,
    sigma: float,
    device: str = "cuda",
) -> dict:
    """
    Run GaussMark.Detect on all samples.

    Args:
        model:        The language model.
        tokenizer:    The tokenizer.
        samples:      List of sample dicts.
        config:       Config dict from generation.
        layer_idx:    Layer index for GaussMark.
        weight_name:  Weight name for GaussMark.
        sigma:        Sigma for GaussMark.
        device:       Device for computation.

    Returns:
        Dict with all p-values, psi values, and detection results.
    """
    print(f"\n[+] Initializing GaussMark detector:")
    print(f"    layer_idx={layer_idx}, weight_name={weight_name}, sigma={sigma}")

    wm = GaussMark(
        model=model,
        layer_idx=layer_idx,
        weight_name=weight_name,
        sigma=sigma,
        device=device,
    )

    p_values = []
    psi_values = []
    grad_norms = []
    dot_products = []
    detection_times = []
    watermarked_flags = []

    print(f"[+] Running detection on {len(samples)} samples ...")
    total_start = time.time()

    for sample in tqdm(samples, desc="  Detecting"):
        prompt_text = sample["prompt_text"]
        generated_ids = sample["generated_ids"]

        # Tokenize prompt
        prompt_enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = prompt_enc["input_ids"]
        attention_mask = prompt_enc.get("attention_mask")

        # Convert generated_ids to tensor
        gen_tensor = torch.tensor([generated_ids]).long()

        # Load watermark key
        watermark_key = load_watermark_key(sample, config)

        # Run detection
        det_start = time.time()
        result = wm.detect(
            input_ids=input_ids,
            generated_ids=gen_tensor,
            watermark_key=watermark_key,
            attention_mask=attention_mask,
            alpha=0.05,
        )
        det_time = time.time() - det_start
        detection_times.append(det_time)

        p_values.append(result["p_value"])
        psi_values.append(result["psi"])
        grad_norms.append(result["grad_norm"])
        dot_products.append(result["dot_product"])
        watermarked_flags.append(result["watermarked"])

    total_time = time.time() - total_start

    # Compute metrics
    metrics = compute_detection_metrics(p_values, thresholds=[0.05, 0.01])

    # Add timing
    metrics["total_detection_time"] = total_time
    metrics["mean_detection_time"] = float(statistics.mean(detection_times))
    metrics["median_detection_time"] = float(statistics.median(detection_times))
    metrics["std_detection_time"] = float(statistics.stdev(detection_times)) if len(detection_times) > 1 else 0.0

    # Add per-sample data
    metrics["p_values"] = p_values
    metrics["psi_values"] = psi_values
    metrics["grad_norms"] = grad_norms
    metrics["dot_products"] = dot_products
    metrics["sample_watermarked"] = watermarked_flags

    return metrics


def print_report(metrics: dict, config: dict) -> None:
    """Print a formatted detection report."""
    print("\n" + "=" * 60)
    print("  GaussMark Detection Report")
    print("=" * 60)

    print(f"\n  Model config:")
    print(f"    Layer:           {config['layer_idx']}")
    print(f"    Weight:          {config['weight_name']}")
    print(f"    Sigma:           {config['sigma']}")
    print(f"    Num tokens:      {config['num_tokens']}")
    print(f"    Num seeds:       {config['num_seeds']}")
    print(f"    Num prompts:     {config['num_prompts']}")
    print(f"    Total samples:   {metrics['num_samples']}")

    print(f"\n  Detection Results:")
    print(f"    Median p-value:     {metrics['median_p_value']:.6f}")
    print(f"    Mean p-value:       {metrics['mean_p_value']:.6f}")
    print(f"    Std p-value:         {metrics['std_p_value']:.6f}")
    print(f"    Min p-value:        {metrics['min_p_value']:.6e}")
    print(f"    Max p-value:        {metrics['max_p_value']:.6f}")
    print(f"    Median psi:         {metrics['median_psi']:.4f}")

    print(f"\n  Detection Rates:")
    print(f"    At FPR=0.05:  {metrics['detection_rate_fpr_0.05']*100:.1f}%  "
          f"({metrics['num_detected_fpr_0.05']}/{metrics['num_samples']})")
    print(f"    At FPR=0.01:  {metrics['detection_rate_fpr_0.01']*100:.1f}%  "
          f"({metrics['num_detected_fpr_0.01']}/{metrics['num_samples']})")

    print(f"\n  Detection Timing:")
    print(f"    Total time:      {metrics['total_detection_time']:.1f}s")
    print(f"    Mean time:       {metrics['mean_detection_time']:.3f}s")
    print(f"    Median time:     {metrics['median_detection_time']:.3f}s")

    print(f"\n  Gradient Stats:")
    print(f"    Mean grad norm:  {np.mean(metrics['grad_norms']):.4f}")
    print(f"    Median grad norm: {np.median(metrics['grad_norms']):.4f}")

    print("=" * 60)


def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Save detection metrics to JSON.

    The per-sample arrays can be large; we save the summary separately
    and optionally the full data.
    """
    # Separate summary from per-sample arrays
    summary = {k: v for k, v in metrics.items()
                if not isinstance(v, (list, np.ndarray))}
    # Remove list items from summary to keep it clean
    for key in ["p_values", "psi_values", "grad_norms", "dot_products", "sample_watermarked"]:
        summary[key + "_summary"] = {
            "mean": float(np.mean(metrics[key])) if len(metrics[key]) > 0 else 0,
            "median": float(np.median(metrics[key])) if len(metrics[key]) > 0 else 0,
            "min": float(np.min(metrics[key])) if len(metrics[key]) > 0 else 0,
            "max": float(np.max(metrics[key])) if len(metrics[key]) > 0 else 0,
        }

    summary_path = output_path.replace(".json", "_detection_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Metrics summary saved to {summary_path}")

    # Also save full p-values for reference
    full_path = output_path.replace(".json", "_detection_full.json")
    full_data = {
        **{k: v for k, v in metrics.items()
           if not isinstance(v, (list, np.ndarray))},
        "p_values": [float(p) for p in metrics["p_values"]],
        "psi_values": [float(psi) for psi in metrics["psi_values"]],
    }
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2)
    print(f"[+] Full detection data saved to {full_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect watermarked text using GaussMark.Detect."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help="Path to the watermarked outputs JSON from watermark_inference.py",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=None,
        help="Layer index (default: read from results JSON or use 20)",
    )
    parser.add_argument(
        "--weight_name",
        type=str,
        default=None,
        help="MLP weight name (default: read from results JSON or use 'up_proj')",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian sigma (default: read from results JSON or use 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--save_full",
        action="store_true",
        help="Save full per-sample detection results to JSON",
    )
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Load samples and config
    config, samples = load_samples(args.results_path)

    # Use CLI args or fall back to config from results file
    layer_idx = args.layer_idx if args.layer_idx is not None else config.get("layer_idx", 20)
    weight_name = args.weight_name if args.weight_name is not None else config.get("weight_name", "up_proj")
    sigma = args.sigma if args.sigma is not None else config.get("sigma", 1e-5)

    # Run detection
    metrics = run_detection(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        config=config,
        layer_idx=layer_idx,
        weight_name=weight_name,
        sigma=sigma,
        device=args.device,
    )

    # Print report
    print_report(metrics, config)

    # Save metrics
    save_metrics(metrics, args.results_path)

    print("\n[+] Detection complete.")


if __name__ == "__main__":
    main()
