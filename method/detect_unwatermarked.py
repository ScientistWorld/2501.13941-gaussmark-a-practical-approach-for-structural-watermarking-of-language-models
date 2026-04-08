#!/usr/bin/env python3
"""
Detect watermarked text using GaussMark.Detect — for UNWATERMARKED baseline.

This script loads unwatermarked samples and runs GaussMark.Detect with
regenerated watermark keys (from seeds). Since the text was generated
WITHOUT watermark perturbation, p-values should be uniformly distributed
under H0, giving ~5% false positive rate at FPR=0.05.

Usage:
    python method/detect_unwatermarked.py --results_path /path/to/unwatermarked.json
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

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

sys.path.insert(0, str(Path(__file__).parent))
from gaussmark import GaussMark, compute_detection_metrics


def load_model_and_tokenizer(model_path: str, device: str = "cuda", torch_dtype=torch.bfloat16):
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
    model.eval()
    print(f"[+] Model loaded in {time.time() - t0:.1f}s. Device: {next(model.parameters()).device}")
    return model, tokenizer


def run_detection_unwatermarked(
    model, tokenizer, samples: list,
    layer_idx: int, weight_name: str, sigma: float, device: str = "cuda",
) -> dict:
    print(f"\n[+] Running detection on UNWATERMARKED samples:")
    print(f"    layer_idx={layer_idx}, weight_name={weight_name}, sigma={sigma}")

    wm = GaussMark(
        model=model, layer_idx=layer_idx,
        weight_name=weight_name, sigma=sigma, device=device,
    )

    p_values, psi_values, grad_norms, dot_products = [], [], [], []
    detection_times = []

    print(f"[+] Running detection on {len(samples)} unwatermarked samples...")
    total_start = time.time()

    for i, sample in enumerate(samples):
        if i % 5 == 0:
            print(f"  Detecting: {i}/{len(samples)}")
        prompt_text = sample["prompt_text"]
        generated_ids = sample["generated_ids"]

        prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = prompt_enc["input_ids"]
        attention_mask = prompt_enc.get("attention_mask")
        gen_tensor = torch.tensor([generated_ids]).long()

        # Regenerate watermark key from seed.
        # The key is generated with the OUTER seed (42 + seed_idx), not the per-prompt seed.
        prompt_idx = sample.get("prompt_idx", 0)
        seed = sample.get("seed", 42)
        outer_seed = int(seed) - int(prompt_idx)  # key uses outer seed
        watermark_key = wm.regenerate_watermark_key(outer_seed)

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

    total_time = time.time() - total_start

    metrics = compute_detection_metrics(p_values, thresholds=[0.05, 0.01])
    metrics["total_detection_time"] = total_time
    metrics["mean_detection_time"] = float(statistics.mean(detection_times))
    metrics["median_detection_time"] = float(statistics.median(detection_times))
    metrics["p_values"] = p_values
    metrics["psi_values"] = psi_values

    return metrics


def print_report(metrics: dict, num_samples: int) -> None:
    print("\n" + "=" * 60)
    print("  Unwatermarked Baseline — Detection Report")
    print("=" * 60)
    print(f"\n  Total samples:  {num_samples}")
    print(f"\n  Detection Results (should be ~uniform p-values):")
    print(f"    Median p-value:  {metrics['median_p_value']:.4f}")
    print(f"    Mean p-value:    {metrics['mean_p_value']:.4f}")
    print(f"    Std p-value:     {metrics['std_p_value']:.4f}")
    print(f"    Min p-value:     {metrics['min_p_value']:.6e}")
    print(f"    Max p-value:     {metrics['max_p_value']:.6f}")
    print(f"\n  False Positive Rates:")
    print(f"    At FPR=0.05:  {metrics['detection_rate_fpr_0.05']*100:.1f}%  "
          f"({metrics['num_detected_fpr_0.05']}/{metrics['num_samples']})")
    print(f"    At FPR=0.01:  {metrics['detection_rate_fpr_0.01']*100:.1f}%  "
          f"({metrics['num_detected_fpr_0.01']}/{metrics['num_samples']})")
    print(f"\n  Detection Timing:")
    print(f"    Total time:  {metrics['total_detection_time']:.1f}s")
    print(f"    Mean time:   {metrics['mean_detection_time']:.3f}s")
    print("=" * 60)


def save_metrics(metrics: dict, output_path: str) -> None:
    summary = {k: v for k, v in metrics.items() if not isinstance(v, (list, np.ndarray))}
    summary_path = output_path.replace(".json", "_unwatermarked_detection_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Metrics saved to {summary_path}")

    full_path = output_path.replace(".json", "_unwatermarked_detection_full.json")
    full_data = {
        **{k: v for k, v in metrics.items() if not isinstance(v, (list, np.ndarray))},
        "p_values": [float(p) for p in metrics["p_values"]],
        "psi_values": [float(psi) for psi in metrics["psi_values"]],
    }
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2)
    print(f"[+] Full data saved to {full_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect watermarked text (unwatermarked baseline).")
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/home/user/models/mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--weight_name", type=str, default="up_proj")
    parser.add_argument("--sigma", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    with open(args.results_path, "r") as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"[+] Loaded {len(samples)} unwatermarked samples")

    metrics = run_detection_unwatermarked(
        model=model, tokenizer=tokenizer, samples=samples,
        layer_idx=args.layer_idx, weight_name=args.weight_name,
        sigma=args.sigma, device=args.device,
    )

    print_report(metrics, len(samples))
    save_metrics(metrics, args.results_path)
    print("\n[+] Detection complete.")


if __name__ == "__main__":
    main()
