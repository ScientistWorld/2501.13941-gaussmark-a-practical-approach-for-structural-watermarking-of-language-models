#!/usr/bin/env python3
"""Token-corruption robustness sweep for the GaussMark reproduction.

The GaussMark paper evaluates robustness by corrupting generated text and then
re-running the same gradient/key detector. This script reuses the existing
Mistral generation artifacts and tests delete/insert/substitute corruptions on
the exact generated token ids, preserving the paper's detection computation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, "/home/user/method")
from gaussmark import GaussMark  # noqa: E402
from run_pipeline import (  # noqa: E402
    CHECKPOINT_DIR,
    LAYER_IDX,
    MODEL_PATH,
    SIGMA,
    WEIGHT_NAME,
    load_model_and_tokenizer,
)


WM_RESULTS = Path(CHECKPOINT_DIR) / "watermarked_wm_l20_up_proj_s00031622776601683794.json"
UW_RESULTS = Path(CHECKPOINT_DIR) / "unwatermarked_baseline_exact.json"
OUTPUT_PATH = Path(CHECKPOINT_DIR) / "robustness_token_corruption_mistral.json"
MIN_GENERATED_TOKENS = 50
CORRUPTION_SEEDS = {"delete": 8101, "insert": 8102, "substitute": 8103}


def load_samples(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return [
        sample
        for sample in data["samples"]
        if len(sample.get("generated_ids", [])) >= MIN_GENERATED_TOKENS
    ]


def sample_vocab_id(rng: random.Random, vocab_size: int, special_ids: set[int]) -> int:
    while True:
        token_id = rng.randrange(vocab_size)
        if token_id not in special_ids:
            return token_id


def corrupt_generated_ids(
    generated_ids: list[int],
    corruption: str,
    fraction: float,
    rng: random.Random,
    vocab_size: int,
    special_ids: set[int],
) -> list[int]:
    ids = list(generated_ids)
    if not ids:
        raise ValueError("Cannot corrupt an empty generation.")
    n_changes = max(1, int(round(len(ids) * fraction)))

    if corruption == "delete":
        n_changes = min(n_changes, len(ids) - 1)
        drop_positions = set(rng.sample(range(len(ids)), n_changes))
        return [token_id for pos, token_id in enumerate(ids) if pos not in drop_positions]

    if corruption == "substitute":
        for pos in rng.sample(range(len(ids)), min(n_changes, len(ids))):
            ids[pos] = sample_vocab_id(rng, vocab_size, special_ids)
        return ids

    if corruption == "insert":
        insertions: dict[int, list[int]] = {}
        for _ in range(n_changes):
            pos = rng.randrange(len(ids) + 1)
            insertions.setdefault(pos, []).append(sample_vocab_id(rng, vocab_size, special_ids))
        corrupted: list[int] = []
        for pos in range(len(ids) + 1):
            corrupted.extend(insertions.get(pos, []))
            if pos < len(ids):
                corrupted.append(ids[pos])
        return corrupted

    raise ValueError(f"Unknown corruption type: {corruption}")


def load_watermark_key(wm: GaussMark, sample: dict[str, Any]) -> np.ndarray:
    key_path = sample.get("watermark_key_path")
    sample_seed = int(sample.get("sample_seed", sample.get("seed", 42)))
    if key_path and os.path.exists(key_path):
        return torch.load(key_path, weights_only=False, map_location="cpu").numpy()
    return wm.regenerate_watermark_key(sample_seed)


def detect_corrupted_samples(
    wm: GaussMark,
    samples: list[dict[str, Any]],
    corruption: str,
    fraction: float,
    vocab_size: int,
    special_ids: set[int],
    watermarked: bool,
    max_samples: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    p_values: list[float] = []
    details: list[dict[str, Any]] = []
    selected = samples[:max_samples]

    for idx, sample in enumerate(selected):
        if idx % 10 == 0:
            label = "watermarked" if watermarked else "unwatermarked"
            print(f"    {corruption} {label}: {idx}/{len(selected)}")

        sample_seed = int(sample.get("sample_seed", sample.get("seed", 42)))
        rng = random.Random(CORRUPTION_SEEDS[corruption] + sample_seed)
        corrupted_ids = corrupt_generated_ids(
            sample["generated_ids"],
            corruption=corruption,
            fraction=fraction,
            rng=rng,
            vocab_size=vocab_size,
            special_ids=special_ids,
        )

        if watermarked:
            watermark_key = load_watermark_key(wm, sample)
        else:
            # Under the null, the tested text is independent of the Gaussian key.
            watermark_key = wm.regenerate_watermark_key(sample_seed)

        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        generated = torch.tensor([corrupted_ids], dtype=torch.long)
        result = wm.detect(
            input_ids=input_ids,
            generated_ids=generated,
            watermark_key=watermark_key,
            attention_mask=attention_mask,
            alpha=0.05,
        )
        p_values.append(float(result["p_value"]))
        details.append(
            {
                "sample_idx": int(sample.get("sample_idx", idx)),
                "sample_seed": sample_seed,
                "original_tokens": len(sample["generated_ids"]),
                "corrupted_tokens": len(corrupted_ids),
                "psi": float(result["psi"]),
                "p_value": float(result["p_value"]),
            }
        )

    return p_values, details


def p_value_metrics(p_values: list[float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("No p-values to score.")
    return {
        "num_samples": float(len(p_values)),
        "detection_rate_fpr_0.05": float(sum(p < 0.05 for p in p_values) / len(p_values)),
        "detection_rate_fpr_0.01": float(sum(p < 0.01 for p in p_values) / len(p_values)),
        "median_p_value": float(statistics.median(p_values)),
        "mean_p_value": float(statistics.mean(p_values)),
    }


def pairwise_auc(p_wm: list[float], p_uw: list[float]) -> float:
    wins = sum(1 for p_i in p_wm for p_j in p_uw if p_i < p_j)
    ties = sum(1 for p_i in p_wm for p_j in p_uw if p_i == p_j)
    return float((wins + 0.5 * ties) / (len(p_wm) * len(p_uw)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["delete", "insert", "substitute"],
        choices=["delete", "insert", "substitute"],
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    if not WM_RESULTS.exists() or not UW_RESULTS.exists():
        raise FileNotFoundError(
            "Expected existing watermarked and unwatermarked generation artifacts. "
            "Run method/run_pipeline.py first."
        )

    print("=" * 60)
    print("  GaussMark Token-Corruption Robustness")
    print("=" * 60)
    print(f"model={MODEL_PATH}")
    print(f"layer={LAYER_IDX}, weight={WEIGHT_NAME}, sigma={SIGMA} (sigma^2={SIGMA ** 2})")
    print(f"fraction={args.fraction}, max_samples={args.max_samples}")

    wm_samples = load_samples(WM_RESULTS)
    uw_samples = load_samples(UW_RESULTS)
    if not wm_samples or not uw_samples:
        raise RuntimeError("No usable samples found for robustness evaluation.")

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    model.tokenizer = tokenizer
    wm = GaussMark(
        model=model,
        layer_idx=LAYER_IDX,
        weight_name=WEIGHT_NAME,
        sigma=SIGMA,
        device="cuda",
    )

    vocab_size = int(min(getattr(tokenizer, "vocab_size", model.config.vocab_size), model.config.vocab_size))
    special_ids = {int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None}

    started = time.time()
    results: dict[str, Any] = {}
    sample_details: dict[str, Any] = {}
    suffix = f"{int(round(args.fraction * 100))}pct"

    for corruption in args.corruptions:
        print(f"\n[+] Corruption: {corruption}")
        p_wm, wm_details = detect_corrupted_samples(
            wm=wm,
            samples=wm_samples,
            corruption=corruption,
            fraction=args.fraction,
            vocab_size=vocab_size,
            special_ids=special_ids,
            watermarked=True,
            max_samples=args.max_samples,
        )
        p_uw, uw_details = detect_corrupted_samples(
            wm=wm,
            samples=uw_samples,
            corruption=corruption,
            fraction=args.fraction,
            vocab_size=vocab_size,
            special_ids=special_ids,
            watermarked=False,
            max_samples=args.max_samples,
        )

        metrics = p_value_metrics(p_wm)
        null_metrics = p_value_metrics(p_uw)
        metrics.update(
            {
                "auc": pairwise_auc(p_wm, p_uw),
                "unwatermarked_detection_rate_fpr_0.05": null_metrics["detection_rate_fpr_0.05"],
                "unwatermarked_detection_rate_fpr_0.01": null_metrics["detection_rate_fpr_0.01"],
                "unwatermarked_median_p_value": null_metrics["median_p_value"],
            }
        )
        method_name = f"gaussmark_{corruption}_{suffix}"
        results[method_name] = metrics
        sample_details[method_name] = {
            "watermarked": wm_details,
            "unwatermarked": uw_details,
            "watermarked_p_values": p_wm,
            "unwatermarked_p_values": p_uw,
        }
        print(
            f"    TPR@0.05={metrics['detection_rate_fpr_0.05']:.3f}, "
            f"TPR@0.01={metrics['detection_rate_fpr_0.01']:.3f}, "
            f"null@0.05={metrics['unwatermarked_detection_rate_fpr_0.05']:.3f}, "
            f"AUC={metrics['auc']:.3f}"
        )

    output = {
        "experiment": "robustness_token_corruption_mistral",
        "description": "GaussMark detection after token-level random deletions, insertions, and substitutions.",
        "config": {
            "model_path": MODEL_PATH,
            "layer_idx": LAYER_IDX,
            "weight_name": WEIGHT_NAME,
            "sigma": SIGMA,
            "sigma_sq": SIGMA ** 2,
            "fraction": args.fraction,
            "max_samples": args.max_samples,
            "min_generated_tokens": MIN_GENERATED_TOKENS,
            "vocab_size": vocab_size,
        },
        "results": results,
        "sample_details": sample_details,
        "wallclock_time": time.time() - started,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[+] Robustness artifact written to {out_path}")


if __name__ == "__main__":
    main()
