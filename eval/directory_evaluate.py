#!/usr/bin/env python3
"""Evaluate watermarking result artifacts without importing method code."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any


def load_p_values(path: str) -> tuple[list[float], dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return [float(p) for p in data.get("p_values", [])], data


def detection_metrics(p_values: list[float]) -> dict[str, float]:
    if not p_values:
        return {}
    sorted_p = sorted(p_values)
    n = len(sorted_p)
    median = sorted_p[n // 2] if n % 2 else (sorted_p[n // 2 - 1] + sorted_p[n // 2]) / 2
    return {
        "detection_rate_fpr_0.05": sum(p < 0.05 for p in sorted_p) / n,
        "detection_rate_fpr_0.01": sum(p < 0.01 for p in sorted_p) / n,
        "median_p_value": median,
    }


def pairwise_auc(p_wm: list[float], p_uwm: list[float]) -> float:
    wins = sum(1 for p_i in p_wm for p_j in p_uwm if p_i < p_j)
    ties = sum(1 for p_i in p_wm for p_j in p_uwm if p_i == p_j)
    return (wins + 0.5 * ties) / (len(p_wm) * len(p_uwm))


def latest_watermarked_file(results_dir: str) -> str | None:
    patterns = [
        "marked*_detection_full.json",
        "watermarked*_detection_full.json",
        "*wm*_detection_full.json",
        "*watermark*_detection_full.json",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(results_dir, pattern)))
    candidates = [
        p for p in candidates
        if os.path.isfile(p)
        and "unwatermarked" not in os.path.basename(p)
        and "baseline" not in os.path.basename(p)
        and "robustness" not in os.path.basename(p)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def latest_watermarked_generation_file(results_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(results_dir, "watermarked*.json"))
    candidates = [
        p for p in candidates
        if os.path.isfile(p) and "_detection_" not in os.path.basename(p)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def mean_generation_time(path: str | None) -> float | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    times = [float(t) for t in data.get("generation_times", []) if float(t) > 0]
    if not times:
        return None
    return sum(times) / len(times)


def evaluate(
    results_dir: str,
    experiment: str,
    method_name: str,
    watermarked_file: str | None = None,
    unwatermarked_file: str | None = None,
) -> dict[str, Any]:
    scores: dict[str, Any] = {"experiments": {experiment: {"results": {}}}}

    wm_path = watermarked_file or latest_watermarked_file(results_dir)
    p_wm: list[float] = []
    wm_data: dict[str, Any] = {}
    if wm_path:
        p_wm, wm_data = load_p_values(wm_path)
        if p_wm:
            wm_scores = detection_metrics(p_wm)
            for key in ("auc", "c4_cross_entropy_ratio", "c4_cross_entropy_delta"):
                if key in wm_data:
                    wm_scores[key] = float(wm_data[key])
            scores["experiments"][experiment]["results"][method_name] = wm_scores
            print(f"Watermarked p-values: {len(p_wm)} from {wm_path}")
    else:
        print("No watermarked detection result file found.")

    uwm_path = unwatermarked_file or os.path.join(
        results_dir, "unwatermarked_baseline_unwatermarked_detection_full.json"
    )
    p_uwm: list[float] = []
    if os.path.exists(uwm_path):
        p_uwm, _ = load_p_values(uwm_path)
        if p_uwm:
            uwm_scores = detection_metrics(p_uwm)
            uwm_scores["c4_cross_entropy_ratio"] = 1.0
            uwm_scores["c4_cross_entropy_delta"] = 0.0
            scores["experiments"][experiment]["results"]["unwatermarked"] = uwm_scores
            if method_name in scores["experiments"][experiment]["results"]:
                marked_scores = scores["experiments"][experiment]["results"][method_name]
                marked_scores["unwatermarked_detection_rate_fpr_0.05"] = uwm_scores[
                    "detection_rate_fpr_0.05"
                ]
                marked_scores["unwatermarked_detection_rate_fpr_0.01"] = uwm_scores[
                    "detection_rate_fpr_0.01"
                ]
            print(f"Unwatermarked p-values: {len(p_uwm)} from {uwm_path}")
    else:
        print(f"Unwatermarked result file not found: {uwm_path}")

    if p_wm and p_uwm:
        scores["experiments"][experiment]["results"][method_name]["auc"] = pairwise_auc(p_wm, p_uwm)

    wm_gen_time = mean_generation_time(latest_watermarked_generation_file(results_dir))
    uwm_gen_time = mean_generation_time(os.path.join(results_dir, "unwatermarked_baseline_exact.json"))
    if wm_gen_time and uwm_gen_time and method_name in scores["experiments"][experiment]["results"]:
        scores["experiments"][experiment]["results"][method_name]["generation_time_ratio"] = (
            wm_gen_time / uwm_gen_time
        )
        if "unwatermarked" in scores["experiments"][experiment]["results"]:
            scores["experiments"][experiment]["results"]["unwatermarked"]["generation_time_ratio"] = 1.0

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/home/user/checkpoints")
    parser.add_argument("--experiment", default="watermark_detection_mistral")
    parser.add_argument("--method-name", default="gaussmark_200_tokens")
    parser.add_argument("--output", default="/home/user/scoring/scores.json")
    parser.add_argument("--watermarked-file", default=None)
    parser.add_argument("--unwatermarked-file", default=None)
    args = parser.parse_args()

    scores = evaluate(
        args.results_dir,
        args.experiment,
        args.method_name,
        watermarked_file=args.watermarked_file,
        unwatermarked_file=args.unwatermarked_file,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))
    print(f"Scores written to {args.output}")


if __name__ == "__main__":
    main()
