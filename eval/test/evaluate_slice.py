#!/usr/bin/env python3
"""Held-out test-slice evaluator for GaussMark p-value artifacts.

Computes detection metrics on the second half of samples only (sorted by index).
Global properties (generation_time_ratio, c4_cross_entropy_*) are not split
by design — they are per-generation properties, not per-sample properties.
The test slice evaluates the same samples split by index.
"""

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


def reference_method_metrics(
    scores_path: str,
    experiment: str,
    method_name: str,
) -> dict[str, float]:
    if not os.path.exists(scores_path):
        return {}
    with open(scores_path) as f:
        data = json.load(f)
    experiments = data.get("experiments", {})
    results = experiments.get(experiment, {}).get("results", {})
    method = results.get(method_name, {})
    return {
        key: float(value)
        for key, value in method.items()
        if isinstance(value, (int, float))
    }


def heldout_slice(p_values: list[float]) -> list[float]:
    """Second half of samples (by index order)."""
    split_at = (len(p_values) + 1) // 2
    return p_values[split_at:]


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
        "mean_p_value": sum(sorted_p) / n,
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


def candidate_watermarked_files(results_dir: str) -> list[str]:
    patterns = [
        "marked*_detection_full.json",
        "watermarked*_detection_full.json",
        "*wm*_detection_full.json",
        "*watermark*_detection_full.json",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(results_dir, pattern)))
    deduped = []
    seen: set[str] = set()
    for path in sorted(candidates):
        base = os.path.basename(path)
        if (
            not os.path.isfile(path)
            or "unwatermarked" in base
            or "baseline" in base
            or "robustness" in base
        ):
            continue
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def derive_method_name(filename: str) -> str:
    """Derive a clean method name from a detection full filename.

    Handles patterns like:
      watermarked_layer20_up_proj_sigma1e-04_detection_full.json
      wm_wm_l20_up_proj_s0001_detection_full.json
      watermarked_wm_l20_up_proj_s00031622776601683794_detection_full.json
    Returns the path prefix that uniquely identifies the method variant.
    """
    fname = os.path.basename(filename)
    name = fname.replace("_detection_full.json", "")
    for prefix in ("watermarked_", "wm_wm_", "wm_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name if name else fname.replace("_detection_full.json", "")


def full_file_metrics(path: str) -> dict[str, float]:
    p_values, data = load_p_values(path)
    metrics = detection_metrics(p_values)
    for key in ("auc", "c4_cross_entropy_ratio", "c4_cross_entropy_delta"):
        if key in data and isinstance(data[key], (int, float)):
            metrics[key] = float(data[key])
    return metrics


def resolve_watermarked_file(results_dir: str, experiment: str, method_name: str) -> str | None:
    candidates = candidate_watermarked_files(results_dir)
    if not candidates:
        return None

    exact_matches = [
        path for path in candidates
        if os.path.basename(path).replace("_detection_full.json", "") == method_name
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    reference = reference_method_metrics(
        "/home/user/scoring/scores.json", experiment, method_name
    )
    if not reference:
        return latest_watermarked_file(results_dir)

    def candidate_distance(path: str) -> tuple[float, int, float]:
        metrics = full_file_metrics(path)
        overlap = sorted(set(reference) & set(metrics))
        if not overlap:
            return (float("inf"), 0, -os.path.getmtime(path))

        error = 0.0
        for key in overlap:
            ref_val = float(reference[key])
            cand_val = float(metrics[key])
            scale = max(abs(ref_val), 1e-6)
            error += abs(cand_val - ref_val) / scale
        return (error, -len(overlap), -os.path.getmtime(path))

    return min(candidates, key=candidate_distance)


def evaluate(
    results_dir: str,
    experiment: str,
    method_name: str,
    watermarked_file: str | None = None,
    unwatermarked_file: str | None = None,
    slice_name: str = "test",
) -> dict[str, Any]:
    scores: dict[str, Any] = {
        "experiments": {experiment: {"results": {}}},
        "_slice": slice_name,
        "_audit": {experiment: {}},
    }

    wm_path = watermarked_file or resolve_watermarked_file(results_dir, experiment, method_name)
    p_wm: list[float] = []
    if wm_path:
        all_p_wm, _ = load_p_values(wm_path)
        p_wm = heldout_slice(all_p_wm)
        if p_wm:
            wm_scores = detection_metrics(p_wm)
            scores["experiments"][experiment]["results"][method_name] = wm_scores
            scores["_audit"][experiment][method_name] = {
                "slice": slice_name,
                "file": os.path.basename(wm_path),
                "num_samples_total": len(all_p_wm),
                "num_samples_used": len(p_wm),
            }
            print(f"Test: {len(p_wm)}/{len(all_p_wm)} p-values from {os.path.basename(wm_path)}")
    else:
        print("No watermarked detection result file found.")

    # Unwatermarked baseline — also slice to second half for consistency
    uwm_path = unwatermarked_file or os.path.join(
        results_dir, "unwatermarked_baseline_unwatermarked_detection_full.json"
    )
    p_uwm: list[float] = []
    if os.path.exists(uwm_path):
        all_p_uwm, _ = load_p_values(uwm_path)
        p_uwm = heldout_slice(all_p_uwm)
        if p_uwm:
            uwm_scores = detection_metrics(p_uwm)
            scores["experiments"][experiment]["results"]["unwatermarked"] = uwm_scores
            scores["_audit"][experiment]["unwatermarked"] = {
                "slice": slice_name,
                "file": os.path.basename(uwm_path),
                "num_samples_total": len(all_p_uwm),
                "num_samples_used": len(p_uwm),
            }
            if method_name in scores["experiments"][experiment]["results"]:
                marked_scores = scores["experiments"][experiment]["results"][method_name]
                marked_scores["unwatermarked_detection_rate_fpr_0.05"] = uwm_scores[
                    "detection_rate_fpr_0.05"
                ]
                marked_scores["unwatermarked_detection_rate_fpr_0.01"] = uwm_scores[
                    "detection_rate_fpr_0.01"
                ]
                marked_scores["detection_margin_fpr_0.05"] = (
                    marked_scores["detection_rate_fpr_0.05"]
                    - uwm_scores["detection_rate_fpr_0.05"]
                )
            print(f"Test: {len(p_uwm)}/{len(all_p_uwm)} unwatermarked p-values from {os.path.basename(uwm_path)}")
    else:
        print(f"Unwatermarked result file not found: {uwm_path}")

    if p_wm and p_uwm:
        scores["experiments"][experiment]["results"][method_name]["auc"] = pairwise_auc(p_wm, p_uwm)

    # generation_time_ratio and c4_cross_entropy_* are per-generation global
    # properties, not per-sample. They are identical across train and test splits
    # by construction and therefore provide zero anti-shortcut signal.
    # We omit them from the train/test scoring entirely to keep the split meaningful.
    # The original evaluate.sh computes these for reference scoring.

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/home/user/checkpoints")
    parser.add_argument("--experiment", default="watermark_detection_mistral")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--output", default="/home/user/scoring/scores_test.json")
    parser.add_argument("--watermarked-file", default=None)
    parser.add_argument("--unwatermarked-file", default=None)
    args = parser.parse_args()

    # Auto-derive method name from watermarked file if not provided
    method_name = args.method_name
    if not method_name and args.watermarked_file:
        method_name = derive_method_name(args.watermarked_file)
    if not method_name:
        method_name = "unknown_method"

    scores = evaluate(
        args.results_dir,
        args.experiment,
        method_name,
        watermarked_file=args.watermarked_file,
        unwatermarked_file=args.unwatermarked_file,
        slice_name="test",
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))
    print(f"Test scores written to {args.output}")


if __name__ == "__main__":
    main()
