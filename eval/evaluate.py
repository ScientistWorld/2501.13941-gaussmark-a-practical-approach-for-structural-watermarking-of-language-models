#!/usr/bin/env python3
"""
Evaluation code for GaussMark watermarking.

This module evaluates watermarking performance:
1. Detection rate (TPR at FPR=0.05 and FPR=0.01) for watermarked text
2. False positive rate for unwatermarked text
3. Model quality (perplexity on held-out data)

The eval/ directory contains evaluation logic that does NOT depend on the
method's implementation. It uses the same interface that a scientist agent
would use when evaluating their own method.
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Optional
import argparse


def evaluate_detection(
    p_values_watermarked: List[float],
    p_values_unwatermarked: List[float],
) -> Dict[str, float]:
    """
    Evaluate watermark detection performance.

    Args:
        p_values_watermarked: P-values from watermarked text detection
        p_values_unwatermarked: P-values from unwatermarked text detection

    Returns:
        Dictionary of metrics
    """
    p_wm = np.array(p_values_watermarked)
    p_uwm = np.array(p_values_unwatermarked)

    metrics = {}

    # True Positive Rate at different FPR thresholds
    for thresh in [0.05, 0.01, 0.001]:
        metrics[f"tpr_fpr_{thresh}"] = float(np.mean(p_wm < thresh))
        metrics[f"fpr_fpr_{thresh}"] = float(np.mean(p_uwm < thresh))

    # Detection statistics
    metrics["median_p_watermarked"] = float(np.median(p_wm))
    metrics["median_p_unwatermarked"] = float(np.median(p_uwm))
    metrics["mean_p_watermarked"] = float(np.mean(p_wm))
    metrics["mean_p_unwatermarked"] = float(np.mean(p_uwm))

    # Number of samples
    metrics["num_watermarked"] = len(p_wm)
    metrics["num_unwatermarked"] = len(p_uwm)

    # Count detected at p < 0.05
    metrics["num_detected_watermarked"] = int(np.sum(p_wm < 0.05))
    metrics["num_detected_unwatermarked"] = int(np.sum(p_uwm < 0.05))

    return metrics


def compute_roc_auc(p_values_watermarked: List[float], p_values_unwatermarked: List[float]) -> float:
    """Compute ROC AUC for detection."""
    p_wm = np.array(p_values_watermarked)
    p_uwm = np.array(p_values_unwatermarked)

    # For ROC: lower p-value = more likely to be watermarked
    # Concatenate and compute AUC
    labels = np.concatenate([np.ones(len(p_wm)), np.zeros(len(p_uwm))])
    scores = np.concatenate([p_wm, p_uwm])

    # AUC with inverted scores (lower p = higher score)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels, -scores)
    except:
        auc = 0.5

    return auc


def evaluate_p_values(p_values: List[float], thresholds: List[float] = [0.05, 0.01, 0.001]) -> Dict[str, float]:
    """
    Evaluate detection results from p-values.

    Args:
        p_values: List of p-values from detection
        thresholds: List of FPR thresholds

    Returns:
        Dictionary of metrics
    """
    p = np.array(p_values)
    metrics = {}

    for thresh in thresholds:
        detected = np.sum(p < thresh)
        metrics[f"detection_rate_fpr_{thresh}"] = float(detected / len(p))
        metrics[f"num_detected_fpr_{thresh}"] = int(detected)

    metrics["median_p_value"] = float(np.median(p))
    metrics["mean_p_value"] = float(np.mean(p))
    metrics["std_p_value"] = float(np.std(p))
    metrics["num_samples"] = len(p)

    return metrics


def load_p_values(results_path: str) -> Dict[str, List[float]]:
    """
    Load p-values from detection results.

    Expected format: {
        "watermarked": {"p_values": [...], ...},
        "unwatermarked": {"p_values": [...], ...}
    }
    """
    if not os.path.exists(results_path):
        return {}

    with open(results_path, "r") as f:
        data = json.load(f)

    return data


def run_evaluation(
    watermarked_results: str,
    unwatermarked_results: Optional[str] = None,
    output_path: str = "/home/user/scoring/scores.json",
) -> Dict[str, Any]:
    """
    Run full evaluation and write scores.json.

    Args:
        watermarked_results: Path to watermarked detection results
        unwatermarked_results: Path to unwatermarked detection results
        output_path: Path to write scores.json

    Returns:
        Evaluation metrics
    """
    results = {}

    # Load watermarked p-values
    if os.path.exists(watermarked_results):
        with open(watermarked_results, "r") as f:
            data = json.load(f)
        if "p_values" in data:
            p_wm = data["p_values"]
        elif "samples" in data:
            p_wm = [s.get("p_value", 1.0) for s in data["samples"]]
        else:
            p_wm = []
    else:
        p_wm = []

    # Load unwatermarked p-values
    if unwatermarked_results and os.path.exists(unwatermarked_results):
        with open(unwatermarked_results, "r") as f:
            data = json.load(f)
        if "p_values" in data:
            p_uwm = data["p_values"]
        elif "samples" in data:
            p_uwm = [s.get("p_value", 1.0) for s in data["samples"]]
        else:
            p_uwm = []
    else:
        p_uwm = []

    # Evaluate
    if p_wm:
        metrics = evaluate_p_values(p_wm)
        results["watermarked_detection"] = metrics

        if p_uwm:
            from sklearn.metrics import roc_auc_score
            labels = np.concatenate([np.ones(len(p_wm)), np.zeros(len(p_uwm))])
            scores = np.concatenate([p_wm, p_uwm])
            try:
                auc = roc_auc_score(labels, -scores)
                results["watermarked_detection"]["auc"] = float(auc)
            except:
                pass

    if p_uwm:
        metrics_uwm = evaluate_p_values(p_uwm)
        results["unwatermarked_detection"] = metrics_uwm

    # Write scores.json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GaussMark watermarking")
    parser.add_argument("--watermarked_results", type=str, required=True)
    parser.add_argument("--unwatermarked_results", type=str, default=None)
    parser.add_argument("--output", type=str, default="/home/user/scoring/scores.json")
    args = parser.parse_args()

    results = run_evaluation(
        args.watermarked_results,
        args.unwatermarked_results,
        args.output,
    )

    print("\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
