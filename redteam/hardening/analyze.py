#!/usr/bin/env python3
"""Rank the hardening cohort and measure paired noise.

Rank (the gym's formula): sum over experiments e and scored metrics m of
weight_e * |coefficient_m| * improvement_m / std_m, where improvement is taken
in the metric's better direction relative to the reference and std_m is the
population standard deviation of the metric across the cohort.  Coefficient-0
metrics (validity diagnostics) are not ranked.

Noise: paired bootstrap over the hidden test documents (every method sees the
same documents, keys and sampling noise), recomputing every metric from the
per-text p-values and per-document cross-entropies the evaluator dumped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = Path(__file__).resolve().parent / "results"
REF_NAME = "reference_replay"
B = 1000
MAIN, ROB = "watermark_detection_mistral", "watermark_robustness_mistral"


def load(p):
    with open(p) as f:
        return json.load(f)


def scored_metrics(ref):
    out = []
    for e, spec in ref["experiments"].items():
        for m, d in spec["metrics"].items():
            if abs(d["coefficient"]) > 0:
                out.append((e, m, spec["weight"], abs(d["coefficient"]), 1.0 if d["higher_is_better"] else -1.0))
    return out


def cohort_rank(scores, metrics, ref_name, members=None):
    names = list(scores)
    members = members or names
    stds = {}
    for e, m, *_ in metrics:
        vals = np.array([scores[n][e][m] for n in members], dtype=float)
        stds[(e, m)] = float(vals.std())
    ranks = {}
    for n in names:
        r = 0.0
        for e, m, w, c, s in metrics:
            if stds[(e, m)] > 0:
                r += w * c * s * (scores[n][e][m] - scores[ref_name][e][m]) / stds[(e, m)]
        ranks[n] = r
    return ranks, stds


# ---- weighted metric helpers for the bootstrap ------------------------------
def w_rate(p, w, a):
    return float((w * (p < a)).sum() / w.sum())


def w_median(p, w):
    o = np.argsort(p)
    cw = np.cumsum(w[o])
    return float(p[o][np.searchsorted(cw, 0.5 * cw[-1])])


def w_auc(pos, wp, neg, wn):
    o = np.argsort(neg)
    q, wq = neg[o], wn[o]
    cw = np.concatenate([[0.0], np.cumsum(wq)])
    W = cw[-1]
    right = np.searchsorted(q, pos, "right")
    left = np.searchsorted(q, pos, "left")
    greater = W - cw[right]
    equal = cw[right] - cw[left]
    return float((wp * (greater + 0.5 * equal)).sum() / (wp.sum() * W))


def metrics_from_dump(d, counts):
    pv = {k: np.array(v, float) for k, v in d["pvalues"].items()}
    di = {k: np.array(v, int) for k, v in d["doc_index"].items()}
    w = {k: counts[di[k]].astype(float) for k in pv}
    nulls = ("null_unwatermarked", "null_human", "null_cross_key")
    pn = np.concatenate([pv[k] for k in nulls])
    wn = np.concatenate([w[k] for k in nulls])
    nll_w = np.array(d["nll_watermarked"], float)
    nll_0 = np.array(d["nll_original"], float)
    return {
        MAIN: {
            "detection_rate_fpr_0.05": w_rate(pv["marked"], w["marked"], 0.05),
            "detection_rate_fpr_0.01": w_rate(pv["marked"], w["marked"], 0.01),
            "auc": w_auc(pv["marked"], w["marked"], pn, wn),
            "median_p_value": w_median(pv["marked"], w["marked"]),
            "c4_cross_entropy_ratio": float((counts * nll_w).sum() / (counts * nll_0).sum()),
        },
        ROB: {
            "detection_rate_fpr_0.05": w_rate(pv["marked_corrupted"], w["marked_corrupted"], 0.05),
            "detection_rate_fpr_0.01": w_rate(pv["marked_corrupted"], w["marked_corrupted"], 0.01),
            "auc": w_auc(pv["marked_corrupted"], w["marked_corrupted"], pv["null_corrupted"], w["null_corrupted"]),
            "median_p_value": w_median(pv["marked_corrupted"], w["marked_corrupted"]),
        },
    }


def main():
    ref = load(ROOT / "scoring" / "reference.json")
    metrics = scored_metrics(ref)
    cohort = load(RES / "scores_test_cohort.json")
    status = cohort["_status"]
    names = sorted(status)
    scores = {n: {e: cohort["experiments"][e]["results"][n] for e in (MAIN, ROB)} for n in names}
    ok = [n for n in names if status[n]["status"] == "ok"]
    rank_all, std_all = cohort_rank(scores, metrics, REF_NAME)
    rank_ok, std_ok = cohort_rank(scores, metrics, REF_NAME, members=ok)

    # determinism: reference replay vs the reference row of scoring/scores_test.json
    official = load(ROOT / "scoring" / "scores_test.json")
    off_ref = {e: official["experiments"][e]["results"].get("gaussmark_200_tokens") for e in (MAIN, ROB)}
    determinism = {e: {m: [off_ref[e][m], scores[REF_NAME][e][m]] for m in off_ref[e]} for e in off_ref
                   if off_ref[e]}

    # paired bootstrap over documents
    dumps = {n: load(RES / "dump_test" / f"{n}.json") for n in ok if (RES / "dump_test" / f"{n}.json").is_file()}
    rng = np.random.default_rng(0)
    ndoc = len(next(iter(dumps.values()))["nll_original"]) if dumps else 0
    boot = {n: [] for n in dumps}
    for _ in range(B if dumps else 0):
        counts = np.bincount(rng.integers(0, ndoc, ndoc), minlength=ndoc)
        vals = {n: metrics_from_dump(dumps[n], counts) for n in dumps}
        for n in dumps:
            r = 0.0
            for e, m, w, c, s in metrics:
                if std_all[(e, m)] > 0:
                    r += w * c * s * (vals[n][e][m] - vals[REF_NAME][e][m]) / std_all[(e, m)]
            boot[n].append({"rank_delta": r,
                            "main_tpr05_delta": vals[n][MAIN]["detection_rate_fpr_0.05"] - vals[REF_NAME][MAIN]["detection_rate_fpr_0.05"],
                            "robust_tpr05_delta": vals[n][ROB]["detection_rate_fpr_0.05"] - vals[REF_NAME][ROB]["detection_rate_fpr_0.05"],
                            "ce_ratio_delta": vals[n][MAIN]["c4_cross_entropy_ratio"] - vals[REF_NAME][MAIN]["c4_cross_entropy_ratio"],
                            "ref_rank_unpaired": sum(
                                w * c * s * (vals[REF_NAME][e][m] - scores[REF_NAME][e][m]) / std_all[(e, m)]
                                for e, m, w, c, s in metrics if std_all[(e, m)] > 0)})
    paired = {}
    for n, rows in boot.items():
        if not rows:
            continue
        summ = {}
        for k in rows[0]:
            v = np.array([r[k] for r in rows])
            summ[k] = {"mean": float(v.mean()), "sd": float(v.std()), "ci95": [float(np.percentile(v, 2.5)),
                                                                              float(np.percentile(v, 97.5))]}
        paired[n] = summ

    # train vs test for the cohort members also run on train
    train_cmp = {}
    tr_path = RES / "scores_train_cohort.json"
    if tr_path.is_file():
        tr = load(tr_path)
        offtr = load(ROOT / "scoring" / "scores_train.json")
        rows = {n: tr["experiments"] for n in tr["_status"]}
        for n in tr["_status"]:
            train_cmp[n] = {e: {"train": tr["experiments"][e]["results"][n],
                                "test": scores.get(n, {}).get(e)} for e in (MAIN, ROB)}
        for tn, sn in (("gaussmark_200_tokens", REF_NAME), ("unwatermarked", "unwatermarked_baseline")):
            if tn in offtr["experiments"][MAIN]["results"]:
                train_cmp[sn] = {e: {"train": offtr["experiments"][e]["results"][tn],
                                     "test": scores.get(sn, {}).get(e)} for e in (MAIN, ROB)}

    out = {
        "rank_std_over_all_attempts": rank_all,
        "rank_std_over_successful_attempts": rank_ok,
        "std_all": {f"{e}/{m}": v for (e, m), v in std_all.items()},
        "std_ok": {f"{e}/{m}": v for (e, m), v in std_ok.items()},
        "status": {n: {"status": status[n]["status"], "reason": status[n].get("reason"),
                       "elapsed_s": status[n].get("elapsed_s"),
                       "measured": status[n].get("measured"),
                       "gates": status[n].get("gates")} for n in names},
        "scores": scores,
        "paired_bootstrap_vs_reference": paired,
        "determinism_reference_vs_replay": determinism,
        "train_vs_test": train_cmp,
    }
    with open(RES / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = ["| method | status | rank (all) | rank (ok) | TPR.05 | TPR.01 | AUC | med p | CE ratio | rob TPR.05 | rob AUC | paired Δrank 95% CI |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for n in sorted(names, key=lambda x: -rank_all[x]):
        s, r = scores[n][MAIN], scores[n][ROB]
        ci = paired.get(n, {}).get("rank_delta", {}).get("ci95")
        ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        meas = status[n].get("measured") or {}
        lines.append(
            f"| {n} | {status[n]['status']} | {rank_all[n]:+.3f} | {rank_ok[n]:+.3f} | "
            f"{s['detection_rate_fpr_0.05']:.4f} | {s['detection_rate_fpr_0.01']:.4f} | {s['auc']:.4f} | "
            f"{s['median_p_value']:.3g} | {s['c4_cross_entropy_ratio']:.5f} | {r['detection_rate_fpr_0.05']:.4f} | "
            f"{r['auc']:.4f} | {ci_s} |")
    with open(RES / "analysis.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for n in names:
        if status[n]["status"] != "ok":
            print(n, "->", status[n].get("reason"))
        if status[n].get("measured"):
            print(n, "measured:", json.dumps(status[n]["measured"]))


if __name__ == "__main__":
    main()
