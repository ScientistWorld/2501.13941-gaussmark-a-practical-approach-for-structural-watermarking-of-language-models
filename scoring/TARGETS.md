# Evaluation Targets

Every target below is an entry of `scoring/scores_test.json` (hidden slice) and
`scoring/scores_train.json` (public slice). Reference numbers are the paper's
method as reproduced here (GaussMark, Mistral-7B-Instruct-v0.3, Table 2
settings), hidden slice / public slice. Validity and quality are enforced as
gates: a run that fails any gate gets the worst value of every metric in both
experiments (see `briefing/evaluation.md`).

---

## watermark_detection_mistral (weight 0.65)

**What it tests:** detection of marked 200-token generations at the nominal
p-value thresholds, under a hard quality budget, with false positives held at
the nominal level on unwatermarked-model text, human text and wrong-key text.

### Benefit targets
| metric | direction | coef | paper | reference (test / train) | baseline (test) |
|---|---|---:|---:|---:|---:|
| `detection_rate_fpr_0.05` (primary) | higher | 1.0 | ~0.62 (Fig. 2a, base Mistral) | 0.820 / 0.875 | 0.053 |
| `detection_rate_fpr_0.01` | higher | 0.6 | ~0.40 (Fig. 7a) | 0.630 / 0.685 | 0.018 |
| `auc` (marked vs pooled nulls) | higher | 0.4 | ~0.85 (Fig. 7b) | 0.961 / 0.977 | 0.504 |
| `median_p_value` | lower | 0.2 | ~0.09 (Fig. 1a) | 0.00275 / 0.00259 | 0.479 |

### Constraint targets
| metric | rule | paper | reference (test / train) |
|---|---|---:|---:|
| `c4_cross_entropy_ratio` (coef 0.8, lower better) | gate: must be <= 1.015 | 1.0097 (Table 3) | 1.0108 / 1.0113 |
| `false_positive_rate_0.05` (coef 0, diagnostic) | gate: binomial test vs 0.05 per null family | 0.05 (Prop. 3.2) | 0.057 / 0.048 |
| `false_positive_rate_0.01` (coef 0, diagnostic) | gate: binomial test vs 0.01 per null family | 0.01 | 0.0125 / 0.0033 |

False-positive rates are not ranked: an exact test sits at the nominal level,
and a more conservative test is not better.

---

## watermark_robustness_mistral (weight 0.35)

**What it tests:** detection after 20% of the tokens of every text are randomly
substituted, deleted or inserted (paper Sec. 4.3). This is where the paper's
method falls short; it is the gym's headroom experiment.

| metric | direction | coef | paper | reference (test / train) | baseline (test) |
|---|---|---:|---:|---:|---:|
| `detection_rate_fpr_0.05` (primary) | higher | 1.0 | ~0.57 at 20% (Fig. 4, mean of 3 attacks, longer texts) | 0.288 / 0.310 | 0.060 |
| `detection_rate_fpr_0.01` | higher | 0.5 | — | 0.103 / 0.120 | 0.010 |
| `auc` (corrupted marked vs corrupted unwatermarked) | higher | 0.5 | — (Fig. 17) | 0.774 / 0.792 | 0.532 |
| `median_p_value` | lower | 0.2 | — (Fig. 16) | 0.152 / 0.119 | 0.449 |
| `false_positive_rate_0.05` (diagnostic, gated) | — | 0 | 0.05 | 0.0375 / 0.075 | 0.0375 |

---

## Notes

- Not scored (no hidden-test measurement): Llama3.1-8B, Phi3.5-Mini,
  SuperGLUE, GSM-8K, AlpacaEval, roundtrip translation, generation/detection
  latency. The paper's numbers are kept under `_paper_reported_unscored` in
  `scoring/reference.json`.
- Generation cost is identical for every method by construction (the evaluator
  decodes once per prompt with fixed sampling noise).
