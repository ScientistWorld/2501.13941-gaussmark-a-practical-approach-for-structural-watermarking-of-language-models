# HARDENING — GaussMark gym (final iteration)

Verification job `de3a73e8-6a2` (1×A100, exit 0, ~2.5 h) ran everything below
through the final, unchanged evaluators. Harness: `redteam/hardening/`
(`run_hardening.sh`, `make_plugins.py`, `analyze.py`); raw results, per-text
p-value dumps, logs and the analysis are in `redteam/hardening/results/`
(`analysis.md`, `analysis.json`).

## What was wrong (REDTEAM.md) and what changed

The old hidden test read a JSON list of p-values the method wrote, split it by
position, and scored only a 45/47-sample half of one Mistral experiment
(quality and timing omitted; Llama/Phi/SuperGLUE/GSM-8K never emitted). Any
p-values, any order, any sample count, an empty file or a missing null file
were accepted; the method controlled its own null comparison; brute-force
rejection sampling and a quality-for-detection trade both paid.

Fixes, all applied together:

1. **The evaluator runs the method on inputs only `eval/test/` holds.** A
   method is a plugin `checkpoints/watermarkers/<name>/watermarker.py` with
   `make_watermark(model, key_seed) -> {param: delta}` and
   `detect(model, key_seed, prompts, texts) -> p-values`. The evaluator owns
   the model, 400 hidden C4 validation documents, 8 hidden key seeds (50
   documents each), the decoding (temperature 1, top-p 0.9, exactly 200 tokens,
   one sample per prompt, per-prompt sampling noise fixed so every method sees
   the same random numbers), the null texts (original-model generations, human
   continuations, marked texts under the wrong key), the 20% token corruptions,
   and the C4 cross-entropy measurement. `detect` gets a shuffled, unlabeled
   mixture. Every scored number is recomputed from these raw outputs.
2. **Gates → FAILED run → worst value in every metric of both experiments:**
   missing plugin / exception / >3 h; malformed or non-finite deltas; wrong
   number of p-values or any value not a finite number in [0,1]; any change to
   the model's weights, buffers, modules, forward functions or hooks
   (checked and restored after every plugin call); C4 cross-entropy ratio
   > 1.015; a false-positive count significantly above nominal (one-sided
   binomial p < 1e-3) on any null family or the pooled nulls, at 0.05 and 0.01.
   The score file is rewritten from scratch (all methods "pending" = worst)
   before anything runs, so planted or stale scores never survive.
3. **Every method on disk is scored** (every directory under
   `checkpoints/watermarkers/`), `scripts/evaluate_test.sh` takes no arguments.
4. **Scored quality**: `c4_cross_entropy_ratio` (paper Table 3) is a ranked
   metric (coef 0.8) and a hard budget. Wall-clock metrics were dropped.
5. **Phantom experiments removed** from the ranking (Llama, Phi, SuperGLUE,
   GSM-8K: no hidden-test measurement); their paper numbers are kept verbatim
   under `_paper_reported_unscored` in `scoring/reference.json`.
6. **Bigger, fixed test**: 400 marked texts and 1,600 null texts (was 45/47);
   fixed per-prompt sampling noise makes the test deterministic.
7. **Headroom experiment** `watermark_robustness_mistral` (paper Sec. 4.3,
   Fig. 4): 20% random substitutions/deletions/insertions applied by the
   evaluator to marked and null texts.
8. Setup restored toward the paper: fixed-length 200-token generations (no
   EOS-truncated samples dropped), one key per deployment as in the paper's
   per-seed runs, exact token IDs, prompts = `<s>` + first 200 tokens. The
   method (Algorithm 1/2, layer 20 up_proj, σ²=1e-5) is unchanged.

Weights: detection 0.65, robustness 0.35 (new). The paper's headline claim is
detectability without quality loss; robustness is a secondary claim
("relatively robust") but is where the method falls short, so it gets a
substantial weight. CE ratio coef 0.8 so that, within the budget, quality
counts nearly as much as the primary detection rate. False-positive rates are
reported with coefficient 0 (an exact test sits at nominal; more conservative
is not better) and enforced by the gate.

## Reference score before / after

| | before (old hidden test) | after (new hidden test) |
|---|---|---|
| TPR p<.05 / p<.01 | .7778 / .5778 (45 texts) | **.8200 / .6300** (400 texts) |
| AUC / median p | .9319 / .00651 | **.9608 / .00275** |
| null FPR p<.05 / p<.01 | .1064 / 0 (47 texts) | .0567 / .0125 (1,200 texts) |
| C4 CE ratio | not scored | **1.0108** (budget 1.015; paper 1.0097) |
| robustness TPR .05 / .01 / AUC / median p | — | **.2875 / .1025 / .7744 / .152** |
| train slice (TPR .05, robust TPR .05, CE) | — | .875, .310, 1.0113 |

The reproduction shows the paper's claim in the paper's direction: the
unwatermarked baseline detects at chance (TPR .0525, AUC .504) with CE ratio
1.000, GaussMark detects 82% with a 1.08% cross-entropy increase (paper:
0.97%). Replaying the reference plugin in a second evaluator run gave
bit-identical detection metrics and CE (robust AUC .774425 vs .774431: one
p-value differs by GPU nondeterminism).

## Do better results score higher? (variants, before / after)

Rank = Σ weight × |coef| × improvement / std across the cohort (std over all 16
attempts; "ok-only" std in `analysis.md` gives the same order). Paired 95% CI:
bootstrap over the 400 hidden documents, every method on the same documents.

| variant | before (REDTEAM rank) | after: status | after: rank | paired 95% CI | key numbers |
|---|---:|---|---:|---|---|
| reference replay | 0 | ok | 0 | — | as above |
| **better: robust detector** (score only tokens with p_θ ≥ 1e-4; exact under H0) | not run | ok | **+0.512** | **[+0.303, +0.716]** | robust TPR .3675 (+.080, CI [+.038,+.123]), robust AUC .804; clean metrics and CE unchanged |
| paper's rank-reduced setting (Table 7: layer 30, 1024 PCs dropped) | not run | ok | −0.408 | [−0.783, −0.028] | CE 1.0075 (better), TPR .775, robust TPR .195 (worse) |
| weaker: σ² = 2.5e-6 | not run | ok | −1.490 | [−1.880, −1.158] | TPR .42, robust .125, CE 1.0026 |
| watermark removed (unwatermarked baseline) | −2.36 | ok | −3.080 | [−3.426, −2.777] | TPR .0525, AUC .504, CE 1.000 |
| σ² = 1.4e-5 (turn the knob) | — | FAILED (CE 1.0155) | −5.98 | | would have had TPR .9175 |
| robust detector + σ² = 1.4e-5 | — | FAILED (CE 1.0155) | −5.98 | | would have had TPR .9175, robust .465 |
| honest memory fix (red team) | −0.022 | not rerun | | | memory is not a metric (see "left") |

Before, seed noise (rank spread .235–.51) was as large as the effects and the
only "better" variant was a quality-degrading shortcut. After, the test is
deterministic, the reference's own unpaired bootstrap SD is .154, and a
genuinely better detector beats the reference with a paired interval that
excludes zero; a weaker variant and the baseline lose by far more than noise.

## Major shortcuts: fix and score before / after

All "after" rows ran through `eval/test/evaluate.py` (cohort run) or
`scripts/evaluate_test.sh` (planted score). FAILED = worst value everywhere
(rank −5.98 in this cohort; −8.52 with ok-only std), i.e. below the
unwatermarked baseline.

| shortcut | before | fix | after |
|---|---:|---|---|
| fabricated p-values (p≈0 for all texts) | +1.29 | evaluator-owned null texts + binomial validity gate | FAILED: 400/400 nulls rejected in every family |
| impossible p-values (−1e6) | +2.13 | p must be finite in [0,1] | FAILED at first `detect` call |
| reorder test / move null / copied outputs | +1.29 / +0.36 / −2.36 | method never sees labels, ordering or the null set; evaluator generates nulls | closed by construction (no input to manipulate) |
| single sample / wrong count / missing null | +1.29 / 0 | fixed test size; output count checked | `shortcut_wrong_count`: FAILED (99 values for 100 texts) |
| empty output / no plugin | 0 (tie) | missing plugin = failed run with worst values | `shortcut_empty_dir`: FAILED; `shortcut_crash`: FAILED |
| unlisted weaker method | 0 (tie) | every directory on disk is scored | closed |
| direct score overwrite | 0 (overwritten) | file rewritten from scratch before any work | planted TPR 1.0 overwritten by .82 (`planted_score_check.json`) |
| ignored quality / timing | 0 (tie) | CE ratio scored + budget; method cannot report quality | closed (method writes no metric) |
| quality trade-off (σ ×1.5, σ²=2.25e-5) | **+0.80** | CE budget 1.015 | FAILED: CE 1.0260 (would have had TPR .9625) |
| brute-force rejection sampling | **+0.89** | evaluator decodes exactly one sample per prompt with its own noise; `make_watermark` never sees a prompt | impossible through the interface |
| key-independent watermark (one fixed key, detector ignores key) | not tested | wrong-key null family | FAILED: 344/400 wrong-key texts flagged |
| anti-conservative p-values (p/2) | not tested | binomial gate per family and pooled | FAILED: 123/1200 pooled nulls at p<.05 (would have had TPR .88) |
| writing the watermark into the evaluator's model | not tested | weights/buffers/modules/hooks checked and restored after every call | FAILED: `model.layers.20.mlp.up_proj.weight` modified |
| summary flags, NaN, null values, all-positive | −2.61 / −0.21 / crash / −3.21 | the method reports only p-values; NaN/None rejected | closed |

## Shortcuts left, and why

- **In-process access to evaluator state.** The plugin runs in the evaluator's
  Python process: it could read `eval/test/data/test_set.json`, walk the call
  stack to the labels, or monkeypatch torch/transformers (e.g. the sampler's
  `argmax`). Closing this needs process sandboxing; the integrity check covers
  the model only. Prohibited in `scoring/CONSTRAINTS.md`; the separate grader
  must reject any plugin that touches `eval/`, `inspect`/frames, or patches
  libraries.
- **Operating-point tuning up to the quality budget.** The reference sits at
  CE 1.0108 under a 1.015 budget; σ² = 1.4e-5 already exceeds it (1.0155), so
  only ~σ² 1.35e-5 is reachable. This is the paper's own selection rule
  (maximize detection subject to quality), not a loophole; the gain is limited
  and CE is ranked with coefficient 0.8.
- **Unbounded detection compute.** A detector may spend much more compute per
  text (only a 3 h/method limit). More detection compute is a legitimate
  method choice, but it makes the hidden test heavier.
- **Resource improvements are not rewarded** (memory, latency). Wall-clock
  metrics are excluded by design; the red team's memory fix stays a tie in
  principle (headroom limitation, not a loophole).
- **Quality is a C4 cross-entropy proxy.** A delta that hurts specific
  capabilities (reasoning, instruction following) without moving C4
  cross-entropy is not caught; SuperGLUE/GSM-8K/AlpacaEval are not run.

## Headroom check

- No scored reference metric sits at its bound: TPR .82/.63 (bound 1), AUC
  .961 (1), median p .00275 (0), CE ratio 1.0108 (1.0, reached only by the
  no-watermark baseline), robustness TPR .2875/.1025, AUC .774, median .152.
- Added `watermark_robustness_mistral` (weight .35), where the paper's method
  keeps ~35% of its clean detection rate; a better detector already gains
  +.080 TPR there (CI excludes 0) without touching clean detection or quality.
- Brute force cannot be expressed (no generation control); the simple
  baseline scores far below the reference (−3.08); the gates stop every
  shortcut tried.

## Could not fix / stated plainly

- **The hidden test needs one CUDA GPU** (it executes the method: ~1.5 min
  shared setup + ~7 min per GaussMark-like method on an A100). Without a GPU
  `scripts/evaluate_test.sh` writes worst values for every method and exits.
- Model is Mistral-7B-**Instruct**-v0.3 (paper: base Mistral-7B); 200 tokens
  only; 8 keys × 50 prompts (paper: 1K prompts × 3 seeds); HuggingFace
  decoding (paper: vLLM); bf16 gradients. Llama/Phi, downstream benchmarks,
  roundtrip translation and the KGW comparison are not reproduced.
- The robustness paper value (≈.57) is read from Figure 4 at the figure's
  longer text length; the reproduced .2875 at 200 tokens is not on the same
  scale (validation passes, ratio 1.98).
- Gates are statistical: an exactly valid detector fails one of the 10 binomial
  checks with probability ≈1%. The reference's human-text null rejected 28/400
  at p<.05 (tail p .048) — within the gate, slightly above nominal.
- One corrupted-text p-value differed between two runs of the reference
  (GPU nondeterminism in a gradient); detection metrics were otherwise
  identical.
