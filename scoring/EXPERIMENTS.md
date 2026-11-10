# Experiment Overview

Both experiments are produced by one evaluator run per method on the same
model, prompts, keys and generations; each name matches an entry in
`scoring/scores_test.json` / `scores_train.json`.

---

## watermark_detection_mistral

**Goal:** establish whether a provider holding a secret key can tell its
marked Mistral-7B-Instruct-v0.3 text from text that is independent of that key
— at the nominal false-positive levels, and without making the model worse.

- 200-token generations for 400 hidden C4 prompts (200 public prompts on the
  train slice), 50 prompts per key.
- Benefit: share of marked texts detected at p < 0.05 and p < 0.01, AUC
  against null texts, median p-value of marked texts.
- Constraint: the marked model's cross-entropy on held-out human C4 text
  relative to the original model (hard budget 1.015, lower is better).
- Validity: texts from the original model, human continuations and texts
  marked under a different key must be flagged no more often than the nominal
  level; a significant excess fails the run.

**Rows:** `gaussmark_200_tokens` (the paper's method) and `unwatermarked`
(no watermark; detection at chance, quality ratio exactly 1).

## watermark_robustness_mistral

**Goal:** establish whether detection survives the kind of edits a user might
make — random token substitutions, deletions and insertions affecting 20% of
the generated text — while the edited null texts are still flagged only at the
nominal level.

- Same marked and unwatermarked texts as above, each edited by the evaluator.
- Benefit: detection rates at p < 0.05 / p < 0.01, AUC against edited null
  texts, median p-value.
- The paper's method keeps only part of its detection power here, which is the
  main room for improvement.
