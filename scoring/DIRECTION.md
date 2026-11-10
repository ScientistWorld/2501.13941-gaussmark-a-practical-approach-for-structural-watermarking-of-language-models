# Research Direction

## Research problem

Can a provider embed a key-specific, statistically detectable signal into the
text an LLM generates by changing the model itself, with exact false-positive
control, no measurable loss in model quality, no extra generation cost, and
detection that survives edits of the text?

## What to improve
1. **Detection power** at the nominal levels p < 0.05 / p < 0.01, AUC and
   median p-value, on clean 200-token generations.
2. **Robustness**: detection after 20% random token substitutions, deletions
   and insertions — the paper's method loses most of its power here, so this
   is where most headroom is.
3. **Quality**: lower C4 cross-entropy ratio of the marked model, within the
   1.015 budget.

## In scope
- Which parameters are changed and how the key-dependent change is distributed
  (layers, matrices, subspaces, scales), within the quality budget.
- Detectors: better test statistics, weighting or selection of tokens, use of
  the prompt or not — as long as the p-values stay valid for text that is
  independent of the key (including text marked under another key).
- Quality-aware choices that buy detection at the same cross-entropy.

## Out of bounds
- Anything that changes the decoding, the model architecture, hooks, or the
  evaluator; reading evaluator data; fabricating or miscalibrated p-values
  (these fail the gates).
- Raising detection by exceeding the quality budget (fails the quality gate).
- Switching model, prompts, or generation settings.

## Scientific integrity
- Choose hyperparameters on the public slice (`scripts/evaluate_train.sh`),
  not on hidden-test scores.
- Report the false-positive diagnostics and gate margins together with any
  detection gain.
