# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-10 05:35] - method_runs
- Existing GaussMark implementation runs the paper's core computation: Gaussian perturbation of one MLP weight matrix plus normalized gradient/key detection statistic.
- Fixed the submitted pipeline to use per-sample keys, interpret the paper's reported Mistral variance as sigma^2, and add a C4 cross-entropy quality constraint before attempting `core_claim`.
- Submitting a GPU job to regenerate detection and quality metrics before claiming the next milestone.

### [2026-04-10 05:46] - method_runs
- Audited the prior workspace and kept the milestone at `method_runs` because `scores.json` was stale/mixed and missing C4 quality constraint metrics.
- Fixed the generation path to preserve exact sampled token IDs instead of decode/re-tokenize before detection.
- Normalized sigma defaults to `sqrt(1e-5)` across method entry points and made the main pipeline regenerate a fresh unwatermarked baseline in the same run.
- Removed the non-paper high-sigma diagnostic from the reference target and prepared a new GPU job for a clean `core_claim` attempt.

## Stop Justification
