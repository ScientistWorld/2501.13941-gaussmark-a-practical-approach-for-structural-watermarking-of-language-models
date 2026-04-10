# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-10 05:35] - method_runs
- Existing GaussMark implementation runs the paper's core computation: Gaussian perturbation of one MLP weight matrix plus normalized gradient/key detection statistic.
- Fixed the submitted pipeline to use per-sample keys, interpret the paper's reported Mistral variance as sigma^2, and add a C4 cross-entropy quality constraint before attempting `core_claim`.
- Submitting a GPU job to regenerate detection and quality metrics before claiming the next milestone.

## Stop Justification
