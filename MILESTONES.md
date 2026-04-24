# Reproduction Milestones

**Current: core_claim**

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

### [2026-04-10 06:44] - method_runs
- Continued audit from a reset `method_runs` milestone.
- Found that generation artifacts stored exact generated token IDs but only a 500-character prompt prefix, so detection could condition on a different prompt than generation used.
- Fixed the method pipeline and optional baseline script to store exact prompt token IDs, require `exact_prompt_ids` before reusing cached generations, and detect from the stored prompt IDs.
- Kept the milestone at `method_runs` and prepared a corrected GPU job to regenerate the clean Mistral detection, false-positive, and C4 cross-entropy scores.

### [2026-04-10 07:08] - core_claim
- Corrected GPU job completed successfully on Mistral-7B-Instruct-v0.3 with the paper's GaussMark computation, layer 20 `up_proj`, and sigma=sqrt(1e-5).
- Achieved 78.0% detection at p<0.05 and 60.4% at p<0.01 on 91 usable watermarked 200-token samples, with ROC AUC 0.9477.
- C4 cross-entropy ratio was 1.0102 and delta was 0.0206, close to the paper's Table 3 Mistral quality constraint.
- Unwatermarked detection was 7/94 at p<0.05 and 1/94 at p<0.01. The 5% empirical rate is slightly high but within small-sample binomial variation under a 5% null (P[X>=7] ~= 0.19).

### [2026-04-10 07:36] - core_claim
- Added a token-corruption robustness runner that keeps the GaussMark detection computation intact and applies random deletion, insertion, and substitution corruptions to existing generated token IDs.
- Prepared a GPU test job to evaluate 25% corruption on up to 40 watermarked and unwatermarked Mistral samples per corruption type.
- Keeping the milestone at `core_claim` until the robustness job completes and its artifact is reviewed.

### [2026-04-10 08:05] - core_claim
- Final verification kept the milestone at `core_claim`.
- Tightened scoring so the unwatermarked false-positive rates are explicit constraint metrics on the marked-output result, not just a separate baseline row.
- Confirmed `python3 validate.py --compare` passes and `eval/` still does not import from `method/`.

### [2026-04-24 00:00] - core_claim
- Re-audited the packaged workspace against the paper and confirmed the committed method still matches GaussMark's Gaussian parameter perturbation plus normalized gradient/key detector.
- Found that `environment/setup.sh` was still empty and `action.yaml` incorrectly signaled `done`, which would break a fresh-clone rebuild despite the existing scored artifacts.
- Updated the environment to install the required Python stack and prepared a CPU packaging smoke test job that reruns the reusable evaluator and validator from committed scripts.

## Current Limitation
- Current completed milestone remains `core_claim`: the environment validates the central detection/quality/latency trade-off on the scaled Mistral-7B-Instruct experiment, but completed scored results do not yet cover the paper's full multi-model, robustness, and downstream task suite.
