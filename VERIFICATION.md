# Verification Report

## Initial Assessment
The workspace started at `core_claim`. That milestone is justified: the code implements the paper's watermark generation and detection computation, the completed Mistral-7B-Instruct run produced measured scores, and `python3 validate.py --compare` passed at the start of verification.

The prior action requested an additional robustness GPU test, but the completed scored evidence remains the core Mistral detection/quality/latency experiment. I did not reset the milestone or rerun heavy compute.

## Issues Found
- Minor: The main scored experiment measured unwatermarked false-positive rates only as a separate baseline row. That made the primary marked-output row insufficiently explicit about the false-positive constraint future agents must preserve.
- Minor: The reusable directory evaluator had method-specific default artifact patterns and a method-specific docstring, even though it did not import method code.
- Minor: `scoring/TARGETS.md` used dotted names for the unwatermarked FPR targets that did not match the metric names in `scores.json`.
- Cosmetic: Root has legacy/system artifacts, including an ignored local `models/` directory and ignored root placeholder files. These do not affect the scored gym path; moving 28 GB of model weights during final verification would add risk without improving the benchmark.

## Fixes Applied
- Added explicit `unwatermarked_detection_rate_fpr_0.05` and `unwatermarked_detection_rate_fpr_0.01` constraint metrics to the marked-output result in `scoring/scores.json`.
- Updated `eval/directory_evaluate.py` so future methods can pass a method name and use generic marked-output artifact discovery, while preserving the existing `gaussmark_200_tokens` score key by default.
- Updated `scoring/reference.json` coefficients so primary detection remains the main benefit, thresholded null false-positive rates are explicit negative constraints, and median p-value/AUC are lower-weight diagnostic metrics.
- Updated `scoring/TARGETS.md` and `scoring/EXPERIMENTS.md` to match the revised scoring structure without adding method details.
- Re-ran `bash scripts/evaluate.sh /home/user/checkpoints watermark_detection_mistral` and `python3 validate.py --compare`; both pass.

## Final Judgment
The workspace is ready to serve as a research gym. Improve-mode agents can use the method implementation and measured core-claim scores. From-scratch agents can use the problem/evaluation/scoring guidance and the evaluator without importing from `method/`; the strict method-agnostic files do not describe the paper's algorithm.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: high
- Key limitation: the completed reproduction is scaled to one Mistral-7B-Instruct core experiment, not the full multi-model, robustness, and downstream-task suite.
