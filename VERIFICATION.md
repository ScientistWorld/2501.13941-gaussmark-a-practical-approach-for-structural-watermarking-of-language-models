# Verification Report

## Initial Assessment
The workspace started at `core_claim`, and that milestone was justified. The method implementation matches the paper's core computation, the scored Mistral detection run is present in `scoring/scores.json`, and both `python3 validate.py` and `python3 validate.py --compare --post-design` pass.

The main risk was not algorithmic correctness but gym readiness: the standalone evaluator and some scoring docs still carried unnecessary reproduction-specific assumptions that would make future scientist runs less portable than they should be.

## Issues Found
- Minor: `scripts/evaluate.sh` and `eval/directory_evaluate.py` still defaulted to a baked-in method name and fixed artifact paths, which made the reusable evaluator more coupled to this reproduction than necessary.
- Minor: `scoring/CONSTRAINTS.md` pointed at an outdated prompt path outside the committed download layout.
- Cosmetic: `scoring/EXPERIMENTS.md` referred to the current scored row using the paper method name rather than describing it as the scored marked-output row.

## Fixes Applied
- Generalized `eval/directory_evaluate.py` to:
  - infer a marked method name when none is supplied,
  - discover generic marked and unmarked detection artifacts,
  - discover generic generation artifacts for the latency ratio,
  - keep the existing scored key when the caller provides it explicitly.
- Updated `scripts/evaluate.sh` so it no longer requires a hardcoded default method key and can optionally accept explicit marked/unmarked artifact paths.
- Corrected `scoring/CONSTRAINTS.md` to the portable prompt path under `data/downloads/`.
- Tightened `scoring/EXPERIMENTS.md` wording so it stays method-safe while still matching the existing result keys.
- Re-ran:
  - `bash scripts/evaluate.sh /home/user/checkpoints watermark_detection_mistral gaussmark_200_tokens`
  - `python3 validate.py`
  - `python3 validate.py --compare --post-design`
  - `curl -I -L --max-time 15` on both C4 shard URLs in `scripts/download.sh`

## Final Judgment
The workspace is ready to serve as a gym at the `core_claim` milestone. Improve-mode agents can use the existing method and scored baseline. From-scratch agents can use the problem definition, evaluation rules, constraints, and reusable evaluator without importing `method/` or being told the paper's algorithm in the method-agnostic files.

The benchmark is still intentionally narrow: the scored reproduction covers one scaled Mistral detection-quality slice rather than the full paper's multi-model and robustness suite. That limits breadth, but it does not invalidate the current core-claim gym.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: high
- Key limitation: the scored benchmark is a single scaled Mistral core-claim experiment, not the full paper's broader evaluation matrix.
