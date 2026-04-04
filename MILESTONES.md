# Reproduction Milestones

**Current: method_runs**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-04-04] - none
- Read paper thoroughly: GaussMark (2501.13941) - structural watermarking of LLMs via Gaussian weight perturbation
- Core algorithm: add xi ~ N(0, sigma^2 I) to one MLP layer, detect via gradient test statistic

### [2026-04-04] - method_runs (current)
- Implemented GaussMark.Generate and GaussMark.Detect in method/gaussmark.py
- Model downloaded: Mistral-7B-Instruct-v0.3 (28GB) at /home/user/models/mistralai/Mistral-7B-Instruct-v0.3/
- All scripts written: run_watermark.py, detect.py, evaluate.py
- Briefing files: problem.md, method.md, evaluation.md, overview.md
- Scoring reference: scoring/reference.json with paper's numbers
- GPU job submitted for end-to-end validation
- Fixed seed tracking bug and filename format bug

## Stop Justification

<!-- Do not edit this unless you decide to stop -->
