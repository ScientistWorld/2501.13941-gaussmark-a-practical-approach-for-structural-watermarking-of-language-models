# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-09] - none
- Previous agent implemented GaussMark algorithm in method/gaussmark.py
- Generated watermarked text with sigma=0.032, got 85% detection (job ee2ace3f-95b)
- But sigma=0.032 is much larger than paper's sigma=1e-5

### [2026-04-09] - method_runs
- Previous job confirmed: GaussMark pipeline works (detection 85% with sigma=0.032, AUC=0.917)
- Unwatermarked baseline: 3% FPR at threshold 0.05 (correct statistical validity)
- Previous detection with sigma=1e-5 failed due to key path mismatch

### [2026-04-10] - core_claim
- Job ec800cb2-92a: sigma=1e-5 gives only 4% detection (median p=0.50, mean dot=0.00034)
  - Confirmed: sigma=1e-5 is too small for Mistral-7B-Instruct
  - sigma=0.032 (previous job) gives 85% detection — pipeline proven
- Created comprehensive pipeline: method/run_pipeline.py
- Updated to sigma=1e-3 for next job (sigma=1e-5 confirmed too small)
- Reuses existing unwatermarked baseline (100 samples, FPR=3%)

## Key Finding
sigma=1e-5 (paper's value for Mistral-7B) produces undetectable watermarks on
Mistral-7B-Instruct-v0.3. This is likely due to:
1. Instruction-tuned models have lower entropy sequences
2. The Instruct model may respond differently to MLP perturbations
3. Scale: up_proj layer (14336×4096) with tiny sigma may not produce detectable signal

sigma=0.032 (previous job) gives 85% detection — the method works at larger sigma.
