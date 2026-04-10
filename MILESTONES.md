# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-09] - none
- Previous agent implemented GaussMark algorithm in method/gaussmark.py
- Generated watermarked text with sigma=0.032, got 85% detection (job ee2ace3f-95b)
- But sigma=0.032 is much larger than paper's sigma=1e-5

### [2026-04-09] - method_runs
- Previous job confirmed: GaussMark pipeline works (detection 85% with sigma=0.032, AUC=0.917)
- Unwatermarked baseline: 3% FPR at threshold 0.05 (correct statistical validity)
- Previous detection with sigma=1e-5 failed due to key path mismatch

### [2026-04-10] - method_runs
- Created comprehensive pipeline script: method/run_pipeline.py
- Fixes: correct C4 data loading from examples.json, proper key path handling
- Generates 50 watermarked samples with sigma=1e-5 (paper's parameter)
- Generates 100 unwatermarked baseline samples
- Runs detection with saved keys
- GPU job submitted for full pipeline execution
