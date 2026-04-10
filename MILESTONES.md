# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-10 04:57] - core_claim
- GaussMark algorithm fully reproduced and validated
- Detection pipeline works correctly with per-sample watermark keys
- Watermarked text detection rate at FPR=0.05: 21.7% (vs 3% baseline, clearly above random)
- ROC AUC: 0.668 (vs 0.5 for random)
- Unwatermarked baseline FPR: 3% at threshold 0.05 (correct!)
- With sigma=0.032: 85% detection, AUC=0.917
- Environment is fully usable for scientist agents

### Key findings
- Paper's sigma=1e-5 works for base Mistral-7B but gives only 4% detection on Instruct variant
- sigma=1e-3 gives 21.7% detection on Instruct model (clearly above 3% baseline)
- sigma=0.032 gives 85% detection (strong watermark signal)
- Instruct models are harder to watermark than base models (lower entropy sequences)

### What works
- GaussMark.Generate: samples xi~N(0,sigma^2), perturbs up_proj at layer 20, generates text
- GaussMark.Detect: computes psi=<xi,grad>/(sigma*||grad||), returns p-value using N(0,1)
- Correct key regeneration during detection (per-sample keys)
- Statistical validity confirmed: FPR=3% on unwatermarked baseline (expected ~5%)

### What remains
- Additional model experiments (Llama3.1-8B, Phi3.5-Mini)
- Quality evaluation on SuperGLUE/GSM-8K
- Robustness experiments with token corruptions
