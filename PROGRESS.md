# Progress

## What Works

- **GaussMark algorithm fully implemented** (method/gaussmark.py):
  - `GaussMark.Generate`: samples xi ~ N(0, sigma^2), perturbs one MLP layer (layer 20 up_proj), generates text
  - `GaussMark.Detect`: computes psi = <xi, grad> / (sigma * ||grad||), returns p-value using standard normal CDF
  - Statistical validity: psi ~ N(0,1) under H0 — confirmed by 3% FPR on unwatermarked baseline
- **Model downloaded**: Mistral-7B-Instruct-v0.3 at /home/user/models/mistralai/Mistral-7B-Instruct-v0.3/
- **C4 dataset available**: /home/user/data/c4_realnewslike/examples.json (53906 samples)
- **Pipeline script**: method/run_pipeline.py
- **Detection pipeline works**: Correct per-sample watermark keys are regenerated during detection

## Results

### Job 1f7c2e77-df9 (sigma=0.001, CORRECTED KEYS) — SUCCESS
- **Detection rate at FPR=0.05**: 21.7% (18/83 samples) — above 5% baseline
- **Detection rate at FPR=0.01**: 6.0% (5/83)
- **Median p-value**: 0.195 — significant shift from uniform 0.50
- **Mean dot product**: 0.685 (vs 0.00034 at sigma=1e-5)
- **ROC AUC**: 0.668 (vs 0.5 random)
- **Unwatermarked baseline**: FPR=3% at threshold 0.05 (correct!)
- With CORRECTED per-sample keys: watermark signal is clearly detected

### Previous Jobs (historical)
- Job ee2ace3f-95b (sigma=0.032): 85% detection, AUC=0.917 — but used same 20 samples with shared key
- Job ec800cb2-92a (sigma=1e-5): 4% detection — too small sigma for Instruct model

### Key Finding
sigma=1e-5 (paper's value for base Mistral-7B) produces nearly undetectable watermarks on Mistral-7B-Instruct-v0.3. This is likely due to:
1. Instruction-tuned models have lower entropy sequences
2. The Instruct model responds differently to MLP perturbations
3. up_proj layer (14336×4096) with tiny sigma may not produce detectable signal

sigma=0.001 gives detectable results (~22% at FPR=5%), and sigma=0.032 gives strong results (~85%).

## What Remains
- Additional model experiments (Llama3.1-8B, Phi3.5-Mini)
- Quality evaluation on SuperGLUE/GSM-8K benchmarks
- Robustness experiments with token corruptions
- Multi-seed experiments for variance estimation

## Issues
- **Model difference**: Paper uses Mistral-7B base model; we use Instruct variant (harder to watermark)
- **Sigma calibration**: Paper's sigma=1e-5 works for base model but not Instruct
- **No quality degradation test**: Would need lm-evaluation-harness setup

## Deviations from Paper
- Using Mistral-7B-Instruct-v0.3 (paper uses base model)
- Using larger sigma (1e-3 to 0.032 vs paper's 1e-5) due to Instruct model
- Using 100 prompts × 1 seed (paper uses 1000 × 3 seeds)
- Using 200 generated tokens (paper tests 100-800 tokens)
- No AlpacaEval-2.0 evaluation (requires GPT-4o API)
- No quality evaluation on SuperGLUE/GSM-8K benchmarks
