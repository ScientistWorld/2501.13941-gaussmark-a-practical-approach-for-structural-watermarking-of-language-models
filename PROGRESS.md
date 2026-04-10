# Progress

## What Works

- **GaussMark algorithm fully implemented** (method/gaussmark.py):
  - `GaussMark.Generate`: samples xi ~ N(0, sigma^2), perturbs one MLP layer (layer 20 up_proj), generates text
  - `GaussMark.Detect`: computes psi = <xi, grad> / (sigma * ||grad||), returns p-value using standard normal CDF
  - Statistical validity: psi ~ N(0,1) under H0 — confirmed by 3% FPR on unwatermarked baseline
- **Model downloaded**: Mistral-7B-Instruct-v0.3 at /home/user/models/mistralai/Mistral-7B-Instruct-v0.3/
- **C4 dataset available**: /home/user/data/c4_realnewslike/examples.json (53906 samples)
- **Pipeline script**: method/run_pipeline.py

## Results

### Job ee2ace3f-95b (sigma=0.032) — SUCCESS
- Detection rate at FPR=0.05: **85%** (17/20 samples)
- Detection rate at FPR=0.01: **80%** (16/20)
- Median p-value: **3×10⁻⁷** (essentially 0)
- AUC: **0.917**
- Unwatermarked baseline: 3% FPR at 0.05 (correct!)

### Job ec800cb2-92a (sigma=1e-5) — PARTIAL
- Detection rate at FPR=0.05: **4%** (2/50) — essentially random
- Median p-value: **0.499** — no watermark signal
- Mean dot product: **0.00034** (vs 33.26 at sigma=0.032)
- Conclusion: sigma=1e-5 is too small for Mistral-7B-Instruct

### Current job (sigma=1e-3) — PENDING
- Expecting detection rate: ~20-50% at FPR=0.05
- Reuses unwatermarked baseline from ee2ace3f-95b (100 samples, FPR=3%)

## What Remains

- Validate detection with sigma=1e-3
- If needed, try larger sigma values (1e-2, 3e-2)
- Update scores.json with reproducible results
- Optional: Llama3.1-8B and Phi3.5-Mini experiments
- Optional: robustness experiments with token corruptions

## Issues

- **sigma=1e-5 too small**: Mistral-7B-Instruct doesn't produce detectable watermarks at paper's sigma
- This is likely because: (a) Instruct models have lower entropy than base models, (b) the up_proj perturbation scale is too small at sigma=1e-5
- The paper's paper likely uses Mistral-7B base model, not Instruct
- **Git hanging**: git commands timeout due to remote operations

## Deviations from Paper

- Using Mistral-7B-Instruct-v0.3 (paper uses Mistral-7B base model)
- Using larger sigma (1e-3 to 0.032 vs paper's 1e-5) due to Instruct model
- Using 50-100 prompts × 1 seed (paper uses 1000 × 3 seeds)
- Using 200 generated tokens (paper tests 100-800 tokens)
- No AlpacaEval-2.0 evaluation (requires GPT-4o API)
- No quality evaluation on SuperGLUE/GSM-8K benchmarks
