# Progress

## What Works

- **GaussMark algorithm fully implemented** (method/gaussmark.py):
  - `GaussMark.Generate`: samples xi ~ N(0, sigma^2), perturbs one MLP layer (layer 20 up_proj), generates text with HF generate()
  - `GaussMark.Detect`: computes psi = <xi, grad> / (sigma * ||grad||), returns p-value using standard normal CDF
  - Statistical validity: psi ~ N(0,1) under H0 (proven in paper, correctly implemented)
  - Seed-based key regeneration for detection
- **Model downloaded**: Mistral-7B-Instruct-v0.3 at /home/user/models/mistralai/Mistral-7B-Instruct-v0.3/
- **C4 dataset available**: /home/user/data/c4_realnewslike/examples.json (53906 samples)
- **Pipeline script**: method/run_pipeline.py - comprehensive end-to-end pipeline

## Results

Previous job (ee2ace3f-95b, sigma=0.032):
- Detection rate at FPR=0.05: **85%** (17/20 samples)
- Detection rate at FPR=0.01: **80%** (16/20)
- Median p-value: **3e-7** (essentially 0)
- AUC: **0.917**
- Unwatermarked baseline: 3% FPR at 0.05 (correct!)

GPU job submitted for:
- 50 watermarked samples with sigma=1e-5 (paper's parameter)
- 100 unwatermarked baseline samples
- Proper detection with saved keys

## Remaining

- Await GPU job results
- If sigma=1e-5 detection fails, try larger sigma (1e-3)
- Optional: Llama3.1-8B and Phi3.5-Mini experiments
- Optional: robustness experiments with token corruptions
- Optional: model quality evaluation (SuperGLUE, GSM-8K)

## Issues

- **Git hanging**: git commands timeout due to remote operations. Files are in workspace but not committed.
- **Previous sigma=1e-5 detection failed**: key paths didn't match, keys were regenerated with wrong seed
- **C4 data format**: data.json only has 1 entry; examples.json has the full 53906 samples

## Deviations from Paper

- Using Mistral-7B-Instruct-v0.3 (paper uses Mistral-7B base model)
- Using 50 prompts × 1 seed (paper uses 1000 prompts × 3 seeds)
- Using 200 generated tokens (paper tests 100-800 tokens)
- No AlpacaEval-2.0 evaluation (requires GPT-4o API, not available)
- No roundtrip translation robustness (requires translation models)
- No quality evaluation on SuperGLUE/GSM-8K benchmarks
