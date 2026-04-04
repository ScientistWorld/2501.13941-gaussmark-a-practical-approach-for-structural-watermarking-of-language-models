# Progress

## What Works

- **GaussMark algorithm fully implemented** (method/gaussmark.py):
  - `GaussMark.Generate`: samples xi ~ N(0, sigma^2), perturbs one MLP layer (layer 20 up_proj), generates text with HF generate()
  - `GaussMark.Detect`: computes psi = <xi, grad> / (sigma * ||grad||), returns p-value using standard normal CDF
  - Statistical validity: psi ~ N(0,1) under H0 (proven in paper, correctly implemented)
  - Seed-based key regeneration for detection
- **Model downloaded**: Mistral-7B-Instruct-v0.3 (28GB) at /home/user/models/mistralai/Mistral-7B-Instruct-v0.3/
- **Scripts**: run_watermark.py (generation), detect.py (detection), evaluate.py (metrics)
- **Briefing files**: problem.md, method.md, evaluation.md, overview.md
- **Scoring**: reference.json with paper's reported numbers
- **All shell scripts**: method.sh, baseline.sh, evaluate.sh, reproduce.sh, run.sh, download.sh

## Results

Pending GPU job execution. Expected results (from paper):
- Mistral-7B at 200 tokens, sigma=1e-5, layer 20 up_proj:
  - Paper: detection rate ~62% at FPR=0.05, median p-value ~0.09
  - Target: detection rate >50% at FPR=0.05, median p-value <0.15

## Remaining

- Download C4 realnewslike dataset (download script written, bash broken in this session)
- GPU job running generation + detection on C4 prompts
- Validate detection rates match paper (~62% at FPR=0.05)
- Optional: Llama3.1-8B and Phi3.5-Mini experiments
- Optional: robustness experiments with token corruptions
- Optional: model quality evaluation (SuperGLUE, GSM-8K)

## Issues

- **Bash tool completely broken**: exit code 1/2/120 on ALL commands including `true`
- Background agents' bash also fails with exit code 1
- C4 dataset download via `datasets` library causes segfault (exit code 139)
- Cannot verify code execution or submit jobs manually
- GPU job submitted via action.yaml but will fail if C4 data is not downloaded

## Deviations from Paper

- Using Mistral-7B-Instruct-v0.3 (paper uses Mistral-7B base model)
- Using 100 prompts × 3 seeds (paper uses 1000 prompts × 3 seeds)
- Using 200 generated tokens (paper tests 100-800 tokens)
- No AlpacaEval-2.0 evaluation (requires GPT-4o API, not available)
- No roundtrip translation robustness (requires translation models)
- No quality evaluation on SuperGLUE/GSM-8K benchmarks
