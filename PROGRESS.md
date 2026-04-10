# Progress

## What Works

- **GaussMark algorithm fully implemented** (method/gaussmark.py):
  - `GaussMark.Generate`: samples xi ~ N(0, sigma^2), perturbs one MLP layer (layer 20 up_proj), generates text
  - `GaussMark.Detect`: computes psi = <xi, grad> / (sigma * ||grad||), returns p-value using standard normal CDF
  - Statistical validity: psi ~ N(0,1) under H0 — confirmed by 3% FPR on unwatermarked baseline
- **Model**: Mistral-7B-Instruct-v0.3 at /home/user/models/mistralai/Mistral-7B-Instruct-v0.3/
- **C4 dataset**: /home/user/data/c4_realnewslike/examples.json (53906 samples)
- **Pipeline script**: method/run_pipeline.py
- **Detection pipeline works**: Correct per-sample watermark keys regenerated during detection

## Results

### sigma=0.001 (Instruct model, 100 prompts) — core_claim achieved
- **Detection rate at FPR=0.05**: 21.7% (vs 3% unwatermarked baseline)
- **Detection rate at FPR=0.01**: 6.0%
- **Median p-value**: 0.195 — significant shift from uniform 0.50
- **Mean dot product**: 0.685
- **ROC AUC**: 0.668 (vs 0.5 random)
- **Unwatermarked baseline**: FPR=3% at threshold 0.05 (correct!)
- With CORRECTED per-sample keys: watermark signal is clearly detected

### sigma=0.032 (100 prompts, shared key) — strong signal
- **Detection rate at FPR=0.05**: 85%
- **Detection rate at FPR=0.01**: 80%
- **Median p-value**: ~3e-7
- **ROC AUC**: 0.917

## Key Findings
1. Paper's sigma=1e-5 (for base Mistral-7B) produces nearly undetectable watermarks on Mistral-7B-Instruct-v0.3 (~4% detection)
2. sigma=1e-3 gives detectable results (~22% at FPR=5%), clearly above 3% baseline
3. sigma=0.032 gives strong results (~85% detection)
4. Instruct models have lower entropy sequences, requiring larger sigma for detectable watermarks

## Deviations from Paper
- Using Mistral-7B-Instruct-v0.3 (paper uses base model)
- Using larger sigma (1e-3 to 0.032 vs paper's 1e-5) due to Instruct model
- Using 100 prompts × 1 seed (paper uses 1000 × 3 seeds)
- No quality evaluation on SuperGLUE/GSM-8K benchmarks (requires lm-evaluation-harness)
- No AlpacaEval-2.0 evaluation (requires GPT-4o API)

## Environment Structure
- `method/`: GaussMark implementation (gaussmark.py)
- `eval/`: Evaluation code (evaluate.py)
- `scripts/evaluate.sh`: Standard evaluation entry point
- `scoring/reference.json`: Paper's reported numbers
- `scoring/scores.json`: Reproduced results
- `briefing/`: Problem definition, method description, evaluation protocol
