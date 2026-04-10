# Progress

## What Works

- **GaussMark algorithm fully implemented** (method/gaussmark.py):
  - `GaussMark.Generate`: samples xi ~ N(0, sigma^2), perturbs one MLP layer (layer 20 up_proj), generates text
  - `GaussMark.Detect`: computes psi = <xi, grad> / (sigma * ||grad||), returns p-value using standard normal CDF
  - Statistical validity: psi ~ N(0,1) under H0; current unwatermarked baseline is 7/94 at p<0.05 and 1/94 at p<0.01, slightly high at 5% but plausible under small-sample binomial variation
- **Model**: Mistral-7B-Instruct-v0.3 at /home/user/models/mistralai/Mistral-7B-Instruct-v0.3/
- **C4 dataset**: /home/user/data/c4_realnewslike/examples.json (53906 samples)
- **Pipeline script**: method/run_pipeline.py
- **Detection pipeline works**: Correct per-sample watermark keys regenerated during detection
- **Reusable scoring works**: `scripts/evaluate.sh` reads the detection artifacts, writes `scoring/scores.json`, and validates import separation with `python validate.py --compare`.

## Results

### sigma=sqrt(1e-5) (Mistral-7B-Instruct, 100 prompts, exact prompt/token IDs) — core_claim
- **Watermarked detection rate at FPR=0.05**: 78.0% (71/91 usable samples)
- **Watermarked detection rate at FPR=0.01**: 60.4% (55/91 usable samples)
- **Median p-value**: 0.00320
- **ROC AUC**: 0.9477
- **C4 cross-entropy ratio/delta**: 1.0102 / 0.0206
- **Generation time ratio**: 1.295 in this simple HuggingFace pipeline, including per-sample key generation and weight perturbation/restoration overhead
- **Unwatermarked baseline**: 7.45% at p<0.05 (7/94) and 1.06% at p<0.01 (1/94); the p<0.05 result is slightly above nominal but plausible under small-sample binomial variation (P[X>=7 | n=94, p=0.05] ~= 0.19).

### sigma=0.001 (Instruct model, 100 prompts) — prior method_runs result
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
1. A prior run with sigma=1e-3 gives detectable results (~22% at FPR=5%), clearly above 3% baseline
2. A prior run with sigma=0.032 gives strong results (~85% detection), but this larger perturbation must be constrained by quality metrics
3. Table 2 reports variance sigma^2. The current pipeline uses standard deviation sigma=sqrt(1e-5) for the paper's Mistral setting.
4. C4 cross-entropy ratio/delta were added as lightweight quality constraints before claiming `core_claim`.
5. Current `scores.json` contains the corrected core-claim run with fresh detection, false-positive, and quality metrics.
6. Continued audit found that prompt text had been stored as a 500-character prefix; detection now uses stored exact prompt token IDs and stale cached generations without prompt IDs are not reused.
7. The corrected run supports the core claim: the exact GaussMark method is detectable while the C4 cross-entropy quality ratio remains essentially unchanged.
8. Added `method/run_robustness.py` to evaluate the paper's token-corruption robustness setting by applying random deletions, insertions, and substitutions to existing generated token IDs and re-running the same GaussMark detector.

## Deviations from Paper
- Using Mistral-7B-Instruct-v0.3 instead of the paper's base Mistral-7B
- The current submitted run uses the paper's Mistral variance interpreted correctly as sigma=sqrt(1e-5), but on the Instruct variant
- Using 100 prompts × 1 seed (paper uses 1000 × 3 seeds)
- Using C4 cross-entropy as a lightweight quality constraint instead of full SuperGLUE/GSM-8K benchmark runs
- No AlpacaEval-2.0 evaluation (requires GPT-4o API)
- Token-corruption robustness is being tested on a bounded 40-sample subset rather than the paper's full 1000-prompt x 3-seed robustness sweep.

## Completed Core-Claim Job

- Ran GaussMark on Mistral-7B-Instruct-v0.3 with layer 20 `up_proj`, sigma=sqrt(1e-5), 100 C4 realnewslike prompts, and 200 generated tokens.
- Stores exact prompt and generated token IDs and regenerates/uses per-sample watermark keys for detection.
- Generated a fresh unwatermarked baseline in the same job for the false-positive constraint.
- Wrote detection metrics plus unwatermarked FPR and C4 cross-entropy quality ratio/delta to `scoring/scores.json`.
- Added generation-time ratio as a lightweight latency constraint derived from the existing generation artifacts.

## Pending Robustness Job

- Prepared `scripts/run.sh` to run `python3 /home/user/method/run_robustness.py --fraction 0.25 --max-samples 40`.
- The job reuses the existing watermarked and unwatermarked generation artifacts, applies 25% delete/insert/substitute token corruptions, and writes `/home/user/checkpoints/robustness_token_corruption_mistral.json`.
- Current milestone remains `core_claim` until this GPU job finishes and the robustness result is reviewed.

## Environment Structure
- `method/`: GaussMark implementation (gaussmark.py)
- `eval/`: Evaluation code (evaluate.py)
- `scripts/evaluate.sh`: Standard evaluation entry point
- `scoring/reference.json`: Paper's reported numbers
- `scoring/scores.json`: Reproduced results
- `briefing/`: Problem definition, method description, evaluation protocol
