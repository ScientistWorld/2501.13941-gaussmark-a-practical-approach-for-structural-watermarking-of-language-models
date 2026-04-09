# Scientist Constraints

## Model
The scientist must use Mistral-7B-Instruct-v0.3 (from HuggingFace `mistralai/Mistral-7B-Instruct-v0.3`).
The paper's contribution is a structural watermark — changing the model architecture or switching to a different model (Llama, Phi, etc.) would not demonstrate the same methodological contribution. The scientist may modify the watermarking procedure (which layer, how much noise, rank-reduced variant) but must keep the base model fixed.

## Watermark Layer
The scientist must watermark layer 20 of Mistral-7B (0-indexed, `model.model.layers[20]`). The paper's Table 2 specifies layer 20 for Mistral-7B. The weight to perturb is `up_proj` (MLP upward projection). These are constraints because the paper found these parameters empirically — different layers/weights may have different quality/watermark tradeoffs that are not captured in the evaluation.

## Watermark Key
The watermark key `xi ~ N(0, sigma^2 * I)` must be sampled fresh for each generation. The scientist cannot reuse a fixed key across samples (that would be a different scheme). The key shape must match the `up_proj` weight matrix at layer 20.

## Sigma (Perturbation Strength)
The scientist may vary sigma within a reasonable range (e.g., 0.01 to 0.1) to trade off detectability vs quality. However, the sigma used must come from a principled derivation (matching the paper's theory) rather than being fit to maximize detection rates on the test set.

## Data
The scientist must use C4 realnewslike dataset for prompt generation (first 2000 examples from train split, truncated to 200 tokens as prompts). No external datasets. The test set is the generated text itself — no held-out evaluation set is needed for the detection task.

## Evaluation Protocol
The scientist must use the same detection procedure: run forward+backward pass on the original (unwatermarked) model, compute psi = <xi, grad>/(sigma * ||grad||), report p-values and detection rates at FPR=0.05 and FPR=0.01. The scientist may not change the detection procedure to make it easier (e.g., different thresholds, softened statistics).

## Prohibited
- Switching to a different base model (Llama, Phi, Gemma, etc.) — this is architecture change, not method improvement.
- Modifying the detection procedure (e.g., using a different statistic, different p-value computation).
- Using a non-Gaussian watermark key (e.g., uniform, low-rank structured).
- Training or fine-tuning the model — GaussMark is a zero-training watermarking scheme.

## Compliance Checklist
- [ ] Model is Mistral-7B-Instruct-v0.3 (7B parameters), not a larger or smaller variant
- [ ] Watermarked layer is layer 20 (0-indexed)
- [ ] Watermarked weight is `up_proj` in the MLP at layer 20
- [ ] Watermark key is Gaussian N(0, sigma^2*I), sampled per-generation
- [ ] No fine-tuning or training performed on the base model
- [ ] Detection uses gradient-based psi statistic (not a surrogate classifier)
- [ ] Evaluation uses same FPR thresholds (0.05 and 0.01) as the paper