# Paper's Method

## Key Contribution

GaussMark is a novel **structural watermarking scheme** for LLMs that embeds a watermark by adding Gaussian noise to the model's weights during generation. Unlike token-level watermarking schemes that modify the sampling process, GaussMark modifies the model itself.

The key insight is that small additive perturbations to LLM weights produce models of identical or even improved quality (as observed in model merging and fine-tuning literature). By exploiting this robustness, GaussMark adds just enough noise to create a detectable statistical signal without harming model quality.

## Approach

### GaussMark.Generate (Watermark Embedding)
1. Sample a **watermark key** `xi ~ N(0, sigma^2 * I)` from a multivariate Gaussian distribution
2. Perturb the model weights: `theta_wm = theta + xi` (add the key to the parameters)
3. Generate text using the perturbed model `p_theta_wm`
4. Return: generated text, the watermark key `xi`

Only ONE MLP layer's weights are perturbed. For Mistral-7B: layer 20, `up_proj` weight matrix (58M parameters). The layer is chosen empirically to balance detectability and quality.

### GaussMark.Detect (Watermark Detection)
1. Given: original model `p_theta`, prompt `x`, generated text `y`, watermark key `xi`, variance `sigma`
2. Compute the **test statistic**:
   `psi = <xi, grad_theta log p_theta(y|x)> / (sigma * ||grad_theta log p_theta(y|x)||)`
   where `<,>` is the dot product and `|| ||` is the L2 norm
3. Compute **p-value** = 1 - Phi(psi) where Phi is the standard normal CDF
4. If p-value < alpha (e.g., 0.05), reject the null hypothesis (claim: text is watermarked)

### Key Theoretical Results
- **Statistical Validity (Proposition 3.2)**: Under the null hypothesis H0 (text is NOT watermarked), `psi ~ N(0, 1)` exactly. This means p-values are uniformly distributed and the false-positive rate is exactly controlled.
- **Power Analysis (Proposition 3.4)**: Under a linear softmax model approximation, the test has non-trivial power that increases with model dimension and sequence length.

### Why It Works (Intuition)
- The watermark key `xi` is Gaussian and independent of the generated text under H0
- The gradient `grad_theta log p_theta(y|x)` is also independent of `xi` under H0
- Their dot product, normalized, follows a standard normal distribution
- Under H_A (text IS watermarked), the gradient is correlated with `xi`, making `psi` systematically large

### Practical Implementation
- Perturb only a single MLP layer (not the entire model) to minimize quality impact
- The gradient computation requires only ONE forward pass and ONE backward pass through the watermarked layer
- Generation: NO additional overhead (just weight perturbation, same as regular inference)
- Detection: <1 second for 1K tokens

### Rank-Reduced Variant (Section 4.4)
- Instead of perturbing all directions of a weight matrix, only perturb the **bottom k principal components**
- This allows MORE noise (stronger watermark) while preserving model quality
- Used in some experiments for better quality-watermark tradeoffs

## Reference implementation in this gym

- `method/gaussmark.py` is the evaluator plugin: `make_watermark` returns the
  Gaussian key xi (std sqrt(1e-5)) for `model.layers.20.mlp.up_proj.weight`,
  drawn from the evaluator's key seed; `detect` computes psi with the gradient
  of the summed log-likelihood of the text at the original weights and returns
  1 - Phi(psi). Options in `config.json`: `layer`, `weight`, `variance`,
  `drop_top_pcs` (rank-reduced variant; paper Table 7 uses layer 30 and 1024),
  `min_token_prob`.
- `python3 method/export_watermarker.py --name <name> --config '<json>'`
  installs it under `checkpoints/watermarkers/<name>/`.
- One key per group of 50 prompts (one marked deployment per key), as in the
  paper's per-seed experiments; the evaluator decodes, so generation cost is
  identical to the unmarked model by construction.
