# Scientist Constraints

These constraints define what a scientist must hold fixed when trying to improve on this paper's results. Improvement must come from genuine methodology — not from scaling compute, switching models, or circumventing the evaluation protocol.

---

## Model
- **Required:** Mistral-7B-Instruct-v0.3 from HuggingFace (`mistralai/Mistral-7B-Instruct-v0.3`)
- **Why fixed:** The paper's contribution is a structural watermark. Switching to a different model family (Llama, Phi, Gemma) or a different size variant changes the problem entirely.
- **Deviation in our reproduction:** We use the Instruct variant; the paper uses base Mistral-7B.

## Watermark Target Parameters
- **Required layer:** Layer 20 (0-indexed: `model.model.layers[20]`)
- **Required weight:** `up_proj` in the MLP at that layer
- **Required key shape:** Must match the `up_proj` weight matrix (hidden_size × intermediate_size = 14336 × 4096 for Mistral-7B)
- **Why fixed:** These parameters were found empirically by the paper to balance detectability and quality. Scientists may explore other layers/weights but must report them as separate experiments.

## Watermark Key
- **Required:** `xi ~ N(0, sigma^2 * I)`, sampled fresh for each generation
- **Prohibited:** Reusing the same key across multiple samples (that collapses to a single fixed-key experiment)
- **Shape constraint:** Key must be a full-rank isotropic Gaussian vector, not a structured or low-rank perturbation

## Sigma (Perturbation Strength)
- **Primary setting:** Use standard deviation `sigma=sqrt(1e-5)` for the Mistral experiment because the paper reports variance `sigma^2=1e-5`.
- **Allowed range for diagnostic experiments:** 1e-5 to 0.1
- **Required constraint:** Sigma must be chosen before seeing test-set detection results
- **Prohibited:** Fitting sigma to maximize detection rates on the test set post-hoc
- **Note:** If exploring different sigma values on the Instruct model, report them as separate diagnostic experiments and include the C4 cross-entropy quality constraint.

## Data
- **Prompts:** C4 realnewslike dataset, train split, first available examples, truncated to ≤200 tokens per prompt; the current scaled reproduction evaluates 100 prompts
- **Prompt source:** `/home/user/data/c4_realnewslike/examples.json`
- **No external datasets:** Scientists may not use other prompt sources for the primary detection experiment

## Evaluation Protocol
- **Required procedure:** Compute psi = <xi, grad_theta log p_theta(y|x)> / (sigma * ||grad_theta log p_theta(y|x)||), derive p-value from standard normal CDF
- **Required thresholds:** Report detection rates at FPR=0.05 and FPR=0.01
- **Prohibited:** Changing the detection procedure (different statistics, different p-value computation, different thresholds)
- **Prohibited:** Using a trained classifier instead of the gradient-based test

## Prohibited Actions
1. Switching base model (Llama, Phi, Gemma, etc.) for the primary experiment
2. Training or fine-tuning the base model — GaussMark is zero-training
3. Using a non-Gaussian watermark key (uniform, structured, low-rank)
4. Using a trained classifier for detection instead of the paper's statistical test
5. Modifying the null distribution or p-value computation
6. Fitting hyperparameters on the test set

## Compliance Checklist
- [x] Model is Mistral-7B-Instruct-v0.3
- [x] Watermarked layer is layer 20 (0-indexed)
- [x] Watermarked weight is `up_proj` in MLP at layer 20
- [x] Watermark key is Gaussian N(0, sigma^2*I), sampled per-generation
- [x] No fine-tuning or training performed
- [x] Detection uses gradient-based psi statistic
- [x] Evaluation uses FPR thresholds 0.05 and 0.01
