# Research Direction

This file defines the research problem and approach scope for a scientist agent. Scientists must work within the paper's problem space and approach family — not switch paradigms to trivially beat the numbers.

---

## Research Problem

**Core question:** Can we embed a statistically detectable watermark into an LLM's outputs by modifying model weights, without degrading text quality, slowing generation, or requiring retraining?

The GaussMark paper answers YES by showing that small Gaussian perturbations to a single MLP layer create detectable signals in generated text. The scientist's job is to improve on this: make detection more reliable, preserve quality better, or increase robustness — within the same framework.

## Core Contribution (what to improve)

1. **Detection power:** Increase the fraction of watermarked text detected at fixed FPR
2. **Quality preservation:** Ensure watermarked model matches unwatermarked on benchmarks
3. **Robustness:** Maintain detection after text corruptions (insertions, deletions, paraphrasing)
4. **Sigma-layer tradeoffs:** Find the best layer(s) and noise level for the strongest detectable signal

## Approach Scope (what is allowed)

Scientists may work on:
- **Sigma tuning:** Finding the optimal sigma that maximizes detection while keeping FPR controlled
- **Layer selection:** Exploring which layer(s) produce the strongest watermark signal
- **Rank-reduced perturbation:** Perturbing only bottom-k principal components of the weight matrix (per paper Section 4.4) to allow stronger noise with less quality impact
- **Multi-layer watermarking:** Watermarking multiple layers instead of just one
- **Detection after corruptions:** Improving robustness of detection to text modifications

## Out of Bounds (what is NOT allowed)

These changes replace the method rather than improve it:
- **Token-level watermarking:** Adding biased sampling or logit modifications during generation. This changes the generation process and introduces latency overhead — the whole point of GaussMark is zero overhead.
- **Trained classifiers:** Using a neural network, random forest, or other learned model to classify watermarked vs unwatermarked text. The paper's contribution is a formal statistical test with exact N(0,1) null distribution — replacing it with a learned classifier loses the theoretical FPR guarantees.
- **Switching model families:** Working on Llama or Phi instead of Mistral. The constraints specify Mistral for a reason.
- **Fine-tuning or training:** GaussMark requires NO training. Any approach with gradient-based updates to the model is out of scope.
- **Changing the null distribution:** Using a different test statistic or p-value computation that doesn't have the N(0,1) property.

## Scientific Integrity

Improvements must be:
- **Statistically valid:** FPR must remain ≤ 5% (no inflating TPR by breaking the null hypothesis)
- **Finite-sample honest:** Report the unwatermarked false-positive rate and sample count; small deviations above 5% must be documented rather than hidden.
- **Methodologically principled:** Sigma and layer choices should be motivated by theory or ablations, not test-set snooping
- **Reproducible:** The same detection procedure must work for any prompt, not just the C4 prompts used in evaluation
