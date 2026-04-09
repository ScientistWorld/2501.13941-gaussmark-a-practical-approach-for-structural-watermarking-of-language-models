# Research Direction

## Research Problem
The core question is whether we can embed a statistically detectable watermark into an LLM's outputs without modifying the generation process, degrading model quality, or requiring retraining. Existing watermarking schemes modify token-level sampling (introducing latency and quality tradeoffs). The question is: can we watermark at the model weight level instead?

## Core Contribution
The paper's key insight is that small Gaussian perturbations to LLM weights produce detectable signals in the generated text, without harming model quality. The contribution is:
1. A formal detection framework with exact N(0,1) null distribution (rigorous FPR control)
2. A zero-overhead watermarking scheme (only weights are perturbed, generation is unchanged)
3. An empirical validation that this works across multiple model families

The scientist must work within this contribution: improve the watermark signal, reduce quality impact, or increase robustness — but not replace the approach with token-level watermarking, a trained classifier, or a fundamentally different detection mechanism.

## Approach Scope
The scientist may work on:
- **Sigma tuning**: Finding the optimal sigma that maximizes detection while preserving quality
- **Layer selection**: Exploring which layer(s) produce the strongest watermark signal
- **Rank-reduced perturbation**: Perturbing only the top-k principal components of the weight matrix (mentioned in paper Section 4.4) to allow stronger noise with less quality impact
- **Detection robustness**: Improving detection after text corruptions (insertions, deletions, paraphrasing)
- **Multi-layer watermarking**: Watermarking multiple layers instead of just one
- **Key regeneration**: Improving key sampling strategy (currently uniform Gaussian per sample)

## Out of Bounds
- **Token-level watermarking**: Adding biased sampling or logit modifications during generation. This changes the generation process and has latency overhead — the whole point of GaussMark is zero overhead.
- **Trained classifiers**: Using a trained model (neural network, random forest, etc.) to classify watermarked vs unwatermarked text. The paper's contribution is a formal statistical test with exact null distribution — replacing it with a learned classifier loses the theoretical guarantees.
- **Switching model families**: Working on Llama or Phi instead of Mistral. The constraints specify Mistral for a reason.
- **Fine-tuning or training**: GaussMark requires NO training. Any approach that involves gradient-based updates to the model is out of scope.
- **Changing the null distribution**: Using a different test statistic or p-value computation that doesn't have the N(0,1) property.