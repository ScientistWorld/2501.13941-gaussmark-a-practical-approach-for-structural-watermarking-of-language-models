# Overview

- **Paper ID:** 2501.13941
- **Title:** GaussMark: A Practical Approach for Structural Watermarking of Language Models
- **Domain:** NLP / LLM Watermarking
- **TL;DR:** Embeds a detectable watermark into LLM outputs by adding small Gaussian noise to model weights during generation. Detection uses a gradient-based statistical test with formal guarantees and zero generation overhead.

## Short Summary

GaussMark watermarks LLM outputs by adding a small amount of Gaussian noise to the weights of a single MLP layer during generation. The watermark key (a random Gaussian vector) is sampled fresh for each generation. Detection works by computing a dot product between the watermark key and the gradient of the log-likelihood with respect to the original weights. Under the null hypothesis (unwatermarked text), this test statistic follows a standard normal distribution exactly, giving rigorous false-positive control. Detection is fast (<1s for 1K tokens) and generation has zero overhead since the only change is a weight perturbation before standard inference. Extensive experiments on Mistral-7B, Llama3.1-8B, and Phi3.5-Mini show high detection rates (~95% for Llama at 200 tokens, FPR=5%) with no significant quality degradation on SuperGLUE, GSM-8K, or AlpacaEval benchmarks.

## Key Results

- **Detection**: At 200 generated tokens, detection rates are ~62% (Mistral), ~95% (Llama), ~65% (Phi3.5-Mini) at FPR=5%. Rates increase with longer sequences.
- **Quality**: No significant degradation on SuperGLUE (avg ~0.70), GSM-8K, or AlpacaEval-2.0 compared to unwatermarked baselines.
- **Speed**: Zero generation latency overhead. Detection in <1s for 1K tokens.
- **Robustness**: Retains nontrivial detection after 10-50% token-level corruptions and roundtrip translation attacks.
