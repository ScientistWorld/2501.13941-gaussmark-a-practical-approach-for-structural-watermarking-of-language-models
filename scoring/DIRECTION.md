# Research Direction

This file defines the research problem and approach scope for a scientist agent. Scientists must work within the paper's problem space and approach family — not switch paradigms to trivially beat the numbers.

---

## Research Problem

**Core question:** Can we embed a statistically detectable watermark into an LLM's outputs without degrading text quality, materially slowing generation, or requiring a retrained task model?

The scientist's job is to improve the watermarking system under the same model, data, generation, and evaluation constraints: make detection more reliable, preserve quality better, reduce latency overhead, or increase robustness while maintaining false-positive control.

## Core Contribution (what to improve)

1. **Detection power:** Increase the fraction of watermarked text detected at fixed FPR
2. **Quality preservation:** Ensure watermarked model matches unwatermarked on benchmarks
3. **Robustness:** Maintain detection after text corruptions (insertions, deletions, paraphrasing)
4. **Calibration and robustness:** Improve score calibration and reliability across prompts without weakening the false-positive guarantee

## Approach Scope (what is allowed)

Scientists may work on:
- **Watermark strength and calibration:** Improve detectability while preserving the unwatermarked false-positive rate.
- **Quality-aware variants:** Reduce quality loss or cross-entropy increase at the same detection rate.
- **Efficiency improvements:** Reduce setup or generation overhead without changing the model or evaluation dataset.
- **Robustness checks:** Improve detection after reasonable text modifications, reported as separate experiments when not part of the primary score.
- **Ablations:** Compare design choices using the same prompts, model, thresholds, and quality metrics.

## Out of Bounds (what is NOT allowed)

These changes replace the method rather than improve it:
- **Changing the benchmark model or prompts:** Working on a different model family, size, tokenizer, or prompt source for the primary score.
- **Dataset-specific classification:** Training a classifier that recognizes artifacts of this small evaluation set rather than providing a general watermark detector.
- **Ignoring false-positive control:** Raising marked detection by making unmarked text trigger the detector more often.
- **Changing the evaluator contract:** Editing `scripts/evaluate.sh`, thresholds, or metric definitions to favor a method.
- **Using external paid services:** Calling paid APIs or closed models for generation or detection.

## Scientific Integrity

Improvements must be:
- **Statistically valid:** FPR must remain ≤ 5% (no inflating TPR by breaking the null hypothesis)
- **Finite-sample honest:** Report the unwatermarked false-positive rate and sample count; small deviations above 5% must be documented rather than hidden.
- **Methodologically principled:** Hyperparameters should be selected before held-out scoring or justified by separate ablations, not test-set snooping
- **Reproducible:** The same detection procedure must work for any prompt, not just the C4 prompts used in evaluation
