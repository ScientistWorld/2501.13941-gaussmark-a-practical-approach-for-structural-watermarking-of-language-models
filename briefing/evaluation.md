# Evaluation

## Metrics

### Detection Metrics
- **Detection Rate at FPR=0.05**: Fraction of watermarked text samples detected when the false-positive rate is capped at 5%. This is the primary metric.
- **Detection Rate at FPR=0.01**: Same metric at a stricter 1% false-positive threshold.
- **Median p-value**: The median p-value across all samples. Lower values indicate stronger detection signals.
- **ROC AUC**: Area under the ROC curve, measuring overall discriminative ability.
- **p-value**: A statistical test p-value from the detection procedure. Values below 0.05 (or 0.01) indicate detection at that significance level.

### Quality Metrics
- **SuperGLUE Average**: Average accuracy across 8 language understanding tasks. Watermarked and unwatermarked models should have comparable scores.
- **GSM-8K Accuracy**: Accuracy on grade school math problems. Watermarked models should match unwatermarked.
- **AlpacaEval-2.0 Win Rate**: Preference win rate of watermarked vs. unwatermarked model. Should be near 50% (no preference).
- **Cross-entropy on C4**: Per-token cross-entropy on real text. Watermarked models should not differ significantly from unwatermarked.

### Robustness Metrics
- **Detection Rate after Corruption**: Detection rate after random insertions, deletions, or substitutions of tokens in the generated text.
- **AUC after Roundtrip Translation**: Detection AUC after translating text to another language and back.

## Baselines and Targets

### Paper's Reported Numbers (targets)
| Model | TPR@FPR=0.05 (200 tokens) | TPR@FPR=0.01 (200 tokens) | Median p-value (200 tokens) |
|-------|--------------------------|--------------------------|---------------------------|
| Mistral-7B | ~62% | ~40% | ~0.09 |
| Llama3.1-8B | ~95% | ~82% | ~0.008 |
| Phi3.5-Mini | ~65% | ~40% | ~0.08 |

### Quality Targets
Watermarked models should be within confidence intervals of unwatermarked models on SuperGLUE, GSM-8K, and AlpacaEval.

### Baseline (no watermark)
Unwatermarked text should have p-values uniformly distributed on [0, 1], giving ~5% detection at FPR=0.05 and ~1% at FPR=0.01.

## Evaluation Protocol

### Detection Evaluation
1. Take prompts from the C4 realnewslike dataset (first 1K samples, truncated at 200 tokens)
2. Generate text using the model (with and without watermark)
3. Run detection procedure on each generated sample
4. Compute detection rates at FPR=0.05 and FPR=0.01
5. Report median p-value across samples

### Quality Evaluation
1. Evaluate watermarked and unwatermarked models on SuperGLUE benchmark using lm-evaluation-harness
2. Evaluate on GSM-8K using chain-of-thought prompting
3. Run AlpacaEval-2.0 comparing watermarked vs unwatermarked

### Robustness Evaluation
1. Corrupt watermarked text (insertions, deletions, substitutions at various fractions)
2. Run detection on corrupted text
3. Report detection rate as function of corruption level
4. For semantic robustness, translate text through French and back, then detect

### Generation Settings
- Temperature: 1.0
- Top-p: 0.9 (or 1.0 for exact reproduction)
- Number of seeds: 3 (for variance estimation)
- Number of prompts: 100-1000
