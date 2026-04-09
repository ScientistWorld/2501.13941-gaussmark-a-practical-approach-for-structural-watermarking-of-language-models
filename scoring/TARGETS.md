# Evaluation Targets

## Primary: detection_rate_fpr_0.05
- **What**: Detection rate of watermarked text at 5% false-positive rate threshold.
- **Why**: Primary metric for watermark detectability. Higher is better.
- **How to run**: Generate 200 tokens using GaussMark-watermarked Mistral-7B, run detection on each sample, compute fraction with p-value < 0.05.

## Constraint: unwatermarked_fpr_0.05
- **What**: False-positive rate of detection on genuinely unwatermarked text at FPR=0.05.
- **Why**: Must stay ≤ 5% to avoid false accusations. A scientist who improves detection rate by inflating FPR has broken the method.
- **How to run**: Run detection on text generated without watermark. Fraction with p-value < 0.05 must be ≤ 5%.

## Primary: detection_rate_fpr_0.01
- **What**: Detection rate at a stricter 1% false-positive threshold.
- **Why**: Secondary sensitivity metric. Higher is better but must not sacrifice FPR control.

## Ablation: auc
- **What**: ROC AUC of watermarked vs unwatermarked p-values.
- **Why**: Overall discriminative power of the detector, not threshold-dependent.
- **How to run**: Compute AUC using watermarked p-values as positive class, unwatermarked as negative, with inverted scores (lower p = more likely watermarked).

## Ablation: median_p_value
- **What**: Median p-value across watermarked samples.
- **Why**: Indicates typical strength of detection signal. Lower is better.
- **How to run**: Take median of all p-values from watermarked detection.

## Constraint: model_quality_preserved
- **What**: Watermarking should not degrade model quality on downstream tasks.
- **Why**: If quality degrades significantly, the watermark is unusable in practice.
- **Metric**: SuperGLUE avg accuracy (higher is better, watermarked should match unwatermarked ± 1%).
- **Status**: Planned but not yet reproduced — requires lm-evaluation-harness setup.