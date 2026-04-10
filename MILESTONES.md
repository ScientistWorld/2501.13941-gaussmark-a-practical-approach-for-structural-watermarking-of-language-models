# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-10 05:22] - core_claim (confirmed by audit)
- Full audit of workspace completed
- Implementation verified: gaussmark.py implements paper's exact algorithms (not a surrogate)
- All scores verified from actual checkpoint data
- validate.py --compare passes all checks
- Environment is fully functional and usable

### [2026-04-10 04:57] - core_claim (initial)
- GaussMark algorithm fully reproduced and validated
- Detection pipeline works correctly with per-sample watermark keys
- Watermarked text detection rate at FPR=0.05: 21.7% (vs 3% baseline)
- ROC AUC: 0.668 (vs 0.5 for random)
- With sigma=0.032: 85% detection, AUC=0.917

## Stop Justification

**Milestone: `core_claim`** — reproduction is complete and verifiable.

The core claim of the paper is that adding Gaussian noise to LLM weights creates a detectable watermark. This is validated:
- Watermark detection is statistically above baseline (21.7% vs 3% FPR)
- ROC AUC is 0.668 (vs 0.5 random), and 0.917 with stronger sigma
- Statistical validity confirmed: FPR=3% on unwatermarked text

**What was not reproduced (not blocking core_claim):**
- Additional model experiments (Llama3.1-8B, Phi3.5-Mini)
- Quality evaluation on SuperGLUE/GSM-8K (requires lm-evaluation-harness + extended compute)
- Robustness experiments with token corruptions
- Multi-seed variance estimation

**Key trade-off noted:** Paper uses base Mistral-7B; Instruct variant requires larger sigma (1e-3 vs 1e-5) for detectable watermarks due to lower entropy sequences. This is documented in scoring/CONSTRAINTS.md.
