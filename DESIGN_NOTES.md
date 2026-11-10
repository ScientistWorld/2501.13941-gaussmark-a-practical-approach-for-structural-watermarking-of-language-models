# Design Notes — GaussMark gym (after hardening)

The earlier position-split design (method-written p-value lists split into
halves) was replaced in the final iteration; see `HARDENING.md` for why.

## Current design
- **Method = plugin** `checkpoints/watermarkers/<name>/watermarker.py`:
  `make_watermark(model, key_seed) -> {param: delta}`,
  `detect(model, key_seed, prompts, texts) -> p-values`.
- **Evaluator runs the method** on its own inputs: `eval/train/evaluate.py`
  (public: 200 C4 train-shard documents, 4 public keys) and
  `eval/test/evaluate.py` (hidden: 400 C4 validation documents, 8 secret keys).
  The two files are independent copies with slice-specific constants.
  Evaluation sets are built by `eval/*/build_*_set.py` (CPU, token IDs only).
- The evaluator decodes (fixed per-prompt sampling noise, 200 tokens, top-p
  0.9), generates the unwatermarked nulls, uses human continuations and
  wrong-key marked texts as further nulls, applies 20% token corruptions,
  measures C4 cross-entropy, and applies the gates (validity, quality,
  integrity, output format). Failed runs get worst values everywhere.
- Experiments: `watermark_detection_mistral` (0.65), `watermark_robustness_mistral` (0.35).
- Requires one CUDA GPU; ~7 min per method on an A100.
