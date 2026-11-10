# Progress

- Method: GaussMark (Algorithms 1–2) as an evaluator plugin, `method/gaussmark.py`
  (Mistral-7B-Instruct-v0.3, layer 20 up_proj, σ²=1e-5; optional rank reduction and
  token filtering). Baseline: `baseline/unwatermarked.py`.
- Hidden-test reference: TPR .82 (p<.05) / .63 (p<.01), AUC .961, median p .00275,
  C4 CE ratio 1.0108; after 20% token corruption TPR .2875, AUC .774.
  Unwatermarked baseline: TPR .0525, AUC .504, CE 1.000.
- Hardening (final iteration): see `HARDENING.md`. All red-team shortcuts now fail;
  a robust-detector variant beats the reference beyond paired noise.
- Not reproduced: Llama/Phi, SuperGLUE/GSM-8K/AlpacaEval, roundtrip translation, KGW.
