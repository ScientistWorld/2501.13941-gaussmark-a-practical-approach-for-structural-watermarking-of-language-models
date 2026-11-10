# Scientist Constraints

Improvement must come from the watermark (what is changed in the model per
key) and the detector — not from changing the setting or the measurement.

## Fixed by the evaluator (cannot be changed by a method)
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3`, bf16. A watermark may only
  add finite, key-dependent deltas to existing parameters; module structure,
  hooks, buffers and the base weights must be left untouched (checked after
  every call; any change fails the run).
- **Prompts, keys, null texts, corruptions:** C4 realnewslike prompts of
  `<s>` + 200 tokens; key seeds; unwatermarked-model, human and wrong-key null
  texts; 20% token edits — all owned by `eval/`. The hidden slice's inputs live
  only in `eval/test/`.
- **Decoding:** temperature 1, top-p 0.9, exactly 200 new tokens, special
  tokens disabled, one sample per prompt, evaluator-fixed sampling noise. A
  method cannot see prompts at generation time, resample, or select among
  candidates.
- **Thresholds and budgets:** nominal levels p < 0.05 and p < 0.01; quality
  budget C4 cross-entropy ratio <= 1.015; validity gate one-sided binomial
  p < 0.001 per null family; 3 h per method.
- **Test size:** 400 hidden documents, 8 keys of 50 documents.

## Interface
- `checkpoints/watermarkers/<name>/watermarker.py` with
  `make_watermark(model, key_seed) -> {param_name: delta}` and
  `detect(model, key_seed, prompts, texts) -> p-values in [0, 1]`
  (see `briefing/problem.md`). Every directory there is scored.
- `make_watermark` must be a deterministic function of `key_seed` and the
  original weights; `detect` gets the original weights.

## Prohibited
1. Reading or using anything under `eval/` (hidden data, seeds, evaluator
   state), inspecting the caller's stack/frames, or patching torch,
   transformers or the evaluator at runtime.
2. Returning p-values that are not genuine p-values for "this text was
   generated under this key" (e.g. using information other than the prompt,
   the text, the key and the original model).
3. Editing `eval/`, `scripts/evaluate*.sh`, thresholds or metric definitions.
4. Switching the model, prompt source or decoding settings.

## Compliance checklist
- [x] Same model and tokenizer; watermark expressed as parameter deltas
- [x] Detector returns one valid p-value per text
- [x] No access to `eval/` from the method
- [x] Quality within the 1.015 cross-entropy budget
