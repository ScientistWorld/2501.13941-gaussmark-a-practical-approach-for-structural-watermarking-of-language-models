# Scientist Constraints

These constraints define what a scientist must hold fixed when trying to improve on this paper's results. Improvement must come from genuine methodology — not from scaling compute, switching models, or circumventing the evaluation protocol.

---

## Model
- **Required:** Mistral-7B-Instruct-v0.3 from HuggingFace (`mistralai/Mistral-7B-Instruct-v0.3`)
- **Why fixed:** Switching to a different model family, size, tokenizer, or instruction-tuning variant changes the difficulty and makes scores non-comparable.
- **Deviation in our reproduction:** We use the Instruct variant; the paper uses base Mistral-7B.

## Data
- **Prompts:** C4 realnewslike dataset, train split, first available examples, truncated to ≤200 tokens per prompt; the current scaled reproduction evaluates 100 prompts
- **Prompt source:** `/home/user/data/c4_realnewslike/examples.json`
- **No external datasets:** Scientists may not use other prompt sources for the primary detection experiment

## Generation
- **Required length:** 200 new tokens per prompt for the primary experiment.
- **Required sampling:** temperature=1.0 and top-p=0.9, unless a separately named diagnostic experiment documents a different setting.
- **No paid APIs:** All text must be generated locally from the fixed open model.

## Output Contract
- **Required artifacts:** Each method must produce marked-output results and an unmarked comparison in the format accepted by `scripts/evaluate.sh`.
- **Required statistics:** The evaluator must receive per-sample p-values or equivalent detector scores that can be converted into thresholded false-positive-rate metrics.
- **Required timing:** Generation timing should be measured in the same run for marked and unmarked outputs when reporting `generation_time_ratio`.

## Evaluation Protocol
- **Required thresholds:** Report detection rates at FPR=0.05 and FPR=0.01
- **Required null check:** Report the unwatermarked false-positive rates; improvements that inflate the unmarked detection rate are invalid.
- **Required quality check:** Report C4 cross-entropy ratio/delta or a directly comparable held-out language-model quality metric.
- **Prohibited:** Changing `scripts/evaluate.sh` or `eval/` to favor a specific method.
- **Prohibited:** Choosing hyperparameters after inspecting the held-out detection labels, then reporting that run as the primary result.

## Prohibited Actions
1. Switching base model (Llama, Phi, Gemma, etc.) for the primary experiment
2. Training or fine-tuning the base model for the primary comparison unless reported as a separate non-comparable experiment
3. Using external labels, hidden dataset metadata, or memorized prompts to classify outputs
4. Reporting only marked-output detection without the unmarked false-positive constraint
5. Modifying the evaluator, thresholds, or null calibration post-hoc to improve the score
6. Fitting hyperparameters on the test set

## Compliance Checklist
- [x] Model is Mistral-7B-Instruct-v0.3
- [x] No fine-tuning or training performed
- [x] Evaluation uses FPR thresholds 0.05 and 0.01
- [x] Unwatermarked false-positive rates are reported
- [x] C4 cross-entropy quality metrics are reported
- [x] Generation-time ratio is reported
