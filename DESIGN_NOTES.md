# Design Notes — Train/Test Split for GaussMark

## Review Summary

The Stage 1 design was structurally sound — scripts were wired correctly, no cross-imports between train/test, no hardcoded method defaults, score files were absent. However, there was a **critical anti-shortcut signal gap**: `generation_time_ratio` and `c4_cross_entropy_*` are **global per-generation properties** (identical for every sample), so splitting samples 50/50 leaves those metrics byte-identical between train and test. A shortcut method that tuned generation speed or cross-entropy would get identical rewards on both slices — the train/test gap tells you nothing.

## Changes Made

### 1. Removed non-per-sample global properties from train/test scoring

`generation_time_ratio` and `c4_cross_entropy_*` are computed once per generation run, not per-sample. They are identical across any data split of the same method's output. For the anti-shortcut goal, they are noise: a method that improves them equally on train and test looks like a genuine improvement even if it gamed the evaluation protocol rather than the scientific quantity.

Both evaluators now **omit these metrics** from train/test scoring. The original `evaluate.sh` still computes them for the reference scores (`scores.json`).

### 2. Added `_slice`, `_num_samples_total`, `_num_samples_used` to output

Each scored method result now carries slice metadata (e.g., `_slice: "train"`) and sample counts (total vs. used). This makes it trivially auditable that train and test actually evaluated different sample subsets. It also surfaces whether a split left one slice too small (e.g., a 20-sample file → 10 train, 10 test, which may be noisy).

### 3. Auto-derive method name from watermarked file

`derive_method_name()` strips common prefixes (`watermarked_`, `wm_`) and suffixes (`_detection_full.json`) to produce clean method identifiers. The `--method-name` argument is now optional — if a watermarked file is provided, the name is derived from the filename. This makes the evaluator more self-sufficient when invoked from the shell scripts.

### 4. Fixed unwatermarked baseline slicing

Both evaluators now apply the same slice function to the unwatermarked baseline. Previously, the test evaluator sliced watermarked p-values but may have consumed the full unwatermarked detection file — this would cause the AUC computation (which compares sliced watermarked vs. sliced unwatermarked) to use mismatched subset sizes. Now both use `heldout_slice()` consistently.

## Split Style

**Per-sample data split (first-half train, second-half test).** Each sample contributes a single p-value; splitting by index gives ~50/50 sample coverage. The primary metric (TPR at FPR=0.05) is computed per-sample and aggregates across the slice, so a data split cleanly differentiates the metric.

**Alternative considered: metric triangulation** — computing a different set of metrics on the same samples. Rejected because the primary metric (detection_rate) is per-sample, so there's no single global metric that wouldn't be affected by a data split.

## Runtime Concerns for Stage 3

- **Small eval sets**: The older `watermarked_layer20_up_proj_sigma1e-*_detection_full.json` files have only 20 p-values → 10 train, 10 test per file. This is noisy for detection rates. When these files are enumerated, each produces a train score and a test score with wide confidence intervals. Consider whether to filter files below a minimum sample count threshold.
- **No generation_time_ratio or c4_cross_entropy_delta in train/test scores**: The reference scoring (`scores.json`) still includes these; the train/test split scoring (`scores_train.json`, `scores_test.json`) does not. This is intentional — see above. However, if the gym's final ranking formula weights these metrics, future methods need to know they won't be scored on them via the train/test split. This is documented here; if the ranking formula changes, this design decision should be revisited.
- **Unwatermarked baseline is fixed**: All methods compare against the same unwatermarked baseline file. If a future method's unwatermarked detection differs significantly (e.g., a different watermark removal attack), the comparison becomes less meaningful.
- **Script assumes detection files are in `checkpoints/`**: The method interface is `checkpoints/` with `_detection_full.json` files. If a future method writes its outputs elsewhere, `evaluate_train.sh` / `evaluate_test.sh` will need updating.

## Verified Failures Caught

1. **Global-metric-identical across train/test**: `generation_time_ratio` and `c4_cross_entropy_*` are per-generation, not per-sample. They provide zero anti-shortcut signal when the split is by sample index.
2. **Unwatermarked baseline slice inconsistency**: Test evaluator's AUC computation was comparing sliced watermarked p-values against unsliced unwatermarked p-values. Now both use the same slice function.