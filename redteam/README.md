Audit runs are passive-output methods. Each directory has a `run.sh`, the exact
submitted artifacts, unchanged hidden-test logs, exit statuses, and test scores.
Run them only in a scheduled job. Every test uses `bash scripts/evaluate_test.sh`
with **no arguments**. No candidate modifies eval/, the evaluator's environment,
or any evaluator process. The model process exits before GPU outputs are scored.

`common.py` installs artifacts into the default checkpoints/ location and the
declared row into scoring/scores.json, then restores all original files. Original
reference artifacts are reused where the request allows it; the from-scratch
fabrication shortcut uses no reference data at all. Neither uses test labels.

The rank is cohort dependent. `summarize.py` uses absolute coefficients, declared
directions, and population standard deviations across the recorded audit cohort.
The original scores_test.json is the zero point. Raw scores are authoritative;
the displayed cohort rank is an illustrative fully specified comparison.

`original/` preserves pre-audit files solely for restoration and rank analysis;
shortcut methods do not consult its hidden-test scores or reference configuration.
`reference_source/` holds the explicitly permitted reference output reuse.
