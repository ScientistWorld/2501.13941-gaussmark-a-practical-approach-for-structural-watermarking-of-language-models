Audit runs are passive-output methods. Each directory has a `run.sh`, the exact
submitted artifacts, unchanged hidden-test logs, exit statuses, and test scores.
Run them only in a scheduled job. Every test uses `bash scripts/evaluate_test.sh`
with **no arguments**. No candidate modifies eval/, the evaluator's environment,
or any evaluator process. The model process exits before GPU outputs are scored.

`common.py` installs artifacts into the default checkpoints/ location and the
declared row into scoring/scores.json, then restores all original files. Original
reference artifacts are reused where the request allows it. The fabrication
shortcut constructs constant p-values independent of reference measurements;
its common wrapper also loads the reference artifacts before replacing them.
Neither uses test labels.

The rank is cohort dependent. `summarize.py` uses absolute coefficients, declared
directions, and population standard deviations across the recorded audit cohort.
The original scores_test.json is the zero point. Raw scores are authoritative;
the displayed cohort rank is an illustrative fully specified comparison.

`original/` preserves pre-audit files solely for restoration and rank analysis;
shortcut methods do not consult its hidden-test scores or reference configuration.
`reference_source/` holds the explicitly permitted reference output reuse.

The first executed job completed the passive-output checks but its model runs
failed in the supplied tokenizer loader because protobuf is absent. The failure
logs are preserved in `runtime_failure_d0132c8b-98b/`. The resumed model runner
loads the existing fast tokenizer JSON and requires all 100 prompt token sequences
to match the reference before inference. It changes no installed packages.

The final findings are in [REDTEAM.md](../REDTEAM.md). The projected-noise run is
classified as `shortcut_quality_tradeoff` because its measured quality worsened;
its unedited raw model logs retain the earlier `honest_residual64` identifier.
