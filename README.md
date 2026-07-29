# Frozen normalization coefficients — 2501.13941

These are the **locked** per-paper variance-normalization coefficients used to score submissions on this benchmark task. Once locked they are frozen, so any future submission is scored against a stable, reproducible normalizer (the published score no longer drifts as new models are added to the cohort).

Each `coeffs_<mode>_<phase>.json` carries `cells` (per-cell `coefficient`, `source`, `higher_is_better`), `paper_baseline`, and the cohort it was computed from.

- `normalization/coeffs_improve_test.json` — mode=`improve` phase=`test` locked with **7** cohort models at `2026-06-29T19:52:46.077837+00:00`
- `normalization/coeffs_improve_train.json` — mode=`improve` phase=`train` locked with **7** cohort models at `2026-06-29T19:52:46.077837+00:00`
- `normalization/coeffs_innovate_test.json` — mode=`innovate` phase=`test` locked with **4** cohort models at `2026-07-29T00:33:47.643399+00:00`
- `normalization/coeffs_innovate_train.json` — mode=`innovate` phase=`train` locked with **4** cohort models at `2026-07-29T00:33:47.643399+00:00`

This is a standalone snapshot branch; it intentionally does not contain the rest of the paper's code or run history.
