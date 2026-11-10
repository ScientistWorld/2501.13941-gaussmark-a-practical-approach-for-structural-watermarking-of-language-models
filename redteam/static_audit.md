Audit state before measurements (2026-11-10)

The original iteration commit is recorded in original/base_commit.txt. The first
prepared job was cancelled by the scheduler before executing run.sh; its logs
contain no method results. This continuation resubmits with per-case persistence.

Source findings to confirm with the recorded runs:
- eval/test/evaluate_slice.py accepts passive JSON p_values and splits the list
  by position, using the second floor(n/2) values. The visible train evaluator
  uses the first ceil(n/2), so selection/reordering needs no hidden answers.
- Neither test generation nor detection is recomputed. No text, prompt IDs,
  key, fixed sample count, finite range, provenance, or minimum n is checked.
- Empty lists produce an empty results object with exit code 0. Missing null
  omits FPR, margin and AUC instead of assigning worst scores.
- scripts/evaluate_test.sh enumerates rows only from writable scores.json,
  excludes unwatermarked and processes only watermark_detection_mistral. It
  does not enumerate every candidate artifact on disk. Resolving a row uses
  its writable full-array metrics or newest file, not a stable artifact ID.
- The hidden test explicitly omits c4_cross_entropy_ratio/delta and
  generation_time_ratio. The score config's other four experiments never run.
- Self-reported detection summaries and flags are recomputed from p_values;
  directly planting scores_test.json should be overwritten on a successful
  test. Failure cases may retain a stale score file: do not count them as wins.
- Reference hidden Mistral TPRs are .7777778/.5777778, FPRs .106383/0,
  AUC .931915, median .00651384, mean .0576105, margin .671395.
  FPR .01 is already at its lower bound; all other emitted reference metrics
  have room to improve. Configured Llama400 TPR .05=1 is another bound, but
  that experiment does not appear in the hidden scores at all.

Paper / reproduction comparison (paper is used ONLY for this comparison;
shortcut code never reads it): actual Algorithm 1 Gaussian parameter perturbation
and Algorithm 2 normalized unperturbed likelihood-gradient test are implemented.
Mistral layer20/up_proj and variance1e-5 match Table2. Gym uses Instruct-v0.3
instead of paper base Mistral; 100 prompts / 1 seed instead of 1K / 3 seeds;
HuggingFace autoregressive generation instead of vLLM; top_p=.9; bf16 weights
and gradients; samples shorter than 50 generated tokens are discarded; generated
lengths may stop before 200 tokens. Quality is 25 prompt-prefix C4 CE checks
using one key rather than the paper's 1K-text CE plus SuperGLUE/GSM8K/AlpacaEval.
No scored corruption or paraphrase experiment, no Llama or Phi runs. Prompt
prefix decoding/re-tokenization and BOS handling are internally consistent
between generation and detection, but prompt prefixes, not generated response
quality, are used for the CE proxy. Keys are fresh per prompt with predictable
seed42+i and stored in output metadata, so this does not test secret-key use.
The test truncates successful detections into halves, rather than withholding
new prompts for a fresh detector execution. Apparent nominal TPR@FPR is actually
p<alpha with measured null .106 at alpha .05; it is not empirical FPR=.05.

All numeric rank deltas must state their cohort because the user specifies
per-metric normalization across attempts. Absolute coefficient values and
higher_is_better, with weight .5 on Mistral, are implemented in summarize.py.
A nonzero evaluator exit status must not be credited with stale reference scores.
