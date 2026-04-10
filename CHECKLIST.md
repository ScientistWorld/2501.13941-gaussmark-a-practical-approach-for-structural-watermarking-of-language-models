# Reproduction Checklist

Check off items as you complete them. Order doesn't matter — work however makes sense for this paper.

## Briefing
- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json` — paper's reported numbers
- [x] Workspace validated via `python validate.py`
- [x] `scoring/scores.json` — reproduced numbers
- [x] `scoring/TARGETS.md` — primary, constraint, and ablation targets
- [x] `scoring/CONSTRAINTS.md` — what a scientist must hold fixed
- [x] `scoring/DIRECTION.md` — research direction scope

## Code
- [x] Evaluation code in `eval/`
- [x] Method implementation in `method/` (gaussmark.py)
- [x] Baseline implementation in `baseline/` (unwatermarked generation)

## Scripts
- [x] `scripts/evaluate.sh`
- [x] `scripts/reproduce.sh`
- [x] `scripts/download.sh` (idempotent, size comment at top)
- [x] `scripts/baseline.sh`
- [x] `scripts/method.sh`

## Environment
- [x] `environment/container.def` + `environment/setup.sh`

## Data
- [x] Dataset acquired: C4 realnewslike at /home/user/data/c4_realnewslike/
- [x] Pretrained model: Mistral-7B-Instruct-v0.3 at /home/user/models/mistralai/

## Milestones
- [x] `none` — Just started
- [x] `method_runs` — Algorithm executes end-to-end without errors
- [x] `core_claim` — Minimum experiment supports central claim
- [ ] `core_claim_plus` — Token-corruption robustness extension is submitted but not yet reviewed

## Notes
- Model deviation: Using Mistral-7B-Instruct-v0.3 (paper uses base model)
- Sigma note: the paper reports Mistral variance sigma^2=1e-5; the current job uses standard deviation sigma=sqrt(1e-5)
- Not reproduced: SuperGLUE/GSM-8K quality evaluation, Llama/Phi experiments, full-scale robustness
