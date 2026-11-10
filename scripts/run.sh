#!/bin/bash
# Hardening verification job (1 GPU). Resumable: each stage leaves a marker.
#   1. build the public train slice and the hidden test slice (CPU)
#   2. install the reference (GaussMark) and the unwatermarked baseline
#   3. reference runs on the train slice  -> scoring/scores_train.json, scores.json
#   4. planted-score check + hidden test   -> scoring/scores_test.json
#   5. variants and shortcuts on the hidden test (redteam/hardening/results)
#   6. a few variants on the train slice (train/test parity)
#   7. analysis (cohort rank, paired bootstrap) and validation
set -euo pipefail
cd /home/user
H=redteam/hardening
R=$H/results
mkdir -p "$R"
echo "=== hardening job $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python3 eval/train/build_train_set.py
python3 eval/test/build_test_set.py

bash scripts/baseline.sh
bash scripts/method.sh

if [ ! -f "$R/.stage3_done" ]; then
    bash scripts/evaluate_train.sh 2>&1 | tee "$R/evaluate_train.log"
    cp scoring/scores_train.json scoring/scores.json
    touch "$R/.stage3_done"
fi

if [ ! -f "$R/.stage4_done" ]; then
    # A planted perfect score must not survive a run of the hidden test.
    cat > scoring/scores_test.json <<'JSON'
{"experiments": {"watermark_detection_mistral": {"results": {"gaussmark_200_tokens": {
 "detection_rate_fpr_0.05": 1.0, "detection_rate_fpr_0.01": 1.0, "auc": 1.0, "median_p_value": 0.0,
 "c4_cross_entropy_ratio": 1.0}}}}}
JSON
    bash scripts/evaluate_test.sh 2>&1 | tee "$R/evaluate_test.log"
    python3 - <<'PY' | tee "$R/planted_score_check.json"
import json
s = json.load(open("scoring/scores_test.json"))
r = s["experiments"]["watermark_detection_mistral"]["results"]["gaussmark_200_tokens"]
print(json.dumps({"planted_detection_rate_fpr_0.05": 1.0,
                  "after_hidden_test": r["detection_rate_fpr_0.05"],
                  "planted_overwritten": r["detection_rate_fpr_0.05"] != 1.0 or r["auc"] != 1.0,
                  "status": s["_status"]["gaussmark_200_tokens"]["status"]}, indent=2))
PY
    touch "$R/.stage4_done"
fi

python3 $H/make_plugins.py
python3 eval/test/evaluate.py --watermarkers-dir $H/plugins_test --output "$R/scores_test_cohort.json" \
    --resume-dir "$R/partial_test" --dump-dir "$R/dump_test" 2>&1 | tee -a "$R/cohort_test.log"
python3 eval/train/evaluate.py --watermarkers-dir $H/plugins_train --output "$R/scores_train_cohort.json" \
    --resume-dir "$R/partial_train" --dump-dir "$R/dump_train" 2>&1 | tee -a "$R/cohort_train.log"

python3 $H/analyze.py 2>&1 | tee "$R/analysis.log"
python3 validate.py --compare --post-design 2>&1 | tee "$R/validate.log" || true
echo "=== done $(date) ==="
