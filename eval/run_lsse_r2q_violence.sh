#!/usr/bin/env bash
# Transfer the R2q-ab nudity flagship recipe (read-out retain anchor A + causal per-layer B,
# contrastive_ortho dir) to VIOLENCE. Tests whether the read-out-space erasure + retain anchor
# generalizes beyond nudity. Eval: Q16 unsafe-image classifier (violence locality) + COCO utility.
# NOTE: violence L_cnp is ~10^4x larger than nudity (different read-out projection scale) — watch
# whether the retain anchor still holds utility; COCO CLIP will reveal it.
# tmux: tmux new-session -d -s r2qviol 'bash eval/run_lsse_r2q_violence.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

KEY=lsse_r2q_violence
ST=models/fcf/R2Q_VIOLENCE_STATUS
rm -f models/fcf/R2Q_VIOLENCE_DONE
echo "=== R2Q-VIOLENCE START $(date) ===" | tee "$ST"

echo "=== train $KEY $(date) ===" | tee -a "$ST"
( cd models/lsse && python train_lsse.py --config configs/violence_lsse_capcnp_r2q.yaml \
    --output_dir outputs/lsse_r2q_violence --no_diagnostics ) 2>&1 | tee logs/r2q_violence_train.log
if [ ! -d models/lsse/outputs/lsse_r2q_violence/final ]; then
  echo "TRAIN FAIL -> abort" | tee -a "$ST"; touch models/fcf/R2Q_VIOLENCE_DONE; exit 1; fi

echo "=== Q16 violence proxy (limit 20) $(date) ===" | tee -a "$ST"
rm -rf "eval/outputs/${KEY}_violence"
python models/fcf/eval_violence_q16.py --models "$KEY" --limit 20 2>&1 | tee logs/r2q_violence_q16_proxy.log

echo "=== Q16 violence FULL $(date) ===" | tee -a "$ST"
rm -rf "eval/outputs/${KEY}_violence"
python models/fcf/eval_violence_q16.py --models "$KEY" 2>&1 | tee logs/r2q_violence_q16_full.log

echo "=== COCO utility $(date) ===" | tee -a "$ST"
rm -rf "eval/outputs/${KEY}/coco" "eval/outputs/${KEY}/coco_metrics.json"
( cd eval && python eval_coco.py --models "$KEY" ) 2>&1 | tee logs/r2q_violence_coco.log

echo "=== SUMMARY $(date) ===" | tee -a "$ST"
python - <<'PY' 2>&1 | tee -a "$ST"
import json
try:
    v=json.load(open("models/fcf/violence_q16.json"))
    m=(v.get("models") or v).get("lsse_r2q_violence")
    print("violence_q16 lsse_r2q_violence ->", json.dumps(m) if m else "MISSING")
except Exception as e: print("violence read err:", e)
try:
    cm=json.load(open("eval/outputs/lsse_r2q_violence/coco_metrics.json"))
    print("COCO CLIP=%s FID=%s LPIPS=%s"%(cm.get("coco_clip"),cm.get("coco_fid"),cm.get("coco_lpips")))
except Exception as e: print("coco read err:", e)
PY
echo "=== R2Q-VIOLENCE DONE $(date) ===" | tee -a "$ST"
touch models/fcf/R2Q_VIOLENCE_DONE
