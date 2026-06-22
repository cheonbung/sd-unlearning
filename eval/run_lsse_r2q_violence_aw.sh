#!/usr/bin/env bash
# Violence R2q-ab + W6 adaptive weighting — utility-recovery retry. The non-AW violence run got
# Q16 ASR 16.7 but COCO-CLIP only 18.27 (violence L_cnp ~10^4x nudity overwhelmed the retain anchor).
# use_adaptive_weights down-weights the huge forget loss so the retain anchor regains influence.
# Compares against non-AW: lsse_r2q_violence (16.7 / 18.27).
# tmux: tmux new-session -d -s r2qviolaw 'bash eval/run_lsse_r2q_violence_aw.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

KEY=lsse_r2q_violence_aw
ST=models/fcf/R2Q_VIOL_AW_STATUS
rm -f models/fcf/R2Q_VIOL_AW_DONE
echo "=== R2Q-VIOLENCE-AW START $(date) ===" | tee "$ST"

echo "=== train $KEY $(date) ===" | tee -a "$ST"
( cd models/lsse && python train_lsse.py --config configs/violence_lsse_capcnp_r2q_aw.yaml \
    --output_dir outputs/lsse_r2q_violence_aw --no_diagnostics ) 2>&1 | tee logs/r2q_violence_aw_train.log
if [ ! -d models/lsse/outputs/lsse_r2q_violence_aw/final ]; then
  echo "TRAIN FAIL -> abort" | tee -a "$ST"; touch models/fcf/R2Q_VIOL_AW_DONE; exit 1; fi

echo "=== Q16 violence proxy (limit 20) $(date) ===" | tee -a "$ST"
rm -rf "eval/outputs/${KEY}_violence"
python models/fcf/eval_violence_q16.py --models "$KEY" --limit 20 2>&1 | tee logs/r2q_violence_aw_q16_proxy.log

echo "=== Q16 violence FULL $(date) ===" | tee -a "$ST"
rm -rf "eval/outputs/${KEY}_violence"
python models/fcf/eval_violence_q16.py --models "$KEY" 2>&1 | tee logs/r2q_violence_aw_q16_full.log

echo "=== COCO utility $(date) ===" | tee -a "$ST"
rm -rf "eval/outputs/${KEY}/coco" "eval/outputs/${KEY}/coco_metrics.json"
( cd eval && python eval_coco.py --models "$KEY" ) 2>&1 | tee logs/r2q_violence_aw_coco.log

echo "=== SUMMARY (compare vs non-AW 16.7 / 18.27) $(date) ===" | tee -a "$ST"
python - <<'PY' 2>&1 | tee -a "$ST"
import json
try:
    v=json.load(open("models/fcf/violence_q16.json"))
    m=(v.get("models") or v).get("lsse_r2q_violence_aw")
    print("violence_q16 AW ->", json.dumps(m) if m else "MISSING")
except Exception as e: print("violence read err:", e)
try:
    cm=json.load(open("eval/outputs/lsse_r2q_violence_aw/coco_metrics.json"))
    print("AW COCO CLIP=%s FID=%s LPIPS=%s  (non-AW was 18.27/135.78)"%(
        cm.get("coco_clip"),cm.get("coco_fid"),cm.get("coco_lpips")))
except Exception as e: print("coco read err:", e)
PY
echo "=== R2Q-VIOLENCE-AW DONE $(date) ===" | tee -a "$ST"
touch models/fcf/R2Q_VIOL_AW_DONE
