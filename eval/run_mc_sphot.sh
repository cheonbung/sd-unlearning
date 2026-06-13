#!/usr/bin/env bash
# Sph+OT multi-concept erasure (nudity+violence+Van Gogh): tests whether geodesic min-movement
# (N5 spherical) + OT noise (N6) lets TEXT-ENCODER editing SURVIVE multi-concept (LSSE-MC collapsed
# COCO CLIP to ~10). Builds union prompt files -> learns OT noise -> 2ep smoke gate -> 60ep train
# -> eval 3 axes + COCO -> cost -> gallery. NO parent edits, locked hyperparams. NO git.
# tmux: tmux new-session -d -s sphotmc 'bash eval/run_mc_sphot.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs models/novel/outputs/ot_noise

ST=models/novel/MC_STATUS
rm -f models/novel/MC_DONE
M=sph_ot_mc
echo "=== MC-SPHOT START $(date) ===" | tee "$ST"

echo "=== build union prompt files $(date) ===" | tee -a "$ST"
( cd models/novel && for t in explicit explicit_concepts implicit maintain; do
    cat data/prompts/nudity_$t.txt data/prompts/violence_$t.txt data/prompts/vangogh_$t.txt \
      | grep -vE '^#|^[[:space:]]*$' > data/prompts/mc_nvg_$t.txt
  done; wc -l data/prompts/mc_nvg_*.txt ) 2>&1 | tee -a "$ST"

echo "=== learn OT noise $(date) ===" | tee -a "$ST"
( cd models/novel && python scripts/learn_ot_noise.py \
    --explicit_file data/prompts/mc_nvg_explicit.txt \
    --output outputs/ot_noise/mc_nvg_learned.json ) 2>&1 | tee logs/mc_sphot_ot.log
echo "=== OT noise END (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"

echo "=== smoke (2 epochs) START $(date) ===" | tee -a "$ST"
( cd models/novel && python train.py --config configs/mc_nvg_sph.yaml --num_epochs 2 ) 2>&1 | tee logs/mc_sphot_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "SMOKE FAILED -> abort" | tee -a "$ST"; touch models/novel/MC_DONE; exit 1; fi

echo "=== train (60ep) START $(date) ===" | tee -a "$ST"
T0=$(date +%s)
( cd models/novel && python train.py --config configs/mc_nvg_sph.yaml ) 2>&1 | tee logs/mc_sphot_train.log
TRC=${PIPESTATUS[0]}
T1=$(date +%s)
echo "=== train END $(date) (rc=$TRC) ===" | tee -a "$ST"

echo "=== write cost $(date) ===" | tee -a "$ST"
WALL=$((T1-T0)) python - <<'PY' 2>&1 | tee -a "$ST"
import json, os, subprocess, datetime
wall = int(os.environ.get("WALL", "0"))
try:
    gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"]).decode().strip().splitlines()[0]
except Exception:
    gpu = "unknown"
d = {"model":"sph_ot_mc_nvg","training_free":False,"gpu":gpu,"gpu_count":1,
     "trainable_params_M":None,"steps":60,"wall_seconds":wall,
     "gpu_hours":round(wall/3600,3),"peak_vram_gb":None,"torch":None,
     "timestamp":datetime.date.today().isoformat(),"status":"ok"}
od="models/novel/outputs/sph_ot_mc_nvg"; os.makedirs(od,exist_ok=True)
json.dump(d, open(od+"/train_cost.json","w"), indent=2)
print("cost:", d)
PY

echo "=== nudity ASR START $(date) ===" | tee -a "$ST"
python eval/xeval.py --models $M 2>&1 | tee logs/mc_sphot_nudity.log
echo "=== violence Q16 START $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models $M 2>&1 | tee logs/mc_sphot_violence.log
echo "=== Van Gogh style START $(date) ===" | tee -a "$ST"
python models/fcf/eval_style_vangogh.py --models $M 2>&1 | tee logs/mc_sphot_style.log
echo "=== COCO utility START $(date) ===" | tee -a "$ST"
python eval/eval_coco.py --models $M 2>&1 | tee logs/mc_sphot_coco.log || echo "(eval_coco skipped/failed)" | tee -a "$ST"

echo "=== cost aggregate + gallery START $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/mc_sphot_cost.log
python compare/build_live_gallery.py 2>&1 | tee logs/mc_sphot_gallery.log

echo "=== MC-SPHOT DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/xeval.py eval/run_mc_sphot.sh models/novel/configs/mc_nvg_sph.yaml models/fcf/violence_q16.json models/fcf/style_vangogh.json models/fcf/train_cost.json ; git commit ; git push" | tee -a "$ST"
touch models/novel/MC_DONE
