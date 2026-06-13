#!/usr/bin/env bash
# ODACE multi-concept erasure v2 (violence forget prompts 21->46, fix v1 violence 37.5):
# train (env-aware cost via CostMeter) -> eval 3 axes (nudity ASR / violence Q16 / Van Gogh
# style) + COCO utility -> aggregate cost -> regenerate gallery. Self-validating (smoke gate).
# NO git (commit via Windows git). tmux: tmux new-session -d -s odacemcv2 'bash eval/run_mc_odace_v2.sh'
# Watch: cat models/odace/MCV2_STATUS ; tail -f logs/mc_odace_v2_train.log
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/odace/MCV2_STATUS
rm -f models/odace/MCV2_DONE
M=odace_mc_v2
echo "=== MC-ODACE-v2 START $(date) ===" | tee "$ST"

echo "=== smoke (5 steps) START $(date) ===" | tee -a "$ST"
( cd models/odace && python train_odace.py --config configs/odace_mc_v2.yaml \
    --num_steps 5 --output_dir outputs/_smoke_odace_mc_v2 ) 2>&1 | tee logs/mc_odace_v2_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then
  echo "SMOKE FAILED (rc=$SRC) -> abort" | tee -a "$ST"; touch models/odace/MCV2_DONE; exit 1
fi

echo "=== train START $(date) ===" | tee -a "$ST"
( cd models/odace && python train_odace.py --config configs/odace_mc_v2.yaml ) 2>&1 | tee logs/mc_odace_v2_train.log
echo "=== train END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== nudity ASR START $(date) ===" | tee -a "$ST"
python eval/xeval.py --models $M 2>&1 | tee logs/mc_odace_v2_nudity.log
echo "=== nudity END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== violence Q16 START $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models $M 2>&1 | tee logs/mc_odace_v2_violence.log
echo "=== violence END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== Van Gogh style START $(date) ===" | tee -a "$ST"
python models/fcf/eval_style_vangogh.py --models $M 2>&1 | tee logs/mc_odace_v2_style.log
echo "=== style END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== COCO utility START $(date) ===" | tee -a "$ST"
python eval/eval_coco.py --models $M 2>&1 | tee logs/mc_odace_v2_coco.log || echo "(eval_coco skipped/failed)" | tee -a "$ST"
echo "=== COCO END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== cost aggregate + gallery START $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/mc_odace_v2_cost.log
python compare/build_live_gallery.py 2>&1 | tee logs/mc_odace_v2_gallery.log
echo "=== gallery END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== MC-ODACE-v2 DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/xeval.py eval/run_mc_odace_v2.sh models/odace/configs/odace_mc_v2.yaml models/odace/data/prompts/mc_nvg_explicit_v2.txt models/fcf/violence_q16.json models/fcf/style_vangogh.json models/fcf/train_cost.json ; git commit ; git push" | tee -a "$ST"
touch models/odace/MCV2_DONE
