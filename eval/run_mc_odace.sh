#!/usr/bin/env bash
# ODACE multi-concept erasure (nudity + violence + Van Gogh): train (env-aware cost via CostMeter)
# -> eval 3 axes (nudity ASR / violence Q16 / Van Gogh style) + COCO utility -> aggregate cost
# -> regenerate gallery. Self-validating (5-step smoke gate). NO git (commit via Windows git).
# tmux: tmux new-session -d -s odacemc 'bash eval/run_mc_odace.sh'
# Watch: cat models/odace/MC_STATUS ; tail -f logs/mc_odace_train.log
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/odace/MC_STATUS
rm -f models/odace/MC_DONE
M=odace_mc
echo "=== MC-ODACE START $(date) ===" | tee "$ST"

echo "=== smoke (5 steps) START $(date) ===" | tee -a "$ST"
( cd models/odace && python train_odace.py --config configs/odace_mc.yaml \
    --num_steps 5 --output_dir outputs/_smoke_odace_mc ) 2>&1 | tee logs/mc_odace_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then
  echo "SMOKE FAILED (rc=$SRC) -> abort" | tee -a "$ST"; touch models/odace/MC_DONE; exit 1
fi

echo "=== train START $(date) ===" | tee -a "$ST"
( cd models/odace && python train_odace.py --config configs/odace_mc.yaml ) 2>&1 | tee logs/mc_odace_train.log
echo "=== train END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== nudity ASR START $(date) ===" | tee -a "$ST"
python eval/xeval.py --models $M 2>&1 | tee logs/mc_odace_nudity.log
echo "=== nudity END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== violence Q16 START $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models $M 2>&1 | tee logs/mc_odace_violence.log
echo "=== violence END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== Van Gogh style START $(date) ===" | tee -a "$ST"
python models/fcf/eval_style_vangogh.py --models $M 2>&1 | tee logs/mc_odace_style.log
echo "=== style END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== COCO utility START $(date) ===" | tee -a "$ST"
python eval/eval_coco.py --models $M 2>&1 | tee logs/mc_odace_coco.log || echo "(eval_coco skipped/failed)" | tee -a "$ST"
echo "=== COCO END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== cost aggregate + gallery START $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/mc_odace_cost.log
python compare/build_live_gallery.py 2>&1 | tee logs/mc_odace_gallery.log
echo "=== gallery END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== MC-ODACE DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/xeval.py eval/run_mc_odace.sh models/odace/train_odace.py models/odace/configs/odace_mc.yaml models/odace/data/prompts/mc_nvg_explicit.txt models/fcf/violence_q16.json models/fcf/style_vangogh.json models/fcf/train_cost.json compare/build_live_gallery.py ; git commit ; git push" | tee -a "$ST"
touch models/odace/MC_DONE
