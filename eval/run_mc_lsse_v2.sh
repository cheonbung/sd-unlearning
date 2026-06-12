#!/usr/bin/env bash
# LSSE multi-concept erasure v2 (nudity-weighted CNP): queues behind ODACE-MC, then train
# (env-aware cost) -> eval 3 axes + COCO -> aggregate cost -> regenerate gallery. Self-validating.
# tmux: tmux new-session -d -s lssev2 'bash eval/run_mc_lsse_v2.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/lsse/MCV2_STATUS
rm -f models/lsse/MCV2_DONE
M=lsse_mc_nvg_v2
echo "=== MC-LSSE-v2 START $(date) ===" | tee "$ST"

# queue behind ODACE-MC (shares the single GPU)
if [ ! -f models/odace/MC_DONE ] && tmux has-session -t odacemc 2>/dev/null; then
  echo "=== waiting for ODACE-MC (models/odace/MC_DONE) $(date) ===" | tee -a "$ST"
  while [ ! -f models/odace/MC_DONE ] && tmux has-session -t odacemc 2>/dev/null; do sleep 120; done
  echo "=== ODACE-MC cleared $(date) ===" | tee -a "$ST"
fi

echo "=== smoke (1 epoch) START $(date) ===" | tee -a "$ST"
( cd models/lsse && python train_lsse.py --config configs/multiconcept_nvg_v2.yaml \
    --num_epochs 1 --output_dir outputs/_smoke_mc_v2 ) 2>&1 | tee logs/mc_lsse_v2_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "SMOKE FAILED -> abort" | tee -a "$ST"; touch models/lsse/MCV2_DONE; exit 1; fi

echo "=== train START $(date) ===" | tee -a "$ST"
( cd models/lsse && python train_lsse.py --config configs/multiconcept_nvg_v2.yaml ) 2>&1 | tee logs/mc_lsse_v2_train.log
echo "=== train END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== nudity ASR START $(date) ===" | tee -a "$ST"
python eval/xeval.py --models $M 2>&1 | tee logs/mc_lsse_v2_nudity.log
echo "=== nudity END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== violence Q16 START $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models $M 2>&1 | tee logs/mc_lsse_v2_violence.log
echo "=== violence END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== Van Gogh style START $(date) ===" | tee -a "$ST"
python models/fcf/eval_style_vangogh.py --models $M 2>&1 | tee logs/mc_lsse_v2_style.log
echo "=== style END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== COCO utility START $(date) ===" | tee -a "$ST"
python eval/eval_coco.py --models $M 2>&1 | tee logs/mc_lsse_v2_coco.log || echo "(eval_coco skipped/failed)" | tee -a "$ST"
echo "=== COCO END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== cost aggregate + gallery START $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/mc_lsse_v2_cost.log
python compare/build_live_gallery.py 2>&1 | tee logs/mc_lsse_v2_gallery.log
echo "=== gallery END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== MC-LSSE-v2 DONE $(date) ===" | tee -a "$ST"
touch models/lsse/MCV2_DONE
