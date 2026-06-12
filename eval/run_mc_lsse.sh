#!/usr/bin/env bash
# LSSE multi-concept erasure (nudity + violence + Van Gogh): train (env-aware cost via CostMeter)
# -> eval 3 axes (nudity ASR / violence Q16 / Van Gogh style) + COCO utility -> aggregate cost
# -> regenerate gallery. Self-validating (1-epoch smoke gate) + auto-queued behind legacy50.
# NO git (WSL git forbidden; commit via Windows git after MC_DONE).
# tmux: tmux new-session -d -s mclsse 'bash eval/run_mc_lsse.sh'
# Watch: cat models/lsse/MC_STATUS ; tail -f logs/mc_lsse_train.log
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/lsse/MC_STATUS
rm -f models/lsse/MC_DONE
M=lsse_mc_nvg
echo "=== MC-LSSE START $(date) ===" | tee "$ST"

# --- queue behind the legacy50 GPU job if still running ---
if [ ! -f eval/LEGACY50_DONE ] && tmux has-session -t lsse 2>/dev/null; then
  echo "=== waiting for legacy50 (eval/LEGACY50_DONE) $(date) ===" | tee -a "$ST"
  while [ ! -f eval/LEGACY50_DONE ] && tmux has-session -t lsse 2>/dev/null; do sleep 120; done
  echo "=== legacy50 cleared $(date) ===" | tee -a "$ST"
fi

# --- 1-epoch smoke gate: abort early if the multi-concept path is broken ---
echo "=== smoke (1 epoch) START $(date) ===" | tee -a "$ST"
( cd models/lsse && python train_lsse.py --config configs/multiconcept_nvg.yaml \
    --num_epochs 1 --output_dir outputs/_smoke_mc ) 2>&1 | tee logs/mc_lsse_smoke.log
SMOKE_RC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SMOKE_RC) ===" | tee -a "$ST"
if [ "$SMOKE_RC" -ne 0 ]; then
  echo "SMOKE FAILED (rc=$SMOKE_RC) -> abort full run" | tee -a "$ST"
  touch models/lsse/MC_DONE
  exit 1
fi

echo "=== train START $(date) ===" | tee -a "$ST"
( cd models/lsse && python train_lsse.py --config configs/multiconcept_nvg.yaml ) 2>&1 | tee logs/mc_lsse_train.log
echo "=== train END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== nudity ASR START $(date) ===" | tee -a "$ST"
python eval/xeval.py --models $M 2>&1 | tee logs/mc_lsse_nudity.log
echo "=== nudity END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== violence Q16 START $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models $M 2>&1 | tee logs/mc_lsse_violence.log
echo "=== violence END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== Van Gogh style START $(date) ===" | tee -a "$ST"
python models/fcf/eval_style_vangogh.py --models $M 2>&1 | tee logs/mc_lsse_style.log
echo "=== style END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== COCO utility START $(date) ===" | tee -a "$ST"
python eval/eval_coco.py --models $M 2>&1 | tee logs/mc_lsse_coco.log || echo "(eval_coco skipped/failed)" | tee -a "$ST"
echo "=== COCO END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== cost aggregate + gallery START $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/mc_lsse_cost.log
python compare/build_live_gallery.py 2>&1 | tee logs/mc_lsse_gallery.log
echo "=== gallery END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== MC-LSSE DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/xeval.py eval/run_mc_lsse.sh models/lsse/train_lsse.py models/lsse/methods/lsse_trainer.py models/lsse/configs/multiconcept_nvg.yaml models/fcf/violence_q16.json models/fcf/style_vangogh.json models/fcf/train_cost.json compare/build_live_gallery.py ; git commit ; git push" | tee -a "$ST"
touch models/lsse/MC_DONE
