#!/usr/bin/env bash
set -uo pipefail
PY="$HOME/miniconda3/envs/lsse/bin/python"
LOG=/mnt/d/unlearning/SD_unlearning/dace/outputs/dace_plu_run.log
: > "$LOG"
echo "DACE+PLU train start $(date -u +%FT%TZ)" | tee -a "$LOG"
cd /mnt/d/unlearning/SD_unlearning/dace
"$PY" train_dace.py --config configs/nudity_dace_plu.yaml --output_dir outputs/dace_nudity_plu >> "$LOG" 2>&1
echo "TRAIN_DONE rc=$? $(date -u +%FT%TZ)" | tee -a "$LOG"
if [ ! -f outputs/dace_nudity_plu/final/config.json ]; then
  echo "DACE_PLU_ALL_DONE fail=1 (no checkpoint)" | tee -a "$LOG"; exit 1
fi
cd /mnt/d/unlearning/SD_unlearning/lsse
"$PY" evaluate.py --encoder_dir ../dace/outputs/dace_nudity_plu/final --concept nudity \
    --eval_type asr --output_dir outputs/eval/xharness_dace_plu --num_images 50 >> "$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -f outputs/eval/xharness_dace_plu/eval_results.json ]; then
  echo "DACE_PLU_ALL_DONE fail=0 $(date -u +%FT%TZ)" | tee -a "$LOG"
else
  echo "DACE_PLU_ALL_DONE fail=1 rc=$rc $(date -u +%FT%TZ)" | tee -a "$LOG"
fi
