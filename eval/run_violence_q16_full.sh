#!/usr/bin/env bash
# Paper-aligned violence per-attack Q16 full re-eval (see memory: violence-eval-paper-alignment).
# Smoke gate -> full eval (7 models x 3 attacks x {757,249,756} prompts) -> gallery rebuild.
# tmux: tmux new-session -d -s violq16 'bash eval/run_violence_q16_full.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/VIOLQ16FULL_STATUS
: > "$ST"
echo "VIOLENCE Q16 FULL START $(date)" | tee -a "$ST"

echo "[smoke] START $(date +%F_%T)" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --limit 3 2>&1 | tee logs/violq16_smoke.log
rc=${PIPESTATUS[0]}
echo "[smoke] rc=$rc END $(date +%F_%T)" | tee -a "$ST"
if [ "$rc" -ne 0 ]; then
  echo "SMOKE FAILED rc=$rc -> ABORT" | tee -a "$ST"
  touch models/fcf/VIOLQ16FULL_DONE
  exit 1
fi

run() {  # $1=label  $2=shell-string (non-fatal, tee, rc logged)
  local label=$1; shift
  echo "[$label] START $(date +%F_%T)" | tee -a "$ST"
  bash -c "$1" 2>&1 | tee "logs/violq16_${label}.log"
  local rc=${PIPESTATUS[0]}
  echo "[$label] rc=$rc END $(date +%F_%T)" | tee -a "$ST"
  return 0
}

run full_eval "python models/fcf/eval_violence_q16.py"
run gallery    "python compare/build_live_gallery.py"

echo "VIOLENCE Q16 FULL ALL DONE $(date)" | tee -a "$ST"
touch models/fcf/VIOLQ16FULL_DONE
