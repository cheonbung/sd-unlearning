#!/usr/bin/env bash
# Cross-model unified-harness ASR eval.
# Brings FCF-family checkpoints INTO the lsse/evaluate.py harness so that
# baseline/generation/prompts are byte-identical to the already-computed
# LSSE numbers (single pipeline => apples-to-apples comparison).
# Reads parent checkpoints (weights only, no code import); writes to lsse/outputs only.
set -uo pipefail
cd /mnt/d/unlearning/SD_unlearning/lsse
PY="$HOME/miniconda3/envs/lsse/bin/python"
LOG=outputs/xharness.log
: > "$LOG"

# name -> encoder checkpoint dir (relative to lsse/)
names=(fcf_p fcf_e sph_ot)
declare -A ENC=(
  [fcf_p]="../fcf/outputs/fcf_p_nudity/final"
  [fcf_e]="../fcf/outputs/fcf_e_nudity/final"
  [sph_ot]="../../novel/outputs/fcf_p_v2_nudity_spherical_ot/final"
)

echo "XH start $(date -u +%FT%TZ)" | tee -a "$LOG"
fail=0
for name in "${names[@]}"; do
  enc="${ENC[$name]}"
  out="outputs/eval/xharness_${name}"
  if [ -f "$out/eval_results.json" ]; then
    echo "[$name] SKIP exists" | tee -a "$LOG"; continue
  fi
  if [ ! -f "$enc/config.json" ]; then
    echo "[$name] FAIL missing checkpoint $enc" | tee -a "$LOG"; fail=$((fail+1)); continue
  fi
  echo "[$name] START enc=$enc $(date -u +%FT%TZ)" | tee -a "$LOG"
  "$PY" evaluate.py --encoder_dir "$enc" --concept nudity --eval_type asr \
      --output_dir "$out" --num_images 50 >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$out/eval_results.json" ]; then
    echo "[$name] OK $(date -u +%FT%TZ)" | tee -a "$LOG"
  else
    echo "[$name] FAIL rc=$rc $(date -u +%FT%TZ)" | tee -a "$LOG"; fail=$((fail+1))
  fi
done
echo "XH end fail=$fail $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "XHARNESS_DONE fail=$fail" | tee -a "$LOG"
