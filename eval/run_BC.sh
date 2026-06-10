#!/usr/bin/env bash
# Gate B remainder (safe_neg) + Gate C (ODACE v1.5 train -> eval), one log, live via tee.
set -uo pipefail
PY="$HOME/miniconda3/envs/lsse/bin/python"
LOG=/mnt/d/unlearning/SD_unlearning/eval/outputs/gateBC.log
mkdir -p /mnt/d/unlearning/SD_unlearning/eval/outputs
: > "$LOG"

echo "[BC] safe_neg eval $(date -u +%FT%TZ)" | tee -a "$LOG"
cd /mnt/d/unlearning/SD_unlearning/xmodel
"$PY" xeval.py --models safe_neg 2>&1 | tee -a "$LOG"

echo "[BC] ODACE v1.5 train $(date -u +%FT%TZ)" | tee -a "$LOG"
cd /mnt/d/unlearning/SD_unlearning/odace
"$PY" train_odace.py --config configs/nudity_odace_v15.yaml 2>&1 | tee -a "$LOG"

if [ -f outputs/odace_v15/final/config.json ]; then
  echo "[BC] ODACE v1.5 eval $(date -u +%FT%TZ)" | tee -a "$LOG"
  cd /mnt/d/unlearning/SD_unlearning/xmodel
  "$PY" xeval.py --models odace_v15 2>&1 | tee -a "$LOG"
else
  echo "[BC] ODACE v1.5 train produced no checkpoint -- skipping eval" | tee -a "$LOG"
fi

echo "BC_ALL_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
