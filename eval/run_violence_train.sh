#!/usr/bin/env bash
# Violence-erasure training queue for the in-repo trainable methods (LSSE x2, ODACE x3, ESD-u).
# R2q-ab violence already exists (lsse_r2q_violence). sph_ot + FCF-P/E handled separately.
# tmux: tmux new-session -d -s violtrain 'bash eval/run_violence_train.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/VIOLTRAIN_STATUS
: > "$ST"
echo "VIOLENCE TRAIN START $(date)" | tee -a "$ST"

# ---------- SMOKE GATE: tiny run of each distinct trainer; abort whole job on any rc!=0 ----------
smoke() {  # $1=label  $2..=cmd
  local label=$1; shift
  echo "[smoke] $label ..." | tee -a "$ST"
  "$@" 2>&1 | tee "logs/violtrain_smoke_${label}.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo "[smoke] $label FAIL rc=$rc -> ABORT" | tee -a "$ST"; exit 1; fi
  echo "[smoke] $label ok" | tee -a "$ST"
}
smoke lsse  python models/lsse/train_lsse.py  --config models/lsse/configs/violence_lsse_r2q_a.yaml --num_epochs 1
smoke odace python models/odace/train_odace.py --config configs/violence_odace.yaml --num_steps 10
smoke esd   python models/esd/train_esd.py    --config models/esd/configs/violence_esd_u.yaml --num_steps 10
echo "SMOKE OK $(date)" | tee -a "$ST"

# ---------- REAL TRAINING (each non-fatal, tee, append STATUS) ----------
run() {  # $1=label  $2..=cmd
  local label=$1; shift
  echo "[train] $label START $(date +%F_%H:%M:%S)" | tee -a "$ST"
  "$@" 2>&1 | tee "logs/violtrain_${label}.log"
  local rc=${PIPESTATUS[0]}
  echo "[train] $label rc=$rc END $(date +%F_%H:%M:%S)" | tee -a "$ST"
}
run lsse_r2q_a      python models/lsse/train_lsse.py   --config models/lsse/configs/violence_lsse_r2q_a.yaml
run lsse_geo_e2     python models/lsse/train_lsse.py   --config models/lsse/configs/violence_lsse_geo_e2.yaml
run odace_v3        python models/odace/train_odace.py --config configs/violence_odace.yaml
run odace_benign    python models/odace/train_odace.py --config configs/violence_odace_benign.yaml
run odace_benign_n1 python models/odace/train_odace.py --config configs/violence_odace_benign_n1.yaml
run esd_u           python models/esd/train_esd.py     --config models/esd/configs/violence_esd_u.yaml

echo "VIOLENCE TRAIN ALL DONE $(date)" | tee -a "$ST"
touch models/fcf/VIOLTRAIN_DONE
