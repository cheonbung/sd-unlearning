#!/usr/bin/env bash
# Coherence / human-presence probe over existing attack images (no regen, CLIP only).
# Quantifies the Ring-A-Bell generation-collapse: person_prob low on ring_a_bell + high on i2p =
# OOD-specific collapse (ASR~0 for the wrong reason). -> models/fcf/coherence.json
# tmux: tmux new-session -d -s coherence 'bash eval/run_coherence.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/COHERENCE_STATUS
rm -f models/fcf/COHERENCE_DONE
echo "coherence start $(date -u +%FT%TZ)" | tee "$ST"

# smoke gate: tiny real run; abort if rc!=0
python models/fcf/eval_coherence.py --models raw_v14 --limit 4 >/dev/null 2>&1 \
  || { echo "GATE FAIL -> abort" | tee -a "$ST"; touch models/fcf/COHERENCE_DONE; exit 1; }
echo "[gate] ok" | tee -a "$ST"

python models/fcf/eval_coherence.py 2>&1 | tee logs/coherence.log
echo "rc=${PIPESTATUS[0]} done $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/COHERENCE_DONE
