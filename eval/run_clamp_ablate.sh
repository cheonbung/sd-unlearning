#!/usr/bin/env bash
# Phase 0 gate: inference norm-clamp ablation for lsse_r2q_ab + coherence re-score.
# Generates eval/outputs/lsse_r2q_ab_clamp_fs/{ring_a_bell,i2p} then scores person_prob.
# Gate PASS if clamp ring person_prob recovers well above the un-clamped 0.15.
# tmux: tmux new-session -d -s clamp 'bash eval/run_clamp_ablate.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/CLAMP_STATUS
rm -f models/fcf/CLAMP_DONE
echo "clamp ablate start $(date -u +%FT%TZ)" | tee "$ST"

# smoke gate
python eval/ablate_norm_clamp.py --limit 2 >/dev/null 2>&1 \
  || { echo "GATE FAIL -> abort" | tee -a "$ST"; touch models/fcf/CLAMP_DONE; exit 1; }
echo "[gate] ok" | tee -a "$ST"

# full ablation generation
python eval/ablate_norm_clamp.py 2>&1 | tee logs/clamp_ablate.log
echo "[gen] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

# coherence re-score on the clamp variant
python models/fcf/eval_coherence.py --models lsse_r2q_ab_clamp 2>&1 | tee logs/clamp_coherence.log
echo "[coh] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

echo "clamp ablate end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/CLAMP_DONE
