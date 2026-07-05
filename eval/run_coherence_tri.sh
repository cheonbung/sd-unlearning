#!/usr/bin/env bash
# Coherence triangulation: CLIP-independent corroboration (attack-FID + HOG/Haar detectors) of the
# OOD-collapse probe, over EXISTING ring/i2p images (no regen). Queue step 1 of the /goal "자동화" run.
# Smoke gate first (real tiny run; abort whole job if rc!=0), then the curated 14-model spread.
# tmux: tmux new-session -d -s cohtri 'bash eval/run_coherence_tri.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/COHTRI_STATUS
rm -f models/fcf/COHTRI_DONE
echo "cohtri start $(date -u +%FT%TZ)" | tee "$ST"

# --- smoke gate: tiny real run on one model (no FID at limit<40); abort if rc!=0 ---
python eval/eval_coherence_triangulate.py --models raw_v14 --limit 10 >/dev/null 2>&1 \
  || { echo "GATE FAIL -> abort" | tee -a "$ST"; touch models/fcf/COHTRI_DONE; exit 1; }
echo "[gate] ok $(date -u +%FT%TZ)" | tee -a "$ST"

# --- full curated spread (detectors + ring-vs-i2p attack FID) ---
python eval/eval_coherence_triangulate.py 2>&1 | tee logs/coherence_tri.log
echo "[tri] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

echo "cohtri end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/COHTRI_DONE
