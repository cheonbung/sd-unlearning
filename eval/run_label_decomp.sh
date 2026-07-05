#!/usr/bin/env bash
# 8-lab vs 4-lab decomposition over existing full-set images (no regen). Queue step 2 of /goal "자동화".
# Proves the redirect models' higher 8-lab ASR is covered/clothed detections, not exposed nudity.
# Smoke gate first (real tiny run; abort whole job if rc!=0), then the curated 8-model set.
# tmux: tmux new-session -d -s decomp 'bash eval/run_label_decomp.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/DECOMP_STATUS
rm -f models/fcf/DECOMP_DONE
echo "decomp start $(date -u +%FT%TZ)" | tee "$ST"

# --- smoke gate: tiny real run; abort if rc!=0 ---
python eval/eval_label_decomp.py --models raw_v14 --limit 5 >/dev/null 2>&1 \
  || { echo "GATE FAIL -> abort" | tee -a "$ST"; touch models/fcf/DECOMP_DONE; exit 1; }
echo "[gate] ok $(date -u +%FT%TZ)" | tee -a "$ST"

# --- full curated set (per-image 4lab / 8lab / covered-only + per-label rates) ---
python eval/eval_label_decomp.py 2>&1 | tee logs/label_decomp.log
echo "[decomp] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

echo "decomp end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/DECOMP_DONE
