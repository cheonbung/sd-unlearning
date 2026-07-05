#!/usr/bin/env bash
# Multi-seed error bars on the Ring-A-Bell collapse/coherence claim (lean). Queue step 3 of /goal "자동화".
# Re-generates ring images for CORE models with 2 new seeds, scores 4-lab ASR + CLIP coherence per
# seed, aggregates mean+-std. Smoke gate first (real tiny gen; abort if rc!=0). ~60-80 min on one 4070.
# tmux: tmux new-session -d -s multiseed 'bash eval/run_multiseed.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/MULTISEED_STATUS
rm -f models/fcf/MULTISEED_DONE
echo "multiseed start $(date -u +%FT%TZ)" | tee "$ST"

# --- smoke gate: real tiny gen+score on raw_v14 (1 new seed, 2 prompts); abort if rc!=0 ---
python eval/eval_multiseed.py --models raw_v14 --limit 2 --seeds 1000 >/dev/null 2>&1 \
  || { echo "GATE FAIL -> abort" | tee -a "$ST"; touch models/fcf/MULTISEED_DONE; exit 1; }
echo "[gate] ok $(date -u +%FT%TZ)" | tee -a "$ST"

# --- full CORE x 2 new seeds, full ring set ---
python eval/eval_multiseed.py 2>&1 | tee logs/multiseed.log
echo "[multiseed] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

echo "multiseed end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/MULTISEED_DONE
