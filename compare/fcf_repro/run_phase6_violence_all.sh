#!/usr/bin/env bash
# Phase 6: uniform cross-model violence axis. Waits for the main pipeline (P3-ext) to release the GPU
# (its PIPELINE_DONE sentinel), then runs eval_violence_q16.py over ALL Table-A models (I2P + RaB,
# UDA dropped as a dup). Resumable: the 3 FCF/raw models already have images and are just re-scored.
#
# Launch:  tmux new-session -d -s phase6 'bash compare/fcf_repro/run_phase6_violence_all.sh'
# Watch:   cat compare/fcf_repro/PHASE6_STATUS ; tail -f compare/fcf_repro/violence_q16_all.log
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
R=compare/fcf_repro

echo "=== PHASE6 waiter: waiting for $R/PIPELINE_DONE (P3-ext) $(date) ===" | tee "$R/PHASE6_STATUS"
while [ ! -f "$R/PIPELINE_DONE" ]; do sleep 120; done

echo "=== PHASE6 violence-all START $(date) ===" | tee -a "$R/PHASE6_STATUS"
python $R/eval_violence_q16.py 2>&1 | tee "$R/violence_q16_all.log"
echo "=== PHASE6 violence-all END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$R/PHASE6_STATUS"
touch "$R/PHASE6_DONE"
