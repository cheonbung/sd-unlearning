#!/usr/bin/env bash
# Phase 3-ext + Phase 4 + Phase 5 sequential GPU pipeline for the FCF paper-aligned reproduction.
# Single RTX 4070 -> phases run serially. Ordered shortest->longest so quicker results land first:
#   Phase 5 violence Q16 locality (~3-4h) -> Phase 4 COCO FID-5K (~8-9h) -> Phase 3-ext full-set (~20h).
# Each script is independently resumable + per-model crash-tolerant + writes JSON incrementally,
# so a kill/restart of this runner continues where it stopped. tee gives live tmux progress.
#
# Launch:  tmux new-session -d -s fcfrepro 'bash compare/fcf_repro/run_pipeline.sh'
# Watch:   tmux attach -t fcfrepro     (or: tail -f compare/fcf_repro/pipeline.log)
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse

R=compare/fcf_repro
ST=$R/PIPELINE_STATUS
echo "=== PIPELINE START $(date) ===" | tee "$ST"

run_phase () {  # $1=label  $2=logfile  $3..=command
  local label="$1"; shift
  local log="$1"; shift
  echo "=== $label START $(date) ===" | tee -a "$ST"
  "$@" 2>&1 | tee "$log"
  echo "=== $label END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
}

run_phase "PHASE5_violence" "$R/violence_q16.log" \
  python $R/eval_violence_q16.py

run_phase "PHASE4_coco_fid5k" "$R/coco5k.log" \
  python $R/eval_coco_fid5k.py

run_phase "PHASE3ext_fullset_all" "$R/fullset_all.log" \
  python $R/eval_fullset_all.py

echo "=== PIPELINE DONE $(date) ===" | tee -a "$ST"
touch "$R/PIPELINE_DONE"
