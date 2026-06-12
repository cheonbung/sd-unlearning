#!/usr/bin/env bash
# Violence(Q16) + Van Gogh art-style LOCALITY eval, then refresh the live gallery HTML.
# Self-contained background runner: NO git (WSL git is forbidden; commit/push is done with Windows
# git after VS_DONE appears). Single GPU -> serial. Both evals are resumable + per-model
# crash-tolerant + write their JSON incrementally, so a kill/restart of this runner continues where
# it stopped. tee gives live tmux progress. Mirrors run_pipeline.sh conventions.
#
# Launch:  tmux new-session -d -s lsse 'bash models/fcf/run_violence_style.sh'
# Watch:   cat models/fcf/VS_STATUS ; tail -f logs/violence_q16_full.log
# Done:    models/fcf/VS_DONE appears -> then commit via WINDOWS git (command echoed into VS_STATUS).
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

R=models/fcf
ST=$R/VS_STATUS
rm -f "$R/VS_DONE"
echo "=== VS PIPELINE START $(date) ===" | tee "$ST"

run_phase () {  # $1=label  $2=logfile  $3..=command
  local label="$1"; shift
  local log="$1"; shift
  echo "=== $label START $(date) ===" | tee -a "$ST"
  "$@" 2>&1 | tee "$log"
  echo "=== $label END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
}

run_phase "VIOLENCE_q16_all"  "logs/violence_q16_full.log" \
  python $R/eval_violence_q16.py

run_phase "STYLE_vangogh_all" "logs/style_vangogh_full.log" \
  python $R/eval_style_vangogh.py

run_phase "GALLERY_regen"     "logs/gallery_regen.log" \
  python compare/build_live_gallery.py

echo "=== VS PIPELINE DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git, not WSL): git add models/fcf/violence_q16.json models/fcf/style_vangogh.json models/fcf/eval_style_vangogh.py models/fcf/run_violence_style.sh compare/build_live_gallery.py compare/comparison_gallery_live.html ; git commit ; git push" | tee -a "$ST"
touch "$R/VS_DONE"
