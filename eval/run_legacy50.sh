#!/usr/bin/env bash
# Unified 50-prompt refresh for the LEGACY gallery table:
#   1) nudity 50-prompt ASR for the 7 models missing eval/outputs/<m>/metrics.json  (xeval.py)
#   2) violence 50-prompt subset over all 20 models (eval_violence_q16.py --limit 50
#      -> models/fcf/violence_q16_smoke.json)
#   (art-style VanGogh is already a 50-prompt set -> style_vangogh.json, no GPU needed)
#   3) regenerate the live gallery (legacy table gains Violence-50 + VanGogh columns)
# Self-contained background runner: NO git (WSL git forbidden; commit via Windows git after
# LEGACY50_DONE). Both eval scripts are resumable (skip existing images) + write JSON incrementally.
# Mirrors run_violence_style.sh conventions.
#
# Launch:  tmux new-session -d -s lsse 'bash eval/run_legacy50.sh'
# Watch:   cat eval/LEGACY50_STATUS ; tail -f logs/legacy50_violence.log
# Done:    eval/LEGACY50_DONE appears.
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=eval/LEGACY50_STATUS
rm -f eval/LEGACY50_DONE
NUD7=vanilla_lsse,lsse_plu,lsse_plu_w2,sph_ot,dace_v2,dace_plu,odace_v2
echo "=== LEGACY50 START $(date) ===" | tee "$ST"

echo "=== nudity50 (7 missing) START $(date) ===" | tee -a "$ST"
python eval/xeval.py --models "$NUD7" --n_attack 50 2>&1 | tee logs/legacy50_nudity.log
echo "=== nudity50 END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== violence50 (20 models, --limit 50) START $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --limit 50 2>&1 | tee logs/legacy50_violence.log
echo "=== violence50 END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== gallery regen START $(date) ===" | tee -a "$ST"
python compare/build_live_gallery.py 2>&1 | tee logs/legacy50_gallery.log
echo "=== gallery END   $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

echo "=== LEGACY50 DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add models/fcf/violence_q16_smoke.json compare/build_live_gallery.py eval/run_legacy50.sh ; (also any tracked eval/outputs/*/metrics.json) ; git commit ; git push" | tee -a "$ST"
touch eval/LEGACY50_DONE
