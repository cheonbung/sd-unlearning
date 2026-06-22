#!/usr/bin/env bash
# Fill the LEGACY 50-prompt gallery table for the novel LSSE family (verified 2026-06-22).
# run_legacy50.sh covered the older roster; the capcnp/r2q models postdate it. Two parts:
#   A nudity 50-prompt ASR (5 attacks)  xeval.py --n_attack 50 -> eval/outputs/<m>/metrics.json
#       (NEW generation: these models only have full-set _fs images, not the 50-prompt dirs)
#   B violence 50-prompt subset  eval_violence_q16.py --limit 50 -> violence_q16_smoke.json
#       (the 5 novel + 3 MC models already have FULL violence images -> resumable re-score, cheap)
#   C regenerate gallery (legacy table I2P..UDA + mean + Violence-50 cells fill in)
# NOTE: the legacy table is a REFERENCE table; the frozen full-set table is the primary result.
# Both eval scripts are resumable (skip existing PNGs) + write JSON incrementally (other models kept).
# SMOKE-GATED: xeval n_attack=2 on one model first; abort all if it can't produce metrics.
# Launch (WSL conda env lsse):  bash eval/run_fill_legacy.sh
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=eval/FILLLEGACY_STATUS
rm -f eval/FILLLEGACY_DONE
echo "=== FILL-LEGACY START $(date) ===" | tee "$ST"

NUD5="lsse_capcnp,lsse_capcnp_zero,lsse_r2q_a,lsse_r2q_ab,lsse_r2q_violence"
VIO8="lsse_capcnp,lsse_capcnp_zero,lsse_r2q_a,lsse_r2q_ab,lsse_r2q_violence,lsse_mc_nvg,odace_mc,lsse_mc_nvg_v2"

# ---- SMOKE: xeval n_attack=2 on one model. Abort all if metrics.json not produced. ----
echo "=== SMOKE xeval (n_attack 2, lsse_capcnp) $(date) ===" | tee -a "$ST"
python eval/xeval.py --models lsse_capcnp --n_attack 2 2>&1 | tee logs/filllegacy_smoke.log
RC=${PIPESTATUS[0]}
echo "=== SMOKE rc=$RC ===" | tee -a "$ST"
if [ "$RC" -ne 0 ]; then
  echo "SMOKE FAILED (rc=$RC) -> ABORT, no heavy pass run" | tee -a "$ST"
  touch eval/FILLLEGACY_DONE; exit 1
fi

# ---- A: nudity 50-prompt ASR (5 novel models) ----
echo "=== A nudity50 (5 models) $(date) ===" | tee -a "$ST"
python eval/xeval.py --models "$NUD5" --n_attack 50 2>&1 | tee logs/filllegacy_nudity.log
echo "=== A end (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"

# ---- B: violence 50-prompt subset (5 novel + 3 MC; mostly re-score of existing full imgs) ----
echo "=== B violence50 (8 models, --limit 50) $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models "$VIO8" --limit 50 2>&1 | tee logs/filllegacy_violence.log
echo "=== B end (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"

# ---- C: rebuild gallery ----
echo "=== C gallery regen $(date) ===" | tee -a "$ST"
python compare/build_live_gallery.py 2>&1 | tee logs/filllegacy_gallery.log
echo "=== C end (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"

echo "=== FILL-LEGACY DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add models/fcf/violence_q16_smoke.json eval/run_fill_legacy.sh ; (also any tracked eval/outputs/*/metrics.json) ; commit ; push" | tee -a "$ST"
touch eval/FILLLEGACY_DONE
