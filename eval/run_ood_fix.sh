#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix queue: train 3 r2q_ab variants then eval ASR + coherence.
#   P1 slerp     = norm-preserving (manifold) read-out erasure
#   P2 redirect  = shift concept axis to benign anchor coordinate
#   P3 ood       = redirect + synthetic OOD-aware implicit augmentation
# One GPU job at a time, chained. Smoke gate first (real 1-epoch train of the most complex path).
# tmux: tmux new-session -d -s oodfix 'bash eval/run_ood_fix.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX_STATUS
rm -f models/fcf/OODFIX_DONE
echo "oodfix start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real 1-epoch train of the redirect+OOD path (validates new code paths) ----
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse_r2q_ood.yaml \
    --num_epochs 1 --output_dir outputs/_smoke_oodfix --no_diagnostics >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- P1/P2/P3 training (60 epochs each), non-fatal, tee for live progress ----
for cfg in slerp redirect ood; do
  echo "[train $cfg] start $(date -u +%FT%TZ)" | tee -a "$ST"
  python models/lsse/train_lsse.py \
      --config models/lsse/configs/nudity_lsse_r2q_${cfg}.yaml 2>&1 | tee logs/ood_train_${cfg}.log
  echo "[train $cfg] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"
done

MODELS=lsse_r2q_slerp,lsse_r2q_redirect,lsse_r2q_ood

# ---- frozen full-set ASR (generates eval/outputs/<key>_fs/<attack> + fullset_all.json) ----
python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- coherence person_prob (ring vs i2p) over the freshly generated _fs imgs (CLIP only) ----
python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- cost aggregate + gallery refresh (best-effort; new-model gallery rows added in harvest) ----
python eval/aggregate_cost.py 2>&1 | tee -a logs/ood_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood_gallery.log || true

echo "oodfix end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX_DONE
