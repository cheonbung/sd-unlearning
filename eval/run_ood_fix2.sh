#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix, STAGE 2: apply the redirect+OOD fix to the broadly-damaged recipes.
#   capcnp_zero_ood = contrastive_ortho + perlayer + NO anchor + redirect + OOD aug
#   r2q_a_ood       = contrastive_ortho + perlayer + anchor    + redirect + OOD aug
# Chained after run_ood_fix.sh (flagship 3). One GPU job at a time. Eval is resumable.
# tmux: tmux new-session -d -s oodfix2 'bash eval/run_ood_fix2.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX2_STATUS
rm -f models/fcf/OODFIX2_DONE
echo "oodfix2 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real 1-epoch train of the no-anchor path (validates capcnp_zero_ood recipe) ----
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse_capcnp_zero_ood.yaml \
    --num_epochs 1 --output_dir outputs/_smoke_oodfix2 --no_diagnostics >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX2_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- training (60 epochs each), non-fatal, tee ----
for cfg in capcnp_zero_ood r2q_a_ood; do
  echo "[train $cfg] start $(date -u +%FT%TZ)" | tee -a "$ST"
  python models/lsse/train_lsse.py \
      --config models/lsse/configs/nudity_lsse_${cfg}.yaml 2>&1 | tee logs/ood2_train_${cfg}.log
  echo "[train $cfg] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"
done

MODELS=lsse_capcnp_zero_ood,lsse_r2q_a_ood

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood2_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood2_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood2_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood2_gallery.log || true

echo "oodfix2 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX2_DONE
