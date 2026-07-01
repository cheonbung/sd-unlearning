#!/usr/bin/env bash
# CLEAN ABLATION: ODACE benign-anchor with OOD augmentation OFF (attribute the collapse fix to mechanism).
#   odace_benign_noood = erase_mode=benign_anchor, NO ood_aug_file. Same seed/steps/prompt as odace_benign.
# odace_benign (ring 0.99) had benign_anchor AND ood_aug confounded. Cycle-5 showed ood_aug alone fails
# (ring 0.18). If odace_benign_noood keeps ring ~0.99 -> the fix is the MECHANISM (redirect-to-benign).
# tmux: tmux new-session -d -s oodfix11 'bash eval/run_ood_fix11.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX11_STATUS
rm -f models/fcf/OODFIX11_DONE
echo "oodfix11 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate ----
python models/odace/train_odace.py --config configs/nudity_odace_benign_noood.yaml \
    --num_steps 3 --output_dir outputs/_smoke_oodfix11 >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX11_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

echo "[train odace_benign_noood] start $(date -u +%FT%TZ)" | tee -a "$ST"
python models/odace/train_odace.py --config configs/nudity_odace_benign_noood.yaml 2>&1 \
    | tee logs/ood11_train.log
echo "[train odace_benign_noood] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

MODELS=odace_benign_noood

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood11_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood11_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood11_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood11_gallery.log || true

echo "oodfix11 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX11_DONE
