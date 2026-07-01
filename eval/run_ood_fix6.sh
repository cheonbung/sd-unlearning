#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix, CYCLE 5: ODACE collapse-mitigation via forget-set OOD augmentation.
# odace_v3 erases (ASR 4.0) but collapses on ring (person_prob 0.12). odace_ood appends synthetic OOD
# prompts to the forget set so the coherent negative-guidance target is applied to OOD too. One GPU job.
# tmux: tmux new-session -d -s oodfix6 'bash eval/run_ood_fix6.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX6_STATUS
rm -f models/fcf/OODFIX6_DONE
echo "oodfix6 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real tiny ODACE train (validates OOD-aug + odace path) ----
python models/odace/train_odace.py --config configs/nudity_odace_ood.yaml \
    --num_steps 4 --output_dir outputs/_smoke_odace_ood >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX6_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- full ODACE training (1500 steps) ----
echo "[train odace_ood] start $(date -u +%FT%TZ)" | tee -a "$ST"
python models/odace/train_odace.py --config configs/nudity_odace_ood.yaml 2>&1 | tee logs/ood6_train.log
echo "[train odace_ood] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

MODELS=odace_ood

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood6_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood6_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood6_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood6_gallery.log || true

echo "oodfix6 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX6_DONE
