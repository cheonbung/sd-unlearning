#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix, CYCLE 8: ODACE BENIGN-ANCHOR redirect (sph_ot-style fix, UNet space).
#   odace_benign = erase_mode=benign_anchor -> redirect concept output toward "clothed person" output
# All prior cycles PUSHED AWAY (collapse on OOD); sph_ot/FCF-P win by REDIRECTING to a coherent benign
# target. This applies that principle in ODACE's UNet output space (strongest erasure: odace_v3 4-lab 0.7).
# Goal: ring>0.7 AND 4-lab <= sph_ot (0.7) -> first method to BEAT sph_ot.
# tmux: tmux new-session -d -s oodfix9 'bash eval/run_ood_fix9.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX9_STATUS
rm -f models/fcf/OODFIX9_DONE
echo "oodfix9 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real short train of the NEW benign_anchor erase path ----
python models/odace/train_odace.py --config configs/nudity_odace_benign.yaml \
    --num_steps 3 --output_dir outputs/_smoke_oodfix9 >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX9_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

echo "[train odace_benign] start $(date -u +%FT%TZ)" | tee -a "$ST"
python models/odace/train_odace.py --config configs/nudity_odace_benign.yaml 2>&1 \
    | tee logs/ood9_train.log
echo "[train odace_benign] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

MODELS=odace_benign

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood9_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood9_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood9_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood9_gallery.log || true

echo "oodfix9 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX9_DONE
