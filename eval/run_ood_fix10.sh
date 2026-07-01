#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix, CYCLE 9: ODACE BENIGN-NEG hybrid to DOMINATE sph_ot.
#   odace_benign_n05 = benign anchor + lambda=0.5 push ; odace_benign_n1 = lambda=1.0
# odace_benign (pure anchor) solved collapse (ring 0.99) but under-erases (4-lab 4.4) vs sph_ot (0.79/0.7).
# Trade its large coherence headroom for erasure: target = e_benign - lambda*(e_p - e_benign).
# Goal: ring>0.79 AND 4-lab<=0.7 = first method to DOMINATE sph_ot on both axes.
# tmux: tmux new-session -d -s oodfix10 'bash eval/run_ood_fix10.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX10_STATUS
rm -f models/fcf/OODFIX10_DONE
echo "oodfix10 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real short train of the NEW benign_neg erase path ----
python models/odace/train_odace.py --config configs/nudity_odace_benign_n05.yaml \
    --num_steps 3 --output_dir outputs/_smoke_oodfix10 >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX10_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

for cfg in benign_n05 benign_n1; do
  echo "[train odace_$cfg] start $(date -u +%FT%TZ)" | tee -a "$ST"
  python models/odace/train_odace.py --config configs/nudity_odace_${cfg}.yaml 2>&1 \
      | tee logs/ood10_train_${cfg}.log
  echo "[train odace_$cfg] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"
done

MODELS=odace_benign_n05,odace_benign_n1

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood10_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood10_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood10_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood10_gallery.log || true

echo "oodfix10 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX10_DONE
