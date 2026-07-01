#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix, CYCLE 6: MULTI-DIRECTION geodesic (K=4) to push LSSE past sph_ot.
#   geo_tk05 = K=4 geodesic, per-axis eta=0.5 ; geo_tk1 = per-axis eta=1.0
# Rotate frozen read-out away from K=4 concept axes on the sphere (on-manifold) -> more erasure per
# unit coherence loss. Goal: higher ring AND lower ASR than geo_e2 (0.58/15.3) / sph_ot (0.79/14).
# tmux: tmux new-session -d -s oodfix7 'bash eval/run_ood_fix7.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX7_STATUS
rm -f models/fcf/OODFIX7_DONE
echo "oodfix7 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real 1-epoch train of the geo+topK path ----
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse_geo_tk05.yaml \
    --num_epochs 1 --output_dir outputs/_smoke_oodfix7 --no_diagnostics >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX7_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

for cfg in geo_tk05 geo_tk1; do
  echo "[train $cfg] start $(date -u +%FT%TZ)" | tee -a "$ST"
  python models/lsse/train_lsse.py \
      --config models/lsse/configs/nudity_lsse_${cfg}.yaml 2>&1 | tee logs/ood7_train_${cfg}.log
  echo "[train $cfg] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"
done

MODELS=lsse_geo_tk05,lsse_geo_tk1

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood7_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood7_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood7_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood7_gallery.log || true

echo "oodfix7 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX7_DONE
