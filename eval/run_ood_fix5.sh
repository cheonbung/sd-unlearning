#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix, CYCLE 4: MANIFOLD-PRESERVING geodesic erasure (sph_ot ingredient in LSSE).
#   geo_e1 = read-out geodesic, eta=1 ; geo_e2 = eta=2 (rotate frozen read-out away from concept mean
#   along the sphere -> stay on-manifold, no collapse). Goal: low ASR AND high ring person_prob.
# tmux: tmux new-session -d -s oodfix5 'bash eval/run_ood_fix5.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX5_STATUS
rm -f models/fcf/OODFIX5_DONE
echo "oodfix5 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real 1-epoch train of the geodesic path ----
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse_geo_e1.yaml \
    --num_epochs 1 --output_dir outputs/_smoke_oodfix5 --no_diagnostics >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX5_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

for cfg in geo_e1 geo_e2; do
  echo "[train $cfg] start $(date -u +%FT%TZ)" | tee -a "$ST"
  python models/lsse/train_lsse.py \
      --config models/lsse/configs/nudity_lsse_${cfg}.yaml 2>&1 | tee logs/ood5_train_${cfg}.log
  echo "[train $cfg] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"
done

MODELS=lsse_geo_e1,lsse_geo_e2

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood5_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood5_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood5_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood5_gallery.log || true

echo "oodfix5 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX5_DONE
