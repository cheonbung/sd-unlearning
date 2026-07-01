#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix, CYCLE 7: GEODESIC in RAW space (sph_ot geometry) to BEAT sph_ot.
#   geo_raw_e2 = raw-space geodesic eta=2 ; geo_raw_e3 = raw-space geodesic eta=3
# cap_metric_mode="raw" (M^1/2=I) rotates on the TRUE CLIP embedding sphere (= sph_ot geometry) but with
# LSSE's PLU+CLM+retain-anchor on top. Goal: ring>0.7 AND 4-lab <= sph_ot (0.7) -> first LSSE to BEAT it.
# tmux: tmux new-session -d -s oodfix8 'bash eval/run_ood_fix8.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX8_STATUS
rm -f models/fcf/OODFIX8_DONE
echo "oodfix8 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real 1-epoch train of the raw+geodesic path ----
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse_geo_raw_e2.yaml \
    --num_epochs 1 --output_dir outputs/_smoke_oodfix8 --no_diagnostics >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX8_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

for cfg in geo_raw_e2 geo_raw_e3; do
  echo "[train $cfg] start $(date -u +%FT%TZ)" | tee -a "$ST"
  python models/lsse/train_lsse.py \
      --config models/lsse/configs/nudity_lsse_${cfg}.yaml 2>&1 | tee logs/ood8_train_${cfg}.log
  echo "[train $cfg] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"
done

MODELS=lsse_geo_raw_e2,lsse_geo_raw_e3

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood8_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood8_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood8_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood8_gallery.log || true

echo "oodfix8 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX8_DONE
