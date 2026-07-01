#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix, CYCLE 3: RAW-space projection (M^1/2=I) with OVERSHOOT sweep.
#   raw_proj_s2 = raw-space redirect, strength=2 (overshoot past benign)
#   raw_proj_s4 = raw-space redirect, strength=4 (stronger overshoot)
# Goal: break the coherence<->ASR tension (read-out coordinate redirect could not). One GPU job.
# tmux: tmux new-session -d -s oodfix4 'bash eval/run_ood_fix4.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX4_STATUS
rm -f models/fcf/OODFIX4_DONE
echo "oodfix4 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real 1-epoch train of the raw-space path (validates M=I + overshoot) ----
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse_raw_proj_s2.yaml \
    --num_epochs 1 --output_dir outputs/_smoke_oodfix4 --no_diagnostics >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX4_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- training (60 epochs each) ----
for cfg in raw_proj_s2 raw_proj_s4; do
  echo "[train $cfg] start $(date -u +%FT%TZ)" | tee -a "$ST"
  python models/lsse/train_lsse.py \
      --config models/lsse/configs/nudity_lsse_${cfg}.yaml 2>&1 | tee logs/ood4_train_${cfg}.log
  echo "[train $cfg] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"
done

MODELS=lsse_raw_proj_s2,lsse_raw_proj_s4

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood4_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood4_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood4_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood4_gallery.log || true

echo "oodfix4 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX4_DONE
