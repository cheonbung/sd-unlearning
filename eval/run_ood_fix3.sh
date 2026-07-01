#!/usr/bin/env bash
# Ring-A-Bell OOD-collapse fix, CYCLE 2: multi-direction (top-K=4) redirect + OOD on r2q_ab recipe.
# Goal: recover coherence (ring person_prob high) WHILE keeping nudity ASR low (single-dir fixes in
# stage-1/2 recovered coherence but lost ASR 3.1->42-52). One GPU job. Eval resumable.
# tmux: tmux new-session -d -s oodfix3 'bash eval/run_ood_fix3.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/OODFIX3_STATUS
rm -f models/fcf/OODFIX3_DONE
echo "oodfix3 start $(date -u +%FT%TZ)" | tee "$ST"

# ---- smoke gate: real 1-epoch train (validates top-K multi-direction code path) ----
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse_r2q_topk.yaml \
    --num_epochs 1 --output_dir outputs/_smoke_oodfix3 --no_diagnostics >/dev/null 2>&1 \
  || { echo "GATE FAIL (train) -> abort" | tee -a "$ST"; touch models/fcf/OODFIX3_DONE; exit 1; }
echo "[gate] train ok $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- training (60 epochs) ----
echo "[train topk] start $(date -u +%FT%TZ)" | tee -a "$ST"
python models/lsse/train_lsse.py \
    --config models/lsse/configs/nudity_lsse_r2q_topk.yaml 2>&1 | tee logs/ood3_train_topk.log
echo "[train topk] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

MODELS=lsse_r2q_topk

python models/fcf/eval_fullset_all.py --models "$MODELS" 2>&1 | tee logs/ood3_fullset.log
echo "[fullset] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python models/fcf/eval_coherence.py --models "$MODELS" 2>&1 | tee logs/ood3_coherence.log
echo "[coherence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python eval/aggregate_cost.py 2>&1 | tee -a logs/ood3_cost.log || true
python compare/build_live_gallery.py 2>&1 | tee -a logs/ood3_gallery.log || true

echo "oodfix3 end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/OODFIX3_DONE
