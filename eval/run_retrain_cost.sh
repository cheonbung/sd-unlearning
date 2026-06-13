#!/usr/bin/env bash
# Env-aware TRAINING-COST measurement for baselines that lacked cost logging (ESD, DACE, Sph+OT
# single). Re-trains each with the SAME config into a THROWAWAY dir purely to time it, then writes
# train_cost.json (CostMeter schema) to the CANONICAL run-dir so aggregate_cost.py picks it up —
# WITHOUT disturbing the evaluated checkpoints. Queues behind the RPG-RT GPU sweep.
# FCF-official is skipped (trained with the authors' repo, not re-runnable here) -> stays pending.
# tmux: tmux new-session -d -s retraincost 'bash eval/run_retrain_cost.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/RETRAINCOST_STATUS
rm -f models/fcf/RETRAINCOST_DONE
echo "=== RETRAIN-COST START $(date) ===" | tee "$ST"

# queue behind the RPG-RT GPU sweep (shares the single GPU)
if tmux has-session -t rpgrt 2>/dev/null && [ ! -f models/odace/RPGRT_DONE ]; then
  echo "=== waiting for RPG-RT sweep (models/odace/RPGRT_DONE) $(date) ===" | tee -a "$ST"
  while tmux has-session -t rpgrt 2>/dev/null && [ ! -f models/odace/RPGRT_DONE ]; do sleep 120; done
  echo "=== RPG-RT cleared $(date) ===" | tee -a "$ST"
fi

write_cost() {  # $1=model $2=steps $3=wall_seconds $4=dest_json
  MODEL="$1" STEPS="$2" WALL="$3" DEST="$4" python - <<'PY'
import json, os, subprocess, datetime
try:
    gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"]).decode().strip().splitlines()[0]
except Exception:
    gpu = "unknown"
w = float(os.environ["WALL"])
d = {"model": os.environ["MODEL"], "training_free": False, "gpu": gpu, "gpu_count": 1,
     "trainable_params_M": None, "steps": int(os.environ["STEPS"]),
     "wall_seconds": round(w, 1), "gpu_hours": round(w/3600, 3), "peak_vram_gb": None,
     "torch": None, "timestamp": datetime.date.today().isoformat(), "status": "ok",
     "measured_by": "retrain_cost_runner"}
os.makedirs(os.path.dirname(os.environ["DEST"]), exist_ok=True)
json.dump(d, open(os.environ["DEST"], "w"), indent=2)
print("cost ->", os.environ["DEST"], d)
PY
}

run_one() {  # $1=label $2=workdir $3=cmd $4=steps $5=canonical_cost_json
  echo "=== retrain $1 START $(date) ===" | tee -a "$ST"
  T0=$(date +%s)
  ( cd "$2" && eval "$3" ) 2>&1 | tee "logs/retraincost_$1.log"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  WALL=$((T1-T0))
  echo "=== retrain $1 END $(date) (rc=$RC, wall=${WALL}s) ===" | tee -a "$ST"
  if [ "$RC" -eq 0 ]; then write_cost "$1" "$4" "$WALL" "$5" | tee -a "$ST"
  else echo "$1 retrain FAILED (rc=$RC) -> cost not written" | tee -a "$ST"; fi
}

# ESD-u (UNet erase, ~1000 steps) -> throwaway outputs/_costrun
run_one "esd_u" "models/esd" \
  "python train_esd.py --config configs/nudity_esd_u.yaml --output_dir outputs/_costrun" \
  1000 "models/esd/outputs/esd_u/train_cost.json"

# DACE nudity (TE, 30 epochs) and DACE-PLU
run_one "dace_v2" "models/dace" \
  "python train_dace.py --config configs/nudity_dace.yaml --output_dir outputs/_costrun" \
  30 "models/dace/outputs/dace_nudity/train_cost.json"
run_one "dace_plu" "models/dace" \
  "python train_dace.py --config configs/nudity_dace_plu.yaml --output_dir outputs/_costrun_plu" \
  30 "models/dace/outputs/dace_nudity_plu/train_cost.json"

# Sph+OT single (FCF-P spherical+OT, 60 epochs). novel/train.py has no --output_dir; it writes to
# the config's output_dir (outputs/fcf_p_v2_nudity) which is NOT the canonical _spherical_ot dir,
# so the evaluated checkpoint is untouched.
run_one "sph_ot" "models/novel" \
  "python train.py --config configs/nudity_v2.yaml --manifold spherical --ot_noise_file outputs/ot_noise/nudity_learned.json" \
  60 "models/novel/outputs/fcf_p_v2_nudity_spherical_ot/train_cost.json"

echo "=== cleanup throwaway dirs $(date) ===" | tee -a "$ST"
rm -rf models/esd/outputs/_costrun models/dace/outputs/_costrun models/dace/outputs/_costrun_plu models/novel/outputs/fcf_p_v2_nudity 2>/dev/null

echo "=== aggregate + gallery $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/retraincost_aggregate.log
python compare/build_live_gallery.py 2>&1 | tee logs/retraincost_gallery.log

echo "=== RETRAIN-COST DONE $(date) ===" | tee -a "$ST"
echo "NOTE: FCF-official needs the authors' repo to re-measure -> left pending." | tee -a "$ST"
echo "NEXT (WINDOWS git): git add models/fcf/train_cost.json eval/run_retrain_cost.sh ; git commit ; git push" | tee -a "$ST"
touch models/fcf/RETRAINCOST_DONE
