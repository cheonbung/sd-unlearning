#!/usr/bin/env bash
# Fill the EMPTY cells in the live-gallery quantitative table (verified 2026-06-18):
#   Part A  Train GPU-h for 4 models that lacked CostMeter logging:
#           vanilla_lsse / lsse_plu / lsse_plu_w2 (LSSE sweep, ~60 steps each, seconds)
#           + odace_v15 (ODACE on SD1.5, ~0.65 GPU-h). Re-trains into THROWAWAY dirs purely to
#           time it, writes train_cost.json (CostMeter schema) to the CANONICAL run-dir so
#           aggregate_cost.py picks it up WITHOUT disturbing the evaluated checkpoints.
#   Part B  Nudity FULL-SET (1624 prompts x 5 attacks) for the 3 MC models that only had legacy
#           nudity: lsse_mc_nvg / odace_mc / lsse_mc_nvg_v2 -> merged into fullset_all.json.
#   Part C  aggregate_cost + rebuild gallery.
# fcf_p/fcf_e_official train-cost intentionally left "—" (authors' parent repo, not re-runnable here).
# Robust: each step is non-fatal (rc!=0 -> that cell stays pending, batch continues). Resumable
# full-set gen (skips existing PNGs). tmux: tmux new-session -d -s fillcells 'bash eval/run_fill_cells.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/FILLCELLS_STATUS
rm -f models/fcf/FILLCELLS_DONE
echo "=== FILL-CELLS START $(date) ===" | tee "$ST"

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
     "measured_by": "fill_cells_runner"}
os.makedirs(os.path.dirname(os.environ["DEST"]), exist_ok=True)
json.dump(d, open(os.environ["DEST"], "w"), indent=2)
print("cost ->", os.environ["DEST"], d)
PY
}

run_one() {  # $1=label $2=workdir $3=cmd $4=steps $5=canonical_cost_json
  echo "=== retrain $1 START $(date) ===" | tee -a "$ST"
  T0=$(date +%s)
  ( cd "$2" && eval "$3" ) 2>&1 | tee "logs/fillcells_$1.log"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  WALL=$((T1-T0))
  echo "=== retrain $1 END $(date) (rc=$RC, wall=${WALL}s) ===" | tee -a "$ST"
  if [ "$RC" -eq 0 ]; then write_cost "$1" "$4" "$WALL" "$5" | tee -a "$ST"
  else echo "$1 retrain FAILED (rc=$RC) -> cost not written (cell stays pending)" | tee -a "$ST"; fi
}

# ---------------------------------------------------------------- Part A: 4 missing Train GPU-h
# LSSE sweep variants (base config nudity_lsse.yaml; flags per models/lsse/README.md table).
run_one "vanilla_lsse" "models/lsse" \
  "python train_lsse.py --config configs/nudity_lsse.yaml --seed 42 --output_dir outputs/_costrun_vanilla" \
  60 "models/lsse/outputs/sweep/baseline_seed42/train_cost.json"
run_one "lsse_plu" "models/lsse" \
  "python train_lsse.py --config configs/nudity_lsse.yaml --seed 42 --use_plu --output_dir outputs/_costrun_plu" \
  60 "models/lsse/outputs/sweep/plu_seed42/train_cost.json"
run_one "lsse_plu_w2" "models/lsse" \
  "python train_lsse.py --config configs/nudity_lsse.yaml --seed 42 --use_plu --use_margin_cnp --output_dir outputs/_costrun_plu_w2" \
  60 "models/lsse/outputs/sweep/stack_plu_w2_seed42/train_cost.json"
# ODACE on SD1.5 (single-concept recipe, ~1500 steps).
run_one "odace_v15" "models/odace" \
  "python train_odace.py --config configs/nudity_odace_v15.yaml --output_dir outputs/_costrun_v15" \
  1500 "models/odace/outputs/odace_v15/train_cost.json"

echo "=== cleanup throwaway dirs $(date) ===" | tee -a "$ST"
rm -rf models/lsse/outputs/_costrun_vanilla models/lsse/outputs/_costrun_plu \
       models/lsse/outputs/_costrun_plu_w2 models/odace/outputs/_costrun_v15 2>/dev/null

# ---------------------------------------------------------------- Part B: MC nudity full-set (heavy)
echo "=== MC full-set nudity START $(date) ===" | tee -a "$ST"
python models/fcf/eval_fullset_all.py --models lsse_mc_nvg,odace_mc,lsse_mc_nvg_v2 \
  2>&1 | tee logs/fillcells_fullset_mc.log
echo "=== MC full-set nudity END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

# ---------------------------------------------------------------- Part C: aggregate + gallery
echo "=== aggregate + gallery $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/fillcells_aggregate.log
python compare/build_live_gallery.py 2>&1 | tee logs/fillcells_gallery.log

echo "=== FILL-CELLS DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add models/fcf/{train_cost.json,fullset_all.json} eval/run_fill_cells.sh ; commit ; push" | tee -a "$ST"
touch models/fcf/FILLCELLS_DONE
