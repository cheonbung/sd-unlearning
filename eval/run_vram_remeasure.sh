#!/usr/bin/env bash
# Fill the EMPTY peak_vram_gb cells. The MC models got VRAM from their training CostMeter; the
# retrain/fill runners used EXTERNAL wall-timing (write_cost peak_vram_gb=None) so ~9 trained roster
# models show null. CostMeter writes peak_vram_gb to <output_dir>/train_cost.json (torch per-process
# max_memory_allocated). So we RE-TRAIN each into a THROWAWAY dir purely to capture VRAM, then patch
# peak_vram_gb / trainable_params_M / torch into the CANONICAL train_cost.json (preserving its
# gpu_hours), WITHOUT disturbing the evaluated checkpoints. fcf_p/fcf_e = authors' parent repo -> stay "—".
#
# SMOKE-GATED (per the smoke-test-before-bg-jobs rule): runs the cheapest model (vanilla_lsse, ~1min)
# FIRST and aborts if VRAM was not captured, before committing to the ~3h odace/esd re-trains.
# tmux: tmux new-session -d -s vram 'bash eval/run_vram_remeasure.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/VRAM_STATUS
rm -f models/fcf/VRAM_DONE
echo "=== VRAM-REMEASURE START $(date) ===" | tee "$ST"

patch_vram() {  # $1=model  $2=throwaway train_cost.json  $3=canonical train_cost.json
  MODEL="$1" THROW="$2" CANON="$3" python - <<'PY'
import json, os
model, src, dst = os.environ["MODEL"], os.environ["THROW"], os.environ["CANON"]
if not os.path.exists(src):
    print(f"[patch] {model}: NO throwaway cost json at {src} (train script wrote none) -> VRAM stays null")
    raise SystemExit(0)
s = json.load(open(src))
vram, params, torch_v = s.get("peak_vram_gb"), s.get("trainable_params_M"), s.get("torch")
d = json.load(open(dst)) if os.path.exists(dst) else {"model": model, "training_free": False}
if vram is not None: d["peak_vram_gb"] = vram
if params is not None and d.get("trainable_params_M") is None: d["trainable_params_M"] = params
if torch_v is not None and d.get("torch") is None: d["torch"] = torch_v
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(d, open(dst, "w"), indent=2)
print(f"[patch] {model}: peak_vram_gb={vram} -> {dst}")
PY
}

run_vram() {  # $1=label $2=workdir $3=train_cmd(writes throwaway) $4=throwaway_json $5=canonical_json
  echo "=== vram $1 START $(date) ===" | tee -a "$ST"
  ( cd "$2" && eval "$3" ) 2>&1 | tee "logs/vram_$1.log"
  RC=${PIPESTATUS[0]}
  echo "=== vram $1 END $(date) (rc=$RC) ===" | tee -a "$ST"
  if [ "$RC" -eq 0 ]; then patch_vram "$1" "$4" "$5" | tee -a "$ST"
  else echo "$1 retrain FAILED (rc=$RC) -> VRAM stays null" | tee -a "$ST"; fi
}

# ---- SMOKE: vanilla_lsse (~1min). Abort the whole run if VRAM was NOT captured. ----
run_vram "vanilla_lsse" "models/lsse" \
  "python train_lsse.py --config configs/nudity_lsse.yaml --seed 42 --output_dir outputs/_vram_vanilla" \
  "models/lsse/outputs/_vram_vanilla/train_cost.json" \
  "models/lsse/outputs/sweep/baseline_seed42/train_cost.json"
SMOKE_VRAM=$(MODELJSON=models/lsse/outputs/sweep/baseline_seed42/train_cost.json python - <<'PY'
import json, os
try:
    print(json.load(open(os.environ["MODELJSON"])).get("peak_vram_gb"))
except Exception:
    print("None")
PY
)
echo "=== SMOKE peak_vram_gb=$SMOKE_VRAM ===" | tee -a "$ST"
if [ "$SMOKE_VRAM" = "None" ] || [ -z "$SMOKE_VRAM" ]; then
  echo "SMOKE FAILED: CostMeter did not capture VRAM via this approach -> ABORT (no time wasted on odace/esd)" | tee -a "$ST"
  touch models/fcf/VRAM_DONE
  exit 1
fi

# ---- full run (smoke passed) ----
run_vram "lsse_plu" "models/lsse" \
  "python train_lsse.py --config configs/nudity_lsse.yaml --seed 42 --use_plu --output_dir outputs/_vram_plu" \
  "models/lsse/outputs/_vram_plu/train_cost.json" \
  "models/lsse/outputs/sweep/plu_seed42/train_cost.json"
run_vram "lsse_plu_w2" "models/lsse" \
  "python train_lsse.py --config configs/nudity_lsse.yaml --seed 42 --use_plu --use_margin_cnp --output_dir outputs/_vram_plu_w2" \
  "models/lsse/outputs/_vram_plu_w2/train_cost.json" \
  "models/lsse/outputs/sweep/stack_plu_w2_seed42/train_cost.json"
run_vram "dace_v2" "models/dace" \
  "python train_dace.py --config configs/nudity_dace.yaml --output_dir outputs/_vram_dace" \
  "models/dace/outputs/_vram_dace/train_cost.json" \
  "models/dace/outputs/dace_nudity/train_cost.json"
run_vram "dace_plu" "models/dace" \
  "python train_dace.py --config configs/nudity_dace_plu.yaml --output_dir outputs/_vram_dace_plu" \
  "models/dace/outputs/_vram_dace_plu/train_cost.json" \
  "models/dace/outputs/dace_nudity_plu/train_cost.json"
run_vram "sph_ot" "models/novel" \
  "python train.py --config configs/nudity_v2.yaml --manifold spherical --ot_noise_file outputs/ot_noise/nudity_learned.json" \
  "models/novel/outputs/fcf_p_v2_nudity/train_cost.json" \
  "models/novel/outputs/fcf_p_v2_nudity_spherical_ot/train_cost.json"
run_vram "esd_u" "models/esd" \
  "python train_esd.py --config configs/nudity_esd_u.yaml --output_dir outputs/_vram_esd" \
  "models/esd/outputs/_vram_esd/train_cost.json" \
  "models/esd/outputs/esd_u/train_cost.json"
run_vram "odace_v3" "models/odace" \
  "python train_odace.py --config configs/nudity_odace.yaml --output_dir outputs/_vram_odace" \
  "models/odace/outputs/_vram_odace/train_cost.json" \
  "models/odace/outputs/odace_v3/train_cost.json"
# odace_v2 = same single-concept recipe/arch as odace_v3 -> copy the just-measured VRAM
MODELJSON=models/odace/outputs/odace_v3/train_cost.json DST=models/odace/outputs/odace_v2/train_cost.json python - <<'PY'
import json, os
src, dst = os.environ["MODELJSON"], os.environ["DST"]
if os.path.exists(src) and os.path.exists(dst):
    s, d = json.load(open(src)), json.load(open(dst))
    if s.get("peak_vram_gb") is not None:
        d["peak_vram_gb"] = s["peak_vram_gb"]
        json.dump(d, open(dst, "w"), indent=2)
        print("[copy] odace_v2 peak_vram_gb <- odace_v3 =", s["peak_vram_gb"])
PY
run_vram "odace_v15" "models/odace" \
  "python train_odace.py --config configs/nudity_odace_v15.yaml --output_dir outputs/_vram_v15" \
  "models/odace/outputs/_vram_v15/train_cost.json" \
  "models/odace/outputs/odace_v15/train_cost.json"

echo "=== cleanup throwaway dirs $(date) ===" | tee -a "$ST"
rm -rf models/lsse/outputs/_vram_* models/dace/outputs/_vram_* models/esd/outputs/_vram_* \
       models/odace/outputs/_vram_* models/novel/outputs/fcf_p_v2_nudity 2>/dev/null

echo "=== aggregate + gallery $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/vram_aggregate.log
python compare/build_live_gallery.py 2>&1 | tee logs/vram_gallery.log

echo "=== VRAM-REMEASURE DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add models/fcf/train_cost.json eval/run_vram_remeasure.sh compare/build_live_gallery.py ; commit ; push" | tee -a "$ST"
touch models/fcf/VRAM_DONE
