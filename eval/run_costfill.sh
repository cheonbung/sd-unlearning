#!/usr/bin/env bash
# Fill the last empty cells WITHOUT modifying any parent trainer (read+run only):
#   Part 1  SLERP-OT (sph_ot) peak VRAM  -- novel/train.py has no CostMeter, so we poll
#           nvidia-smi device-used during a THROWAWAY re-train (--output_dir _cost_sphot, NOT
#           the evaluated _spherical_ot ckpt) and patch the peak into the canonical
#           models/novel/outputs/fcf_p_v2_nudity_spherical_ot/train_cost.json.
#   Part 2  FCF-P / FCF-E  GPU-h + peak VRAM  -- the authors' FCF_upstream trainers are
#           text-encoder-only (CLIPTextModel, no UNet) and have no CostMeter. We re-run them
#           UNMODIFIED in env lsse with external wall-timing + nvidia-smi VRAM polling:
#             base = concept_forgetting_train.py  (shared)  -> *.pt + regenerates experience.pth
#             FCF-P = base + features_forgetting_P.py   FCF-E = base + features_forgetting_E.py
#           gpu_h = base+branch wall; vram = max(base,branch). Patched into
#           models/fcf/official_fcf_{p,e}/train_cost.json with vram_source="nvidia-smi" (†).
#           Side-effects guarded: experience.pth backed up/restored; P/E overwrite --model_path
#           in place so each runs on its OWN copy of base.pt; all throwaway under FCF_upstream/_cost.
#   Part 3  violence-50 (Q16, --limit 50) for the 3 MC models the Jun-12 LEGACY50 run predates:
#           lsse_mc_nvg / odace_mc / lsse_mc_nvg_v2 -> merged into violence_q16_smoke.json.
#   Part 4  aggregate_cost + rebuild gallery.
# SMOKE-GATED (smoke-test-before-bg-jobs): a 3-prompt FCF base + 1-model 2-prompt viol50 run
# first; abort the matching part if its artifact/VRAM is not captured.
# tmux: tmux new-session -d -s costfill 'bash eval/run_costfill.sh'
set -u
REPO=/mnt/d/unlearning/SD_unlearning
FCFUP=/mnt/d/unlearning/FCF_upstream
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/COSTFILL_STATUS
rm -f models/fcf/COSTFILL_DONE
echo "=== COSTFILL START $(date) ===" | tee "$ST"

# --- nvidia-smi device-used peak around a command string. Sets LAST_RC/LAST_WALL/LAST_PEAK_MB. ---
vram_poll() {  # $1=label  $2=cmd_string  $3=workdir(optional)
  local label="$1" cmd="$2" wd="${3:-$REPO}"
  local pf="$REPO/logs/vrampoll_${label}.txt"; : > "$pf"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$pf" 2>/dev/null; sleep 0.4; done ) &
  local POLL=$!
  local T0=$(date +%s)
  ( cd "$wd" && eval "$cmd" ) 2>&1 | tee "$REPO/logs/cost_${label}.log"
  LAST_RC=${PIPESTATUS[0]}
  local T1=$(date +%s)
  kill "$POLL" 2>/dev/null; wait "$POLL" 2>/dev/null
  LAST_WALL=$((T1-T0))
  LAST_PEAK_MB=$(sort -n "$pf" | tail -1)
  [ -z "$LAST_PEAK_MB" ] && LAST_PEAK_MB=0
  echo "=== vram_poll $label: rc=$LAST_RC wall=${LAST_WALL}s peak=${LAST_PEAK_MB}MB ===" | tee -a "$ST"
}

# --- merge peak_vram_gb (+optional gpu_hours) + vram_source into a run-dir train_cost.json ---
patch_cost() {  # $1=MODEL $2=DEST $3=VRAM_MB [$4=GPU_HOURS] [$5=VRAM_SOURCE]
  MODEL="$1" DEST="$2" VRAM_MB="$3" GPU_HOURS="${4:-}" VRAM_SOURCE="${5:-nvidia-smi}" python - <<'PY'
import json, os, datetime, subprocess
model=os.environ["MODEL"]; dst=os.environ["DEST"]
vram_gb=round(float(os.environ["VRAM_MB"])/1024.0, 2)
gh=os.environ.get("GPU_HOURS","").strip()
src=os.environ["VRAM_SOURCE"]
try: gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"]).decode().strip().splitlines()[0]
except Exception: gpu="unknown"
d=json.load(open(dst)) if os.path.exists(dst) else {"model":model,"training_free":False,"gpu":gpu,"gpu_count":1}
d["model"]=d.get("model",model); d["training_free"]=False
d["peak_vram_gb"]=vram_gb; d["vram_source"]=src
if gh: d["gpu_hours"]=round(float(gh),3); d["wall_seconds"]=round(float(gh)*3600,1)
d.setdefault("gpu",gpu); d.setdefault("timestamp",datetime.date.today().isoformat())
d["measured_by"]="run_costfill"
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(d, open(dst,"w"), indent=2)
print("[patch_cost] %s vram=%.2fG src=%s gpu_h=%s -> %s" % (model, vram_gb, src, gh or "(kept)", dst))
PY
}

# ============================================================ Part 1: SLERP-OT VRAM
echo "=== Part1 SLERP-OT VRAM (throwaway re-train) $(date) ===" | tee -a "$ST"
# train.py has NO --output_dir; it writes to cfg["output_dir"]="outputs/fcf_p_v2_nudity"
# (a THROWAWAY -- the EVALUATED ckpt is the separate outputs/fcf_p_v2_nudity_spherical_ot, untouched).
vram_poll "sphot" \
  "python train.py --config configs/nudity_v2.yaml --manifold spherical --ot_noise_file outputs/ot_noise/nudity_learned.json" \
  "$REPO/models/novel"
if [ "${LAST_RC:-1}" -eq 0 ] && [ "${LAST_PEAK_MB:-0}" -gt 0 ]; then
  patch_cost "sph_ot" "$REPO/models/novel/outputs/fcf_p_v2_nudity_spherical_ot/train_cost.json" "$LAST_PEAK_MB" "" "nvidia-smi" | tee -a "$ST"
else
  echo "Part1 sph_ot FAILED (rc=${LAST_RC:-?}, peak=${LAST_PEAK_MB:-0}) -> VRAM stays null" | tee -a "$ST"
fi
rm -rf "$REPO/models/novel/outputs/fcf_p_v2_nudity" 2>/dev/null

# ============================================================ Part 2: FCF-P / FCF-E
echo "=== Part2 FCF base+P+E (text-encoder, env lsse) $(date) ===" | tee -a "$ST"
EXP="$FCFUP/experience.pth"
EXP_BAK="$FCFUP/experience.pth.bak_costfill"
[ -f "$EXP" ] && cp -f "$EXP" "$EXP_BAK" && echo "backed up experience.pth" | tee -a "$ST"
mkdir -p "$FCFUP/_cost"

# ---- SMOKE: 3-prompt base run (60 epochs x 3 prompts = seconds). Abort FCF part if it fails. ----
echo "=== Part2 SMOKE (3-prompt base) $(date) ===" | tee -a "$ST"
head -n 4 "$FCFUP/data/train/nudity.csv" > "$FCFUP/_cost/smoke_nudity.csv"
vram_poll "fcf_smoke" \
  "python concept_forgetting_train.py --input_prompts _cost/smoke_nudity.csv --save_path _cost/smoke_base" \
  "$FCFUP"
if [ "${LAST_RC:-1}" -ne 0 ] || [ ! -f "$FCFUP/_cost/smoke_base.pt" ] || [ "${LAST_PEAK_MB:-0}" -le 0 ]; then
  echo "Part2 SMOKE FAILED (rc=${LAST_RC:-?}, peak=${LAST_PEAK_MB:-0}) -> SKIP FCF" | tee -a "$ST"
  FCF_OK=0
else
  echo "=== Part2 SMOKE OK -> full FCF $(date) ===" | tee -a "$ST"
  FCF_OK=1
fi

if [ "${FCF_OK:-0}" -eq 1 ]; then
  # base (shared) on the REAL nudity prompts; regenerates experience.pth in FCFUP
  vram_poll "fcf_base" \
    "python concept_forgetting_train.py --input_prompts data/train/nudity.csv --save_path _cost/base" "$FCFUP"
  BASE_RC=${LAST_RC}; BASE_WALL=${LAST_WALL}; BASE_PEAK=${LAST_PEAK_MB}
  if [ "$BASE_RC" -eq 0 ] && [ -f "$FCFUP/_cost/base.pt" ]; then
    # FCF-P branch (own copy; features_forgetting_P overwrites --model_path in place)
    cp -f "$FCFUP/_cost/base.pt" "$FCFUP/_cost/p.pt"
    vram_poll "fcf_p" "python features_forgetting_P.py --model_path _cost/p.pt" "$FCFUP"
    if [ "${LAST_RC}" -eq 0 ]; then
      P_GH=$(python -c "print(($BASE_WALL+$LAST_WALL)/3600.0)")
      P_VRAM=$(python -c "print(max($BASE_PEAK,$LAST_PEAK_MB))")
      patch_cost "fcf_p_official" "$REPO/models/fcf/official_fcf_p/train_cost.json" "$P_VRAM" "$P_GH" "nvidia-smi" | tee -a "$ST"
    else echo "FCF-P branch FAILED (rc=${LAST_RC})" | tee -a "$ST"; fi
    # FCF-E branch (own copy; needs experience.pth regenerated by base)
    cp -f "$FCFUP/_cost/base.pt" "$FCFUP/_cost/e.pt"
    vram_poll "fcf_e" "python features_forgetting_E.py --model_path _cost/e.pt --experience_path experience.pth" "$FCFUP"
    if [ "${LAST_RC}" -eq 0 ]; then
      E_GH=$(python -c "print(($BASE_WALL+$LAST_WALL)/3600.0)")
      E_VRAM=$(python -c "print(max($BASE_PEAK,$LAST_PEAK_MB))")
      patch_cost "fcf_e_official" "$REPO/models/fcf/official_fcf_e/train_cost.json" "$E_VRAM" "$E_GH" "nvidia-smi" | tee -a "$ST"
    else echo "FCF-E branch FAILED (rc=${LAST_RC})" | tee -a "$ST"; fi
  else
    echo "FCF base FAILED (rc=$BASE_RC) -> FCF cells stay '—'" | tee -a "$ST"
  fi
fi
# restore the original experience.pth + clean throwaway
[ -f "$EXP_BAK" ] && mv -f "$EXP_BAK" "$EXP" && echo "restored experience.pth" | tee -a "$ST"
rm -rf "$FCFUP/_cost" 2>/dev/null

# ============================================================ Part 3: MC violence-50
echo "=== Part3 MC violence-50 SMOKE (1 model, --limit 2) $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models lsse_mc_nvg --limit 2 2>&1 | tee logs/cost_viol50_smoke.log
SMK_RC=${PIPESTATUS[0]}
if [ "$SMK_RC" -eq 0 ]; then
  echo "=== Part3 full viol50 (3 MC models, --limit 50) $(date) ===" | tee -a "$ST"
  python models/fcf/eval_violence_q16.py --models lsse_mc_nvg,odace_mc,lsse_mc_nvg_v2 --limit 50 \
    2>&1 | tee logs/cost_viol50_full.log
  echo "=== Part3 viol50 END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
else
  echo "Part3 viol50 SMOKE FAILED (rc=$SMK_RC) -> MC viol50 cells stay '—'" | tee -a "$ST"
fi

# ============================================================ Part 4: aggregate + gallery
echo "=== Part4 aggregate + gallery $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/cost_aggregate.log
python compare/build_live_gallery.py 2>&1 | tee logs/cost_gallery.log

echo "=== COSTFILL DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add models/fcf/train_cost.json models/fcf/violence_q16_smoke.json compare/build_live_gallery.py eval/run_costfill.sh ; commit ; push" | tee -a "$ST"
touch models/fcf/COSTFILL_DONE
