#!/usr/bin/env bash
# Fill esd_u peak VRAM. ESD-u (models/esd/train_esd.py) is UNet diffusion fine-tuning and has no
# CostMeter, so the Jun-18 VRAM run left peak_vram_gb=null (gpu_hours=1.326 is already recorded).
# Same approach as run_costfill.sh: poll nvidia-smi device-used during a THROWAWAY re-train
# (--output_dir outputs/_vram_esd, NOT the evaluated outputs/esd_u ckpt) and patch the peak into
# the canonical models/esd/outputs/esd_u/train_cost.json (gpu_hours preserved), vram_source="nvidia-smi".
# The exact train cmd was already proven rc=0 in the Jun-18 run_vram_remeasure run; the nvidia-smi
# polling mechanism was proven this session (sph_ot/FCF captured non-zero peaks).
# QUEUES behind the costfill job (single GPU). tmux: tmux new-session -d -s esdvram 'bash eval/run_costfill_esd.sh'
set -u
REPO=/mnt/d/unlearning/SD_unlearning
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/ESDVRAM_STATUS
rm -f models/fcf/ESDVRAM_DONE
echo "=== ESDVRAM START $(date) ===" | tee "$ST"

# queue behind costfill (single GPU)
if tmux has-session -t costfill 2>/dev/null && [ ! -f models/fcf/COSTFILL_DONE ]; then
  echo "=== waiting for costfill (models/fcf/COSTFILL_DONE) $(date) ===" | tee -a "$ST"
  while tmux has-session -t costfill 2>/dev/null && [ ! -f models/fcf/COSTFILL_DONE ]; do sleep 60; done
  echo "=== costfill cleared $(date) ===" | tee -a "$ST"
fi

vram_poll() {  # $1=label $2=cmd_string $3=workdir
  local label="$1" cmd="$2" wd="${3:-$REPO}"
  local pf="$REPO/logs/vrampoll_${label}.txt"; : > "$pf"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$pf" 2>/dev/null; sleep 0.4; done ) &
  local POLL=$!
  ( cd "$wd" && eval "$cmd" ) 2>&1 | tee "$REPO/logs/cost_${label}.log"
  LAST_RC=${PIPESTATUS[0]}
  kill "$POLL" 2>/dev/null; wait "$POLL" 2>/dev/null
  LAST_PEAK_MB=$(sort -n "$pf" | tail -1)
  [ -z "$LAST_PEAK_MB" ] && LAST_PEAK_MB=0
  echo "=== vram_poll $label: rc=$LAST_RC peak=${LAST_PEAK_MB}MB ===" | tee -a "$ST"
}

patch_cost() {  # $1=MODEL $2=DEST $3=VRAM_MB [$4=VRAM_SOURCE]   (gpu_hours preserved)
  MODEL="$1" DEST="$2" VRAM_MB="$3" VRAM_SOURCE="${4:-nvidia-smi}" python - <<'PY'
import json, os, datetime, subprocess
model=os.environ["MODEL"]; dst=os.environ["DEST"]
vram_gb=round(float(os.environ["VRAM_MB"])/1024.0, 2); src=os.environ["VRAM_SOURCE"]
try: gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"]).decode().strip().splitlines()[0]
except Exception: gpu="unknown"
d=json.load(open(dst)) if os.path.exists(dst) else {"model":model,"training_free":False,"gpu":gpu,"gpu_count":1}
d["model"]=d.get("model",model); d["training_free"]=False
d["peak_vram_gb"]=vram_gb; d["vram_source"]=src
d.setdefault("gpu",gpu); d.setdefault("timestamp",datetime.date.today().isoformat()); d["measured_by"]="run_costfill_esd"
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(d, open(dst,"w"), indent=2)
print("[patch_cost] %s vram=%.2fG src=%s (gpu_hours kept) -> %s" % (model, vram_gb, src, dst))
PY
}

echo "=== ESD-u VRAM (throwaway UNet re-train, ~80min) $(date) ===" | tee -a "$ST"
vram_poll "esd" \
  "python train_esd.py --config configs/nudity_esd_u.yaml --output_dir outputs/_vram_esd" \
  "$REPO/models/esd"
if [ "${LAST_RC:-1}" -eq 0 ] && [ "${LAST_PEAK_MB:-0}" -gt 0 ]; then
  patch_cost "esd_u" "$REPO/models/esd/outputs/esd_u/train_cost.json" "$LAST_PEAK_MB" "nvidia-smi" | tee -a "$ST"
else
  echo "ESD-u FAILED (rc=${LAST_RC:-?}, peak=${LAST_PEAK_MB:-0}) -> VRAM stays null" | tee -a "$ST"
fi
rm -rf "$REPO/models/esd/outputs/_vram_esd" 2>/dev/null

echo "=== aggregate + gallery $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/esdvram_aggregate.log
python compare/build_live_gallery.py 2>&1 | tee logs/esdvram_gallery.log

echo "=== ESDVRAM DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add models/fcf/train_cost.json compare/build_live_gallery.py eval/run_costfill_esd.sh ; commit ; push" | tee -a "$ST"
touch models/fcf/ESDVRAM_DONE
