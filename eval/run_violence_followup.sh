#!/usr/bin/env bash
# Post-VIOLQ16FULL follow-up queue (see auto-memory violence-eval-paper-alignment):
#   1. violence coherence probe (collapse check for push-away rows; no regen, CLIP-only)
#   2. violence_q16.json legacy split (rows of models no longer shown anywhere in the gallery)
#   3. sph_ot_violence RETRAIN with the paper-aligned implicit_groups (person/body/man/woman fix)
#   4. eval upgrade to 3-attack for sph_ot_violence + the transfer-table models
#   5. gallery rebuild
# FCF-P/E violence NOT queued: FCF_upstream needs its own `ldm` conda env (does not exist yet).
# tmux: tmux new-session -d -s violfollow \
#   'until [ -f models/fcf/VIOLQ16FULL_DONE ]; do sleep 60; done; bash eval/run_violence_followup.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/VIOLFOLLOW_STATUS
: > "$ST"
echo "VIOLFOLLOW START $(date)" | tee -a "$ST"

# ---- smoke gate: tiny coherence probe (validates env + GPU + generated images on disk) ----
echo "[smoke] START $(date +%F_%T)" | tee -a "$ST"
python models/fcf/eval_coherence_violence.py --models raw_v14 --limit 2 2>&1 | tee logs/violfollow_smoke.log
rc=${PIPESTATUS[0]}
echo "[smoke] rc=$rc END $(date +%F_%T)" | tee -a "$ST"
if [ "$rc" -ne 0 ]; then
  echo "SMOKE FAILED rc=$rc -> ABORT" | tee -a "$ST"
  touch models/fcf/VIOLFOLLOW_DONE
  exit 1
fi

run() {  # $1=label  $2=shell-string (non-fatal, tee, rc logged)
  local label=$1; shift
  bash -c "$1" 2>&1 | tee "logs/violfollow_${label}.log"
  local rc=${PIPESTATUS[0]}
  echo "[$label] rc=$rc END $(date +%F_%T)" | tee -a "$ST"
  return 0
}

# ---- 1. full violence coherence probe (7 models x ring/uda/i2p-cap, CLIP-only) ----
run coherence "python models/fcf/eval_coherence_violence.py"

# ---- 2. legacy split: move rows of models displayed nowhere in the gallery ----
echo "[legacy_split] START $(date +%F_%T)" | tee -a "$ST"
python - <<'PY' 2>&1 | tee logs/violfollow_split.log
import json
from pathlib import Path
P = Path("models/fcf/violence_q16.json")
L = Path("models/fcf/violence_q16_legacy.json")
PURGE = {  # discarded-protocol rows for models removed from every gallery table/section
    "lsse_plu", "lsse_plu_w2", "vanilla_lsse", "dace_v2", "dace_plu", "odace_v2", "odace_v15",
    "lsse_mc_nvg", "lsse_mc_nvg_v2", "odace_mc", "odace_mc_v2", "sph_ot_mc",
    "lsse_capcnp", "lsse_r2q_violence_aw",
}
d = json.loads(P.read_text())
legacy = json.loads(L.read_text()) if L.exists() else {
    "_doc": "Legacy violence Q16 rows (pre paper-alignment 2-attack protocol) for models no longer "
            "displayed in the gallery. Moved out of violence_q16.json 2026-07-06.", "models": {}}
moved = []
for k in sorted(PURGE):
    if k in d.get("models", {}):
        legacy["models"][k] = d["models"].pop(k)
        moved.append(k)
L.write_text(json.dumps(legacy, indent=2))
P.write_text(json.dumps(d, indent=2))
print("moved to legacy:", moved)
print("remaining:", sorted(d["models"].keys()))
PY
echo "[legacy_split] rc=${PIPESTATUS[0]} END $(date +%F_%T)" | tee -a "$ST"

run gallery1 "python compare/build_live_gallery.py"

# ---- 3. sph_ot_violence retrain (paper implicit_groups fix; output overwrites
#         models/novel/outputs/fcf_p_v2_violence -> REGISTRY key sph_ot_violence) ----
run sphot_train "cd models/novel && python train.py --config configs/violence_v2.yaml --manifold spherical --ot_noise_file outputs/ot_noise/violence_learned.json"

# ---- 4. eval upgrade: retrained sph_ot_violence full 3-attack + upgrade the transfer-table
#         nudity-trained models from the old 2-attack rows to the 3-attack protocol (resumable:
#         existing i2p/ring images are reused; only UDA generates fresh). NOTE sph_ot_violence
#         images must be REGENERATED after retrain -> wipe its stale output dirs first. ----
run wipe_sphot_imgs "rm -rf eval/outputs/sph_ot_violence_violence"
run violence_eval "python models/fcf/eval_violence_q16.py --models sph_ot_violence,fcf_p_official,fcf_e_official,esd_u,odace_benign_n1,lsse_r2q_violence"

# ---- 4b. coherence probe for the retrained sph_ot_violence ----
run coherence2 "python models/fcf/eval_coherence_violence.py --models sph_ot_violence"

run gallery2 "python compare/build_live_gallery.py"

echo "VIOLFOLLOW ALL DONE $(date)" | tee -a "$ST"
touch models/fcf/VIOLFOLLOW_DONE
