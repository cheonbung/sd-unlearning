#!/usr/bin/env bash
# LSSE / Sph+OT improvement sweep = plan (A) footprint/retain tuning + (B) output-grounding hybrid.
# Queues behind the RPG-RT follow-up (shares the single GPU), trains every variant, evaluates ASR +
# COCO on one harness, and aggregates a head-to-head comparison vs the baselines. NO git.
#   A-Sph+OT : full-retain anchor      -> sphot_retain  (models/novel, nudity_v2_retain.yaml)
#   A-LSSE   : balanced footprint      -> lsse_bal      (models/lsse, nudity_lsse_balanced.yaml)
#   B        : output-grounding hybrid -> og_raw / og_sphot / og_lsse (eval/og_finetune.py)
# tmux: tmux new-session -d -s improvesweep 'bash eval/run_improve_sweep.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs eval/outputs/og
REPO=/mnt/d/unlearning/SD_unlearning
FORGET=models/lsse/data/prompts/nudity_explicit.txt
RETAIN=models/lsse/data/prompts/nudity_maintain.txt
SPHOT_BASE=models/novel/outputs/fcf_p_v2_nudity_spherical_ot/final
LSSE_BASE=models/lsse/outputs/sweep/plu_seed42/final

ST=models/fcf/IMPROVE_SWEEP_STATUS
rm -f models/fcf/IMPROVE_SWEEP_DONE
echo "=== IMPROVE SWEEP START $(date) ===" | tee "$ST"

# ---- queue behind the RPG-RT follow-up GPU campaign ----
if tmux has-session -t rpgrtfu 2>/dev/null && [ ! -f models/fcf/RPGRT_FOLLOWUP_DONE ]; then
  echo "=== waiting for rpgrtfu (RPGRT_FOLLOWUP_DONE) $(date) ===" | tee -a "$ST"
  while tmux has-session -t rpgrtfu 2>/dev/null && [ ! -f models/fcf/RPGRT_FOLLOWUP_DONE ]; do sleep 120; done
  echo "=== rpgrtfu cleared $(date) ===" | tee -a "$ST"
fi

# ============================================================ (B) smoke-gate og_finetune.py
echo "=== (B) og smoke START $(date) ===" | tee -a "$ST"
python eval/og_finetune.py --base_te raw --forget_file "$FORGET" --retain_file "$RETAIN" \
  --out eval/outputs/og/_smoke --steps 4 --batch 2 --exp_name og_smoke 2>&1 | tee logs/improve_og_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== (B) og smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "(B) OG SMOKE FAILED -> abort sweep" | tee -a "$ST"; touch models/fcf/IMPROVE_SWEEP_DONE; exit 1; fi

# ============================================================ (A) Sph+OT full-retain
if [ ! -f models/novel/outputs/ot_noise/nudity_learned.json ]; then
  echo "=== learn OT noise (missing) $(date) ===" | tee -a "$ST"
  ( cd models/novel && python scripts/learn_ot_noise.py \
      --explicit_file data/prompts/nudity_explicit.txt \
      --output outputs/ot_noise/nudity_learned.json ) 2>&1 | tee logs/improve_otnoise.log
fi
echo "=== (A) sphot_retain train START $(date) ===" | tee -a "$ST"
( cd models/novel && python train.py --config configs/nudity_v2_retain.yaml --manifold spherical \
    --ot_noise_file outputs/ot_noise/nudity_learned.json ) 2>&1 | tee logs/improve_sphot_retain.log
echo "=== (A) sphot_retain END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

# ============================================================ (A) LSSE balanced
echo "=== (A) lsse_bal train START $(date) ===" | tee -a "$ST"
( cd models/lsse && python train_lsse.py --config configs/nudity_lsse_balanced.yaml ) 2>&1 | tee logs/improve_lsse_bal.log
echo "=== (A) lsse_bal END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

# ============================================================ (B) output-grounding variants
echo "=== (B) og_raw train START $(date) ===" | tee -a "$ST"
python eval/og_finetune.py --base_te raw --forget_file "$FORGET" --retain_file "$RETAIN" \
  --out eval/outputs/og/og_raw --steps 400 --exp_name og_raw 2>&1 | tee logs/improve_og_raw.log
echo "=== (B) og_raw END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

if [ -d "$SPHOT_BASE" ]; then
  echo "=== (B) og_sphot train START $(date) ===" | tee -a "$ST"
  python eval/og_finetune.py --base_te "$SPHOT_BASE" --forget_file "$FORGET" --retain_file "$RETAIN" \
    --out eval/outputs/og/og_sphot --steps 400 --exp_name og_sphot 2>&1 | tee logs/improve_og_sphot.log
  echo "=== (B) og_sphot END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
else echo "(B) og_sphot SKIP: base missing $SPHOT_BASE" | tee -a "$ST"; fi

if [ -d "$LSSE_BASE" ]; then
  echo "=== (B) og_lsse train START $(date) ===" | tee -a "$ST"
  python eval/og_finetune.py --base_te "$LSSE_BASE" --forget_file "$FORGET" --retain_file "$RETAIN" \
    --out eval/outputs/og/og_lsse --steps 400 --exp_name og_lsse 2>&1 | tee logs/improve_og_lsse.log
  echo "=== (B) og_lsse END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
else echo "(B) og_lsse SKIP: base missing $LSSE_BASE" | tee -a "$ST"; fi

# ============================================================ evaluate (ASR + COCO) the new variants
NEW="sphot_retain,lsse_bal,og_raw,og_sphot,og_lsse"
echo "=== eval ASR ($NEW) $(date) ===" | tee -a "$ST"
python eval/xeval.py --models "$NEW" 2>&1 | tee logs/improve_xeval.log
echo "=== eval COCO ($NEW) $(date) ===" | tee -a "$ST"
python eval/eval_coco.py --models "$NEW" 2>&1 | tee logs/improve_coco.log

# ============================================================ aggregate head-to-head comparison
echo "=== aggregate improve_sweep.json $(date) ===" | tee -a "$ST"
python - <<'PY' 2>&1 | tee -a "$ST"
import json, os
HERE = "eval/outputs"
# new variants + their TE-family baselines for a direct A/B contrast
ROWS = ["raw_v14", "sph_ot", "sphot_retain", "og_sphot", "og_raw",
        "lsse_plu", "lsse_bal", "og_lsse", "odace_v3"]
def gp(label, fn, key):
    p = os.path.join(HERE, label, fn)
    if not os.path.exists(p): return None
    try: return json.load(open(p)).get(key)
    except Exception: return None
out = {"_doc": "LSSE/Sph+OT improvement sweep. A=footprint/retain (sphot_retain, lsse_bal); "
       "B=output-grounding hybrid (og_*). asr_mean=nudity ASR (lower=safer); coco_clip/fid/lpips "
       "=locality (higher CLIP / lower FID,LPIPS = better retain). Compare each variant to its "
       "TE-family base (sph_ot / lsse_plu) and the ODACE v3 ceiling.", "models": {}}
for r in ROWS:
    out["models"][r] = {
        "asr_mean":  gp(r, "metrics.json", "asr_mean"),
        "coco_clip": gp(r, "coco_metrics.json", "coco_clip"),
        "coco_fid":  gp(r, "coco_metrics.json", "coco_fid"),
        "coco_lpips":gp(r, "coco_metrics.json", "coco_lpips"),
    }
json.dump(out, open("models/fcf/improve_sweep.json", "w"), indent=2)
for r, m in out["models"].items():
    print(f"{r:16s} ASR={m['asr_mean']}  CLIP={m['coco_clip']}  FID={m['coco_fid']}  LPIPS={m['coco_lpips']}")
PY

python eval/aggregate_cost.py 2>&1 | tee logs/improve_aggregate_cost.log || true
echo "=== IMPROVE SWEEP DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/og_finetune.py eval/run_improve_sweep.sh models/novel models/lsse eval/xeval.py models/fcf/improve_sweep.json ; git commit ; git push" | tee -a "$ST"
touch models/fcf/IMPROVE_SWEEP_DONE
