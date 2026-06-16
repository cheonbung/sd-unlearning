#!/usr/bin/env bash
# Improvement re-run v2 = the FIXED versions of the two stages that OOMed on the 12GB GPU last pass,
# plus the A-mild Pareto retuning. NO git.
#   (B) output-grounding hybrid (eval/og_finetune.py) -- was CUBLAS_STATUS_EXECUTION_FAILED in the
#       fp32 two-graph backward AND had zero gradient signal. Now: bf16 autocast + sequential
#       forget/retain backward (one UNet graph at a time) + ESD negative-guidance target on an
#       informative timestep band. Variants: og_raw / og_sphot / og_lsse.
#   (A-mild) the every-epoch full-retain (sphot_retain) and balanced (lsse_bal) variants OVERSHOT
#       (recovered locality but lost efficacy). Milder: sphot_retain_mild (full-retain every 2nd
#       epoch) + lsse_bal_mild (beta 1.5 / clm_top_k 3 / ortho 0.10, between PLU and balanced).
#   (DPO) RPG-RT full adaptive attacker -- OOMed at iter0 eval generating a batch of n_query=10
#       sequences. Now sample_rewrites micro-batches to GEN_BATCH=4. Smoke uses n_query=6 (>4) so
#       the gate actually exercises the chunking path before the 6-target full run.
# tmux: tmux new-session -d -s improvev2 'bash eval/run_improve_v2.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
mkdir -p logs eval/outputs/og eval/outputs/rpgrt_dpo
REPO=/mnt/d/unlearning/SD_unlearning
RPG=/mnt/d/unlearning/RPG-RT
FORGET=models/lsse/data/prompts/nudity_explicit.txt
RETAIN=models/lsse/data/prompts/nudity_maintain.txt
SPHOT_BASE=models/novel/outputs/fcf_p_v2_nudity_spherical_ot/final
LSSE_BASE=models/lsse/outputs/sweep/plu_seed42/final

ST=models/fcf/IMPROVE_V2_STATUS
rm -f models/fcf/IMPROVE_V2_DONE
echo "=== IMPROVE V2 START $(date) ===" | tee "$ST"

conda activate lsse

# ============================================================ (B) smoke-gate the FIXED og_finetune
echo "=== (B) og smoke START $(date) ===" | tee -a "$ST"
python eval/og_finetune.py --base_te raw --forget_file "$FORGET" --retain_file "$RETAIN" \
  --out eval/outputs/og/_smoke_v2 --steps 4 --batch 2 --exp_name og_smoke_v2 2>&1 | tee logs/improve_v2_og_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== (B) og smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "(B) OG SMOKE STILL FAILS -> skip B" | tee -a "$ST"; SKIP_B=1; else SKIP_B=0; fi

# ============================================================ (B) output-grounding variants
if [ "$SKIP_B" -eq 0 ]; then
  echo "=== (B) og_raw train START $(date) ===" | tee -a "$ST"
  python eval/og_finetune.py --base_te raw --forget_file "$FORGET" --retain_file "$RETAIN" \
    --out eval/outputs/og/og_raw --steps 400 --exp_name og_raw 2>&1 | tee logs/improve_v2_og_raw.log
  echo "=== (B) og_raw END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
  if [ -d "$SPHOT_BASE" ]; then
    echo "=== (B) og_sphot train START $(date) ===" | tee -a "$ST"
    python eval/og_finetune.py --base_te "$SPHOT_BASE" --forget_file "$FORGET" --retain_file "$RETAIN" \
      --out eval/outputs/og/og_sphot --steps 400 --exp_name og_sphot 2>&1 | tee logs/improve_v2_og_sphot.log
    echo "=== (B) og_sphot END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
  else echo "(B) og_sphot SKIP: base missing $SPHOT_BASE" | tee -a "$ST"; fi
  if [ -d "$LSSE_BASE" ]; then
    echo "=== (B) og_lsse train START $(date) ===" | tee -a "$ST"
    python eval/og_finetune.py --base_te "$LSSE_BASE" --forget_file "$FORGET" --retain_file "$RETAIN" \
      --out eval/outputs/og/og_lsse --steps 400 --exp_name og_lsse 2>&1 | tee logs/improve_v2_og_lsse.log
    echo "=== (B) og_lsse END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
  else echo "(B) og_lsse SKIP: base missing $LSSE_BASE" | tee -a "$ST"; fi
fi

# ============================================================ (A-mild) Sph+OT half-cadence retain
if [ ! -f models/novel/outputs/ot_noise/nudity_learned.json ]; then
  echo "=== learn OT noise (missing) $(date) ===" | tee -a "$ST"
  ( cd models/novel && python scripts/learn_ot_noise.py \
      --explicit_file data/prompts/nudity_explicit.txt \
      --output outputs/ot_noise/nudity_learned.json ) 2>&1 | tee logs/improve_v2_otnoise.log
fi
echo "=== (A-mild) sphot_retain_mild train START $(date) ===" | tee -a "$ST"
( cd models/novel && python train.py --config configs/nudity_v2_retain_mild.yaml --manifold spherical \
    --ot_noise_file outputs/ot_noise/nudity_learned.json ) 2>&1 | tee logs/improve_v2_sphot_retain_mild.log
echo "=== (A-mild) sphot_retain_mild END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

# ============================================================ (A-mild) LSSE balanced-mild
echo "=== (A-mild) lsse_bal_mild train START $(date) ===" | tee -a "$ST"
( cd models/lsse && python train_lsse.py --config configs/nudity_lsse_balanced_mild.yaml ) 2>&1 | tee logs/improve_v2_lsse_bal_mild.log
echo "=== (A-mild) lsse_bal_mild END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

# ============================================================ evaluate (ASR + COCO) new variants
NEW="sphot_retain_mild,lsse_bal_mild,og_raw,og_sphot,og_lsse"
echo "=== eval ASR ($NEW) $(date) ===" | tee -a "$ST"
python eval/xeval.py --models "$NEW" 2>&1 | tee logs/improve_v2_xeval.log
echo "=== eval COCO ($NEW) $(date) ===" | tee -a "$ST"
python eval/eval_coco.py --models "$NEW" 2>&1 | tee logs/improve_v2_coco.log

# ============================================================ aggregate head-to-head v2 comparison
echo "=== aggregate improve_sweep_v2.json $(date) ===" | tee -a "$ST"
python - <<'PY' 2>&1 | tee -a "$ST"
import json, os
HERE = "eval/outputs"
# new variants + their TE-family baselines AND the overshot v1 variants, for a full A/B contrast
ROWS = ["raw_v14", "sph_ot", "sphot_retain", "sphot_retain_mild", "og_sphot", "og_raw",
        "lsse_plu", "lsse_bal", "lsse_bal_mild", "og_lsse", "odace_v3"]
def gp(label, fn, key):
    p = os.path.join(HERE, label, fn)
    if not os.path.exists(p): return None
    try: return json.load(open(p)).get(key)
    except Exception: return None
out = {"_doc": "LSSE/Sph+OT improvement sweep v2. (B)=output-grounding hybrid (og_*, fixed bf16+ESD "
       "neg-guidance). (A-mild)=sphot_retain_mild (full-retain every 2nd epoch), lsse_bal_mild "
       "(beta1.5/clm3/ortho0.10). Compare each to its TE base (sph_ot / lsse_plu), the OVERSHOT v1 "
       "(sphot_retain / lsse_bal), and the ODACE v3 ceiling. asr_mean lower=safer; coco_clip higher "
       "/ coco_fid,coco_lpips lower = better locality.", "models": {}}
for r in ROWS:
    out["models"][r] = {
        "asr_mean":  gp(r, "metrics.json", "asr_mean"),
        "coco_clip": gp(r, "coco_metrics.json", "coco_clip"),
        "coco_fid":  gp(r, "coco_metrics.json", "coco_fid"),
        "coco_lpips":gp(r, "coco_metrics.json", "coco_lpips"),
    }
json.dump(out, open("models/fcf/improve_sweep_v2.json", "w"), indent=2)
for r, m in out["models"].items():
    print(f"{r:20s} ASR={m['asr_mean']}  CLIP={m['coco_clip']}  FID={m['coco_fid']}  LPIPS={m['coco_lpips']}")
PY
python eval/aggregate_cost.py 2>&1 | tee logs/improve_v2_aggregate_cost.log || true

# ============================================================ (DPO) FIXED full adaptive attacker
conda deactivate
conda activate RPG-RT
echo "=== (DPO) smoke (n_query 6 > GEN_BATCH) START $(date) ===" | tee -a "$ST"
python "$REPO"/eval/rpgrt_dpo_attack.py --target raw --iters 1 --n_train 2 --n_eval 2 \
  --group 6 --n_query 6 --out "$REPO"/eval/outputs/rpgrt_dpo/_smoke_v2 2>&1 | tee logs/improve_v2_dpo_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== (DPO) smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then
  echo "(DPO) SMOKE STILL FAILS -> skip full DPO" | tee -a "$ST"
else
  for T in raw odace_v3 odace_mc_v2 esd_u sph_ot fcf_p_official; do
    echo "=== (DPO) full $T START $(date) ===" | tee -a "$ST"
    python "$REPO"/eval/rpgrt_dpo_attack.py --target "$T" --iters 4 --n_train 24 --n_eval 20 \
      --group 6 --n_query 10 --out "$REPO"/eval/outputs/rpgrt_dpo/"$T" 2>&1 | tee "logs/improve_v2_dpo_$T.log"
    echo "=== (DPO) full $T END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
  done
  echo "=== aggregate rpgrt_dpo.json $(date) ===" | tee -a "$ST"
  python - <<'PY' 2>&1 | tee -a "$ST"
import json, glob
out = {"_doc": "RPG-RT FULL adaptive red-team: vicuna-7b(4bit)+LoRA attacker iteratively DPO-fine-"
       "tuned on its own rollouts vs each target (4 iters). asr_*_iter0=frozen attacker; "
       "asr_*_best/final=after DPO. Higher=more vulnerable. iter0->best gap = how much an ADAPTIVE "
       "trained attacker erodes each defense (true worst-case robustness).", "models": {}}
for f in sorted(glob.glob("eval/outputs/rpgrt_dpo/*/dpo_summary_*.json")):
    s = json.load(open(f)); out["models"][s["target"]] = s
json.dump(out, open("models/fcf/rpgrt_dpo.json", "w"), indent=2)
for t, s in out["models"].items():
    print(f"{t:16s} iter0 q={s['asr_query_iter0']:5} -> best q={s['asr_query_best']:5} (iter {s['best_iter']})")
PY
fi

echo "=== IMPROVE V2 DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/ models/novel models/lsse models/fcf/improve_sweep_v2.json models/fcf/rpgrt_dpo.json" | tee -a "$ST"
touch models/fcf/IMPROVE_V2_DONE
