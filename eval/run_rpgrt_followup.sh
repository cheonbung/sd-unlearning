#!/usr/bin/env bash
# RPG-RT FOLLOW-UP campaign = goal (A) + (B), single GPU, no git.
#   (A1) Fill the iter-0 red-team table gaps: attack sld_max + safeclip with the now-fixed local
#        SLD / Safe-CLIP loaders, re-aggregate models/fcf/rpgrt_redteam.json (9 models complete).
#   (A2) Re-measure single-concept ODACE training cost faithfully via the nudity_odace recipe
#        (1500 steps, same trainer/arch as the archived odace_v3) -> train_cost.json for the gallery.
#   (B)  RPG-RT FULL: DPO-fine-tuned attacker (eval/rpgrt_dpo_attack.py) pushed to its limit on the
#        core targets; smoke-gated, then aggregated to models/fcf/rpgrt_dpo.json.
# tmux: tmux new-session -d -s rpgrtfu 'bash eval/run_rpgrt_followup.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
mkdir -p logs eval/outputs/rpgrt eval/outputs/rpgrt_dpo
RPG=/mnt/d/unlearning/RPG-RT
REPO=/mnt/d/unlearning/SD_unlearning

ST=models/fcf/RPGRT_FOLLOWUP_STATUS
rm -f models/fcf/RPGRT_FOLLOWUP_DONE
echo "=== RPG-RT FOLLOW-UP START $(date) ===" | tee "$ST"

# ============================================================ (A1) sld_max + safeclip iter-0 attack
conda activate RPG-RT
for T in sld_max safeclip; do
  echo "=== (A1) attack $T START $(date) ===" | tee -a "$ST"
  ( cd "$RPG" && python "$REPO"/eval/rpgrt_attack_ours.py --target "$T" --n_prompts 20 --n_query 10 \
      --out output/rpgrt_ours ) 2>&1 | tee "logs/rpgrt_$T.log"
  echo "=== (A1) attack $T END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
done
echo "=== (A1) re-aggregate rpgrt_redteam.json $(date) ===" | tee -a "$ST"
cp "$RPG"/output/rpgrt_ours/summary_*.json eval/outputs/rpgrt/ 2>/dev/null
python - <<'PY' 2>&1 | tee -a "$ST"
import json, glob
out = {"_doc": "RPG-RT (NeurIPS25 arXiv 2505.21074) adaptive nudity red-team. asr_prompt=% of I2P "
       "nudity prompts with >=1 NSFW bypass over N vicuna rewrites; asr_query=% of all queries NSFW. "
       "vicuna-7b 4-bit base attack (iter0, no DPO). Higher=more vulnerable (worse defense).",
       "models": {}}
for f in sorted(glob.glob("eval/outputs/rpgrt/summary_*.json")):
    s = json.load(open(f)); out["models"][s["target"]] = s
json.dump(out, open("models/fcf/rpgrt_redteam.json", "w"), indent=2)
print("RPGRT redteam now has:", sorted(out["models"]))
PY

# ============================================================ (B) RPG-RT FULL: DPO attacker
echo "=== (B) DPO smoke START $(date) ===" | tee -a "$ST"
python "$REPO"/eval/rpgrt_dpo_attack.py --target raw --iters 1 --n_train 2 --n_eval 2 \
  --group 2 --n_query 2 --out "$REPO"/eval/outputs/rpgrt_dpo/_smoke 2>&1 | tee logs/rpgrt_dpo_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== (B) DPO smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then
  echo "(B) DPO SMOKE FAILED -> skip full DPO, continue to (A2)" | tee -a "$ST"
else
  for T in raw odace_v3 odace_mc_v2 esd_u sph_ot fcf_p_official; do
    echo "=== (B) DPO full $T START $(date) ===" | tee -a "$ST"
    python "$REPO"/eval/rpgrt_dpo_attack.py --target "$T" --iters 4 --n_train 24 --n_eval 20 \
      --group 6 --n_query 10 --out "$REPO"/eval/outputs/rpgrt_dpo/"$T" 2>&1 | tee "logs/rpgrt_dpo_$T.log"
    echo "=== (B) DPO full $T END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
  done
  echo "=== (B) aggregate rpgrt_dpo.json $(date) ===" | tee -a "$ST"
  python - <<'PY' 2>&1 | tee -a "$ST"
import json, glob
out = {"_doc": "RPG-RT FULL adaptive red-team: vicuna-7b(4bit)+LoRA attacker iteratively DPO-fine-"
       "tuned on its own rollouts vs each target (4 iters). asr_*_iter0=frozen attacker; "
       "asr_*_best/final=after DPO. Higher=more vulnerable. The gap iter0->best = how much an "
       "ADAPTIVE trained attacker erodes each defense (true worst-case robustness).",
       "models": {}}
for f in sorted(glob.glob("eval/outputs/rpgrt_dpo/*/dpo_summary_*.json")):
    s = json.load(open(f)); out["models"][s["target"]] = s
json.dump(out, open("models/fcf/rpgrt_dpo.json", "w"), indent=2)
for t, s in out["models"].items():
    print(f"{t:16s} iter0 q={s['asr_query_iter0']:5} -> best q={s['asr_query_best']:5} (iter {s['best_iter']})")
PY
fi

# ============================================================ (A2) single-concept ODACE train cost
echo "=== (A2) odace single-concept cost re-measure $(date) ===" | tee -a "$ST"
conda activate lsse
T0=$(date +%s)
( cd models/odace && python train_odace.py --config configs/nudity_odace.yaml \
    --output_dir outputs/_costrun_single ) 2>&1 | tee logs/retraincost_odace_single.log
RC=${PIPESTATUS[0]}; T1=$(date +%s); WALL=$((T1-T0))
echo "=== (A2) odace single END (rc=$RC, wall=${WALL}s) ===" | tee -a "$ST"
if [ "$RC" -eq 0 ]; then
  MODEL="odace_single" STEPS=1500 WALL="$WALL" python - <<'PY' 2>&1 | tee -a "$ST"
import json, os, subprocess, datetime
try:
    gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"]).decode().strip().splitlines()[0]
except Exception:
    gpu = "unknown"
w = float(os.environ["WALL"])
d = {"model": os.environ["MODEL"], "training_free": False, "gpu": gpu, "gpu_count": 1,
     "trainable_params_M": None, "steps": int(os.environ["STEPS"]), "wall_seconds": round(w, 1),
     "gpu_hours": round(w/3600, 3), "peak_vram_gb": None, "torch": None,
     "timestamp": datetime.date.today().isoformat(), "status": "ok",
     "note": "single-concept ODACE (nudity_odace recipe, 1500 steps); same trainer/arch as archived odace_v3",
     "measured_by": "retrain_cost_runner"}
# odace_v3 (headline single) and odace_v2 (outputs/odace_nudity) share this recipe class
for dest in ("models/odace/outputs/odace_v3/train_cost.json",
             "models/odace/outputs/odace_nudity/train_cost.json"):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(d, open(dest, "w"), indent=2)
    print("cost ->", dest, d["gpu_hours"], "gpu_h")
PY
else
  echo "(A2) odace single retrain FAILED -> cost not written" | tee -a "$ST"
fi
echo "=== (A2) cleanup throwaway $(date) ===" | tee -a "$ST"
rm -rf models/odace/outputs/_costrun_single 2>/dev/null
python eval/aggregate_cost.py 2>&1 | tee logs/followup_aggregate_cost.log
python compare/build_live_gallery.py 2>&1 | tee logs/followup_gallery.log || true

echo "=== RPG-RT FOLLOW-UP DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/rpgrt_attack_ours.py eval/rpgrt_dpo_attack.py eval/run_rpgrt_followup.sh models/fcf/rpgrt_redteam.json models/fcf/rpgrt_dpo.json models/odace/outputs/odace_v3/train_cost.json ; git commit ; git push" | tee -a "$ST"
touch models/fcf/RPGRT_FOLLOWUP_DONE
