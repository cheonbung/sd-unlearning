#!/usr/bin/env bash
# RPG-RT DPO deep-dive fill: iteratively DPO-fine-tuned adaptive attacker (the method's full training
# stage) vs the 3 OOD-collapse-fix winners missing from rpgrt_dpo.json -- odace_benign / odace_benign_n1
# (UNet swap) + lsse_geo_e2 (TE swap). Targets registered in eval/rpgrt_dpo_attack.py UNET_DIRS/TE_DIRS.
# Additively re-aggregate models/fcf/rpgrt_dpo.json (keeps existing 6). Mirrors run_rpgrt_followup.sh (B).
# HEAVY: ~1.5-2h/target (vicuna-7b 4bit + LoRA DPO, 4 iters). Single GPU, RPG-RT conda env, NO git.
# tmux: tmux new-session -d -s rpgrtdpo 'bash eval/run_rpgrt_dpo_fill.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RPG-RT
mkdir -p logs eval/outputs/rpgrt_dpo
RPG=/mnt/d/unlearning/RPG-RT
REPO=/mnt/d/unlearning/SD_unlearning

ST=models/fcf/RPGRT_DPO_FILL_STATUS
rm -f models/fcf/RPGRT_DPO_FILL_DONE
echo "=== RPG-RT DPO FILL START $(date) ===" | tee "$ST"

# --- smoke gate: tiny real DPO loop on one new target; abort whole job if rc!=0 ---
echo "=== DPO smoke (lsse_geo_e2 1iter 2x2) START $(date) ===" | tee -a "$ST"
( cd "$RPG" && python "$REPO"/eval/rpgrt_dpo_attack.py --target lsse_geo_e2 --iters 1 --n_train 2 \
    --n_eval 2 --group 2 --n_query 2 --out "$REPO"/eval/outputs/rpgrt_dpo/_smoke3 ) 2>&1 \
  | tee logs/rpgrt_dpo_fill_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== DPO smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "DPO SMOKE FAILED -> abort" | tee -a "$ST"; touch models/fcf/RPGRT_DPO_FILL_DONE; exit 1; fi

# --- full DPO attack on the 3 new targets ---
for T in odace_benign odace_benign_n1 lsse_geo_e2; do
  echo "=== DPO full $T START $(date) ===" | tee -a "$ST"
  ( cd "$RPG" && python "$REPO"/eval/rpgrt_dpo_attack.py --target "$T" --iters 4 --n_train 24 \
      --n_eval 20 --group 6 --n_query 10 --out "$REPO"/eval/outputs/rpgrt_dpo/"$T" ) 2>&1 \
    | tee "logs/rpgrt_dpo_$T.log"
  echo "=== DPO full $T END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
done

# --- re-aggregate ALL dpo summaries -> rpgrt_dpo.json (existing 6 + new 3, additive) ---
echo "=== aggregate rpgrt_dpo.json $(date) ===" | tee -a "$ST"
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

python compare/build_live_gallery.py 2>&1 | tee logs/rpgrt_dpo_gallery.log || true
echo "=== RPG-RT DPO FILL DONE $(date) ===" | tee -a "$ST"
touch models/fcf/RPGRT_DPO_FILL_DONE
