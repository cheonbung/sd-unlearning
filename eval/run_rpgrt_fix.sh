#!/usr/bin/env bash
# Re-run the two RPG-RT follow-up steps that failed the first time, now that their bugs are fixed:
#   (1) safeclip attack  - safeclip_loader.py now aligns the text_model. key prefix (4.44/4.48).
#   (2) DPO full attack   - rpgrt_dpo_attack.py now toggles gradient checkpointing in dpo_step
#                           (fixes the 12GB CUDA OOM); smoke-gated so a residual OOM just skips.
# Queues behind the improvement sweep (shared GPU). Merges into models/fcf/rpgrt_redteam.json and
# writes models/fcf/rpgrt_dpo.json. NO git.
# tmux: tmux new-session -d -s rpgrtfix 'bash eval/run_rpgrt_fix.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
mkdir -p logs eval/outputs/rpgrt eval/outputs/rpgrt_dpo
RPG=/mnt/d/unlearning/RPG-RT
REPO=/mnt/d/unlearning/SD_unlearning

ST=models/fcf/RPGRT_FIX_STATUS
rm -f models/fcf/RPGRT_FIX_DONE
echo "=== RPGRT FIX START $(date) ===" | tee "$ST"

# ---- queue behind the improvement sweep ----
if tmux has-session -t improvesweep 2>/dev/null && [ ! -f models/fcf/IMPROVE_SWEEP_DONE ]; then
  echo "=== waiting for improvesweep (IMPROVE_SWEEP_DONE) $(date) ===" | tee -a "$ST"
  while tmux has-session -t improvesweep 2>/dev/null && [ ! -f models/fcf/IMPROVE_SWEEP_DONE ]; do sleep 120; done
  echo "=== improvesweep cleared $(date) ===" | tee -a "$ST"
fi

conda activate RPG-RT

# ============================================================ (1) safeclip attack (loader fixed)
echo "=== (1) safeclip attack START $(date) ===" | tee -a "$ST"
( cd "$RPG" && python "$REPO"/eval/rpgrt_attack_ours.py --target safeclip --n_prompts 20 --n_query 10 \
    --out output/rpgrt_ours ) 2>&1 | tee logs/rpgrt_safeclip.log
echo "=== (1) safeclip attack END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
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

# ============================================================ (2) DPO full attack (OOM fixed)
echo "=== (2) DPO smoke START $(date) ===" | tee -a "$ST"
python "$REPO"/eval/rpgrt_dpo_attack.py --target raw --iters 1 --n_train 2 --n_eval 2 \
  --group 2 --n_query 2 --out "$REPO"/eval/outputs/rpgrt_dpo/_smoke 2>&1 | tee logs/rpgrt_dpo_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== (2) DPO smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then
  echo "(2) DPO SMOKE STILL FAILING -> skip full DPO" | tee -a "$ST"
else
  for T in raw odace_v3 odace_mc_v2 esd_u sph_ot fcf_p_official; do
    echo "=== (2) DPO full $T START $(date) ===" | tee -a "$ST"
    python "$REPO"/eval/rpgrt_dpo_attack.py --target "$T" --iters 4 --n_train 24 --n_eval 20 \
      --group 6 --n_query 10 --out "$REPO"/eval/outputs/rpgrt_dpo/"$T" 2>&1 | tee "logs/rpgrt_dpo_$T.log"
    echo "=== (2) DPO full $T END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
  done
  echo "=== (2) aggregate rpgrt_dpo.json $(date) ===" | tee -a "$ST"
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

echo "=== RPGRT FIX DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/ models/safeclip/safeclip_loader.py models/fcf/rpgrt_redteam.json models/fcf/rpgrt_dpo.json ; git commit ; git push" | tee -a "$ST"
touch models/fcf/RPGRT_FIX_DONE
