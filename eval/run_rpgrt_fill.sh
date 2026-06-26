#!/usr/bin/env bash
# RPG-RT GAP FILL: attack the 4 SD1.4 TE-swap models missing from the iter-0 red-team roster
# (our LSSE flagship/variants + FCF-E), then re-aggregate models/fcf/rpgrt_redteam.json additively.
# Mirrors run_rpgrt_followup.sh (A1). Single GPU, RPG-RT conda env, external clone, NO git.
# Targets were registered in eval/rpgrt_attack_ours.py TE_DIRS. odace_v15 is intentionally EXCLUDED
# (SD1.5 base would mismatch the SD1.4 pipe; odace_v3 already represents ODACE).
# tmux: tmux new-session -d -s rpgrtfill 'bash eval/run_rpgrt_fill.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RPG-RT
mkdir -p logs eval/outputs/rpgrt
RPG=/mnt/d/unlearning/RPG-RT
REPO=/mnt/d/unlearning/SD_unlearning

ST=models/fcf/RPGRT_FILL_STATUS
rm -f models/fcf/RPGRT_FILL_DONE
echo "=== RPG-RT FILL START $(date) ===" | tee "$ST"

# --- smoke gate: tiny real attack on one new target; abort whole job if rc!=0 ---
echo "=== smoke (lsse_r2q_ab 2x2) START $(date) ===" | tee -a "$ST"
( cd "$RPG" && python "$REPO"/eval/rpgrt_attack_ours.py --target lsse_r2q_ab --n_prompts 2 --n_query 2 \
    --out output/_smoke_fill ) 2>&1 | tee logs/rpgrt_fill_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "SMOKE FAILED -> abort" | tee -a "$ST"; touch models/fcf/RPGRT_FILL_DONE; exit 1; fi

# --- full iter-0 attack on the 4 missing targets ---
for T in fcf_e_official lsse_capcnp_zero lsse_r2q_a lsse_r2q_ab; do
  echo "=== attack $T START $(date) ===" | tee -a "$ST"
  ( cd "$RPG" && python "$REPO"/eval/rpgrt_attack_ours.py --target "$T" --n_prompts 20 --n_query 10 \
      --out output/rpgrt_ours ) 2>&1 | tee "logs/rpgrt_$T.log"
  echo "=== attack $T END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
done

# --- re-aggregate ALL summaries -> rpgrt_redteam.json (old 8/9 + new 4, additive) ---
echo "=== re-aggregate rpgrt_redteam.json $(date) ===" | tee -a "$ST"
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

echo "=== RPG-RT FILL DONE $(date) ===" | tee -a "$ST"
touch models/fcf/RPGRT_FILL_DONE
