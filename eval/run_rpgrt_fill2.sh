#!/usr/bin/env bash
# RPG-RT GAP FILL 2: iter-0 adaptive red-team for the 3 OOD-collapse-fix winners missing from the
# roster -- odace_benign / odace_benign_n1 (UNet swap) + lsse_geo_e2 (TE swap), all SD1.4 base so the
# SD14 pipe is faithful. Targets registered in eval/rpgrt_attack_ours.py UNET_DIRS/TE_DIRS. Then
# additively re-aggregate models/fcf/rpgrt_redteam.json. (DPO deep-dive table is separate/heavier.)
# Mirrors run_rpgrt_fill.sh. Single GPU, RPG-RT conda env, external clone, NO git.
# tmux: tmux new-session -d -s rpgrtfill2 'bash eval/run_rpgrt_fill2.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RPG-RT
mkdir -p logs eval/outputs/rpgrt
RPG=/mnt/d/unlearning/RPG-RT
REPO=/mnt/d/unlearning/SD_unlearning

ST=models/fcf/RPGRT_FILL2_STATUS
rm -f models/fcf/RPGRT_FILL2_DONE
echo "=== RPG-RT FILL2 START $(date) ===" | tee "$ST"

# --- smoke gate: tiny real attack on one new target; abort whole job if rc!=0 ---
echo "=== smoke (lsse_geo_e2 2x2) START $(date) ===" | tee -a "$ST"
( cd "$RPG" && python "$REPO"/eval/rpgrt_attack_ours.py --target lsse_geo_e2 --n_prompts 2 --n_query 2 \
    --out output/_smoke_fill2 ) 2>&1 | tee logs/rpgrt_fill2_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "SMOKE FAILED -> abort" | tee -a "$ST"; touch models/fcf/RPGRT_FILL2_DONE; exit 1; fi

# --- full iter-0 attack on the 3 new targets ---
for T in odace_benign odace_benign_n1 lsse_geo_e2; do
  echo "=== attack $T START $(date) ===" | tee -a "$ST"
  ( cd "$RPG" && python "$REPO"/eval/rpgrt_attack_ours.py --target "$T" --n_prompts 20 --n_query 10 \
      --out output/rpgrt_ours ) 2>&1 | tee "logs/rpgrt_$T.log"
  echo "=== attack $T END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
done

# --- re-aggregate ALL summaries -> rpgrt_redteam.json (old + new, additive) ---
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

echo "=== RPG-RT FILL2 DONE $(date) ===" | tee -a "$ST"
touch models/fcf/RPGRT_FILL2_DONE
