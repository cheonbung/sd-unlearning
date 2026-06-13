#!/usr/bin/env bash
# RPG-RT red-teaming attack (NeurIPS25, arXiv 2505.21074) against OUR unlearned models.
# Uses the RPG-RT dataset + vicuna-7b LLM prompt-rewrite attack (4-bit for 12GB GPU) via the
# self-contained scripts/rpgrt_attack_ours.py in the external clone D:\unlearning\RPG-RT.
# Smoke (downloads vicuna ~13GB, validates pipeline) -> attack raw baseline + odace_mc_v2 ->
# copy summaries into our repo + aggregate. Runs in the RPG-RT conda env. NO git.
# tmux: tmux new-session -d -s rpgrt 'bash eval/run_rpgrt.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RPG-RT
mkdir -p logs eval/outputs/rpgrt
RPG=/mnt/d/unlearning/RPG-RT

ST=models/odace/RPGRT_STATUS
rm -f models/odace/RPGRT_DONE
echo "=== RPG-RT START $(date) ===" | tee "$ST"

echo "=== smoke (vicuna dl + 2x2) START $(date) ===" | tee -a "$ST"
( cd "$RPG" && python scripts/rpgrt_attack_ours.py --target odace_mc_v2 --n_prompts 2 --n_query 2 \
    --out output/_smoke ) 2>&1 | tee logs/rpgrt_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "SMOKE FAILED -> abort" | tee -a "$ST"; touch models/odace/RPGRT_DONE; exit 1; fi

for T in raw odace_mc_v2; do
  echo "=== attack $T START $(date) ===" | tee -a "$ST"
  ( cd "$RPG" && python scripts/rpgrt_attack_ours.py --target $T --n_prompts 20 --n_query 10 \
      --out output/rpgrt_ours ) 2>&1 | tee logs/rpgrt_$T.log
  echo "=== attack $T END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"
done

echo "=== aggregate $(date) ===" | tee -a "$ST"
cp "$RPG"/output/rpgrt_ours/summary_*.json eval/outputs/rpgrt/ 2>/dev/null
python - <<'PY' 2>&1 | tee -a "$ST"
import json, glob, os
out = {"_doc": "RPG-RT (NeurIPS25 arXiv 2505.21074) adaptive nudity red-team. asr_prompt=% of I2P "
       "nudity prompts with >=1 NSFW bypass over N vicuna rewrites; asr_query=% of all queries NSFW. "
       "vicuna-7b 4-bit base attack (iter0, no DPO). Higher=more vulnerable (worse defense).",
       "models": {}}
for f in sorted(glob.glob("eval/outputs/rpgrt/summary_*.json")):
    s = json.load(open(f)); out["models"][s["target"]] = s
json.dump(out, open("models/fcf/rpgrt_redteam.json", "w"), indent=2)
print("RPGRT aggregate:", json.dumps(out["models"]))
PY

echo "=== RPG-RT DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/run_rpgrt.sh models/fcf/rpgrt_redteam.json ; git commit ; git push" | tee -a "$ST"
touch models/odace/RPGRT_DONE
