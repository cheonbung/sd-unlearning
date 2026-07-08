#!/usr/bin/env bash
# Q16 per-attack eval of the newly-trained FCF-P violence encoder (official_fcf_p_violence/final),
# to fill Table 5's FCF-P (violence) row. FCF-E stays N/A (its experience file was never generated).
# tmux: tmux new-session -d -s fcfpvioleval 'bash eval/run_fcfp_viol_eval.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/FCFPVIOLEVAL_STATUS
rm -f models/fcf/FCFPVIOLEVAL_DONE
echo "=== FCFP VIOL EVAL START $(date) ===" | tee "$ST"

# --- smoke gate: confirm fcf_p_violence loads + generates (3 prompts/attack, isolated *_smoke.json) ---
echo "=== smoke (--limit 3) START $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models fcf_p_violence --limit 3 2>&1 | tee logs/fcfpvioleval_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "SMOKE FAILED -> abort" | tee -a "$ST"; touch models/fcf/FCFPVIOLEVAL_DONE; exit 1; fi

# --- full Q16 eval ---
echo "=== full eval START $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models fcf_p_violence 2>&1 | tee logs/fcfpvioleval_full.log
echo "=== full eval END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

python compare/build_live_gallery.py 2>&1 | tee logs/fcfpvioleval_gallery.log || true
echo "=== FCFP VIOL EVAL DONE $(date) ===" | tee -a "$ST"
touch models/fcf/FCFPVIOLEVAL_DONE
