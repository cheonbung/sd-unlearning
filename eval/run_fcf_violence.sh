#!/usr/bin/env bash
# Train the FCF-P / FCF-E VIOLENCE-forgotten text encoders (authors' upstream code) and eval them,
# so Table 5 (violence, paper-aligned) can show real FCF-P/E rows instead of N/A.
# Pipeline follows REPRODUCE.md 2-3: fetch upstream -> train explicit -> FCF-P/E projection/empirical
# -> convert each trained CLIPTextModel state_dict to an HF dir the harness loads (official_fcf_{p,e}_violence/final)
# -> eval_violence_q16.py. UNCERTAIN steps (upstream arg names, .pt->HF conversion) are smoke-gated:
# if upstream/scripts/data are missing the job aborts before burning GPU.
# tmux: tmux new-session -d -s fcfviol 'bash eval/run_fcf_violence.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/FCFVIOL_STATUS
rm -f models/fcf/FCFVIOL_DONE
echo "=== FCF VIOLENCE START $(date) ===" | tee "$ST"

UP=models/fcf/upstream
OUTP=models/fcf/official_fcf_p_violence/final
OUTE=models/fcf/official_fcf_e_violence/final

# --- fetch upstream (authors' code + violence.csv) if missing ---
if [ ! -f "$UP/concept_forgetting_train.py" ]; then
  echo "=== fetch_external (upstream) $(date) ===" | tee -a "$ST"
  PYTHON=$(which python) bash scripts/fetch_external.sh 2>&1 | tee logs/fcfviol_fetch.log
fi

# --- smoke gate: required upstream scripts + violence training data present ---
echo "=== smoke: verify upstream files $(date) ===" | tee -a "$ST"
MISS=0
for f in "$UP/concept_forgetting_train.py" "$UP/features_forgetting_P.py" \
         "$UP/features_forgetting_E.py" "$UP/data/train/violence.csv"; do
  if [ ! -f "$f" ]; then echo "MISSING: $f" | tee -a "$ST"; MISS=1; fi
done
if [ "$MISS" -ne 0 ]; then
  echo "SMOKE FAILED (upstream/data missing) -> abort; inspect scripts/fetch_external.sh" | tee -a "$ST"
  touch models/fcf/FCFVIOL_DONE; exit 1
fi
echo "=== smoke OK $(date) ===" | tee -a "$ST"

# --- train (explicit forgetting -> FCF-P projection + FCF-E empirical), authors' hyperparams ---
echo "=== train explicit $(date) ===" | tee -a "$ST"
( cd "$UP" && python concept_forgetting_train.py --input_prompts data/train/violence.csv \
    --save_path ../official_violence_explicit.pt ) 2>&1 | tee logs/fcfviol_explicit.log
echo "=== explicit rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"

echo "=== FCF-P projection $(date) ===" | tee -a "$ST"
( cd "$UP" && python features_forgetting_P.py --model_path ../official_violence_explicit.pt ) \
    2>&1 | tee logs/fcfviol_p.log
echo "=== FCF-P rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"

echo "=== FCF-E empirical $(date) ===" | tee -a "$ST"
( cd "$UP" && python features_forgetting_E.py --model_path ../official_violence_explicit.pt \
    --experience_path ../experience_violence.pth ) 2>&1 | tee logs/fcfviol_e.log
echo "=== FCF-E rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"

# --- convert trained CLIPTextModel state_dicts to HF dirs the harness loads (te_swap) ---
echo "=== convert .pt -> HF dirs $(date) ===" | tee -a "$ST"
python - <<'PY' 2>&1 | tee logs/fcfviol_convert.log
import glob, os, torch
from transformers import CLIPTextModel
BASE = "CompVis/stable-diffusion-v1-4"
# features_forgetting_{P,E}.py save the cleaned encoder next to the upstream dir; find the newest .pt
# for each. Adjust globs here if upstream names differ (see logs/fcfviol_{p,e}.log for the real path).
def newest(patterns):
    files = [f for p in patterns for f in glob.glob(p)]
    return max(files, key=os.path.getmtime) if files else None
for tag, out, pats in [
    ("P", "models/fcf/official_fcf_p_violence/final",
       ["models/fcf/upstream/*_P*.pt", "models/fcf/*fcf_p*violence*.pt", "models/fcf/official_violence*P*.pt"]),
    ("E", "models/fcf/official_fcf_e_violence/final",
       ["models/fcf/upstream/*_E*.pt", "models/fcf/*fcf_e*violence*.pt", "models/fcf/official_violence*E*.pt"]),
]:
    pt = newest(pats)
    if not pt:
        print(f"[{tag}] NO .pt matched -> inspect logs/fcfviol_{tag.lower()}.log for the real save path"); continue
    print(f"[{tag}] loading state from {pt}")
    te = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder")
    sd = torch.load(pt, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = te.load_state_dict(sd, strict=False)
    print(f"[{tag}] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    os.makedirs(out, exist_ok=True)
    te.save_pretrained(out)
    print(f"[{tag}] saved -> {out}")
PY
echo "=== convert rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"

# --- eval both violence-trained FCF encoders (Q16 per-attack) ---
if [ -f "$OUTP/config.json" ] || [ -f "$OUTE/config.json" ]; then
  echo "=== eval_violence_q16 (fcf_p/e_violence) $(date) ===" | tee -a "$ST"
  python models/fcf/eval_violence_q16.py --models fcf_p_violence,fcf_e_violence 2>&1 | tee logs/fcfviol_eval.log
  echo "=== eval rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"
  python compare/build_live_gallery.py 2>&1 | tee logs/fcfviol_gallery.log || true
else
  echo "CONVERSION PRODUCED NO HF DIR -> eval skipped; inspect logs/fcfviol_convert.log" | tee -a "$ST"
fi

echo "=== FCF VIOLENCE DONE $(date) ===" | tee -a "$ST"
touch models/fcf/FCFVIOL_DONE
