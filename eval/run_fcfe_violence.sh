#!/usr/bin/env bash
# Complete FCF-E for VIOLENCE: the upstream ships an "experience" vector (mean(T_ori(explicit) -
# T_ori(noise)), shape [1,77,768]) only for nudity, so features_forgetting_E.py had no violence input.
# Here we (1) reverse-engineer the violence experience from data/train/violence.csv (prompt_f vs
# prompt_n), validating the formula by reconstructing the shipped nudity experience (cosine sim),
# (2) re-train the explicit encoder to a fresh path (FCF-P earlier overwrote the shared one),
# (3) run features_forgetting_E.py, (4) convert -> official_fcf_e_violence/final, (5) Q16 eval.
# tmux: tmux new-session -d -s fcfeviol 'bash eval/run_fcfe_violence.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
UP=models/fcf/upstream
ST=models/fcf/FCFEVIOL_STATUS
rm -f models/fcf/FCFEVIOL_DONE
echo "=== FCF-E VIOLENCE START $(date) ===" | tee "$ST"

# --- smoke gate: upstream + violence data present ---
for f in "$UP/concept_forgetting_train.py" "$UP/features_forgetting_E.py" \
         "$UP/data/train/violence.csv" "$UP/experience.pth"; do
  [ -f "$f" ] || { echo "MISSING $f -> abort" | tee -a "$ST"; touch models/fcf/FCFEVIOL_DONE; exit 1; }
done

# --- (1) generate violence experience + validate formula against shipped nudity experience ---
echo "=== gen violence experience (+nudity formula check) $(date) ===" | tee -a "$ST"
python - <<'PY' 2>&1 | tee logs/fcfeviol_experience.log
import torch, pandas as pd
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
dev = "cuda"
tok = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
te = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(dev).eval()
def emb(prompts):
    t = tok(list(prompts), padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        return te(t.input_ids)[0]  # [N,77,768]
def experience(csv):
    df = pd.read_csv(csv)
    return (emb(df["prompt_f"]) - emb(df["prompt_n"])).mean(dim=0, keepdim=True).cpu()  # [1,77,768]
# validate: reconstruct nudity experience, compare to shipped experience.pth
nud = experience("models/fcf/upstream/data/train/nudity.csv")
ship = torch.load("models/fcf/upstream/experience.pth", map_location="cpu")
cos = F.cosine_similarity(nud.flatten(1), ship.flatten(1)).item()
print(f"[validate] nudity experience cosine-sim vs shipped = {cos:.4f} (>0.9 => formula confirmed)")
# generate violence experience
viol = experience("models/fcf/upstream/data/train/violence.csv")
torch.save(viol, "models/fcf/experience_violence.pth")
print(f"[gen] violence experience saved shape={tuple(viol.shape)}")
PY
echo "=== experience rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"

# --- (2) re-train explicit encoder to a fresh path (FCF-P overwrote the shared one) ---
echo "=== train explicit (E-branch) $(date) ===" | tee -a "$ST"
( cd "$UP" && python concept_forgetting_train.py --input_prompts data/train/violence.csv \
    --save_path ../official_violence_explicit_E.pt ) 2>&1 | tee logs/fcfeviol_explicit.log
echo "=== explicit rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"

# --- (3) FCF-E empirical (overwrites official_violence_explicit_E.pt with the FCF-E result) ---
echo "=== FCF-E empirical $(date) ===" | tee -a "$ST"
( cd "$UP" && python features_forgetting_E.py --model_path ../official_violence_explicit_E.pt \
    --experience_path ../experience_violence.pth ) 2>&1 | tee logs/fcfeviol_e.log
echo "=== FCF-E rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"

# --- (4) convert -> HF dir ---
echo "=== convert -> HF dir $(date) ===" | tee -a "$ST"
python - <<'PY' 2>&1 | tee logs/fcfeviol_convert.log
import os, torch
from transformers import CLIPTextModel
pt = "models/fcf/official_violence_explicit_E.pt"; out = "models/fcf/official_fcf_e_violence/final"
te = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
sd = torch.load(pt, map_location="cpu"); sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
miss, unexp = te.load_state_dict(sd, strict=False)
print("missing=", len(miss), "unexpected=", len(unexp))
os.makedirs(out, exist_ok=True); te.save_pretrained(out)
print("saved ->", out, "config:", os.path.exists(out + "/config.json"))
PY
echo "=== convert rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"

# --- (5) Q16 eval + gallery ---
if [ -f models/fcf/official_fcf_e_violence/final/config.json ]; then
  echo "=== eval_violence_q16 (fcf_e_violence) $(date) ===" | tee -a "$ST"
  python models/fcf/eval_violence_q16.py --models fcf_e_violence 2>&1 | tee logs/fcfeviol_eval.log
  echo "=== eval rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"
  python compare/build_live_gallery.py 2>&1 | tee logs/fcfeviol_gallery.log || true
else
  echo "NO HF DIR -> eval skipped; inspect logs/fcfeviol_convert.log" | tee -a "$ST"
fi
echo "=== FCF-E VIOLENCE DONE $(date) ===" | tee -a "$ST"
touch models/fcf/FCFEVIOL_DONE
