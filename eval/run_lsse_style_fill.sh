#!/usr/bin/env bash
# Fill the style-locality gap for 3 LSSE models (lsse_r2q_ab / lsse_r2q_a / lsse_capcnp_zero) that
# predate the style eval roster, so they currently show "--" for style_clip_text / LPIPS_f/u/d.
# Chain (single GPU, conda lsse): VanGogh gen (-> style_clip_text #11, style_img2raw) -> LPIPS_f
# (#forget) -> non-target other_styles gen -> LPIPS_u/LPIPS_d (#12/#13). raw_v14 refs already on disk
# (resumable skip). Merges into models/fcf/style_vangogh.json. tmux:
#   tmux new-session -d -s lssestyle 'bash eval/run_lsse_style_fill.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
M=lsse_r2q_ab,lsse_r2q_a,lsse_capcnp_zero

ST=models/fcf/LSSE_STYLE_FILL_STATUS
rm -f models/fcf/LSSE_STYLE_FILL_DONE
echo "lsse_style_fill start $(date -u +%FT%TZ) models=$M" | tee "$ST"

# --- smoke gate: tiny real VanGogh gen on one target; abort if rc!=0 ---
echo "[gate] smoke" | tee -a "$ST"
python models/fcf/eval_style_vangogh.py --models lsse_r2q_ab --limit 2 >/dev/null 2>&1 \
  || { echo "GATE FAIL -> abort" | tee -a "$ST"; touch models/fcf/LSSE_STYLE_FILL_DONE; exit 1; }
echo "[gate] ok" | tee -a "$ST"

# --- 1) VanGogh gen + style_clip_text (#11) + style_img2raw for the 3 models ---
echo "[1] vangogh gen start $(date -u +%FT%TZ)" | tee -a "$ST"
python models/fcf/eval_style_vangogh.py --models "$M" 2>&1 | tee logs/lssefill_vangogh.log
echo "[1] rc=${PIPESTATUS[0]} done $(date -u +%FT%TZ)" | tee -a "$ST"

# --- 2) LPIPS_f over all on-disk VanGogh dirs (auto-discovers the 3 new ones) ---
echo "[2] lpips_f start $(date -u +%FT%TZ)" | tee -a "$ST"
python eval/lpips_style.py 2>&1 | tee logs/lssefill_lpips_f.log
echo "[2] rc=${PIPESTATUS[0]} done $(date -u +%FT%TZ)" | tee -a "$ST"

# --- 3) non-target other_styles -> LPIPS_u/LPIPS_d for the 3 models (#12/#13) ---
echo "[3] lpips_u start $(date -u +%FT%TZ)" | tee -a "$ST"
python eval/lpips_nontarget_style.py --models "$M" 2>&1 | tee logs/lssefill_lpips_u.log
echo "[3] rc=${PIPESTATUS[0]} done $(date -u +%FT%TZ)" | tee -a "$ST"

echo "lsse_style_fill end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/LSSE_STYLE_FILL_DONE
