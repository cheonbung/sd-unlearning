#!/usr/bin/env bash
# Fill the paper-aligned metrics (#9-#14) for the 3 OOD-fix models (odace_benign, odace_benign_n1,
# lsse_geo_e2), which the main-table fill (run_fillblanks.sh) did not cover:
#   #9/#10 nude_counts  -> eval_nude_counts.py  (NudeNet relabel over EXISTING _fs PNGs, no gen)
#   #14   coco_kid      -> eval_coco_kid.py     (FID-SD/KID over EXISTING coco PNGs vs raw, no gen)
#   #12/#13 style u/d   -> lpips_nontarget_style.py (small non-VanGogh gen, 51 prompts/model)
# (DPO deep-dive table is separate/heavier and intentionally NOT here.)
# tmux: tmux new-session -d -s paperfill 'bash eval/run_paperfill.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/PAPERFILL_STATUS
rm -f models/fcf/PAPERFILL_DONE
echo "paperfill start $(date -u +%FT%TZ)" | tee "$ST"

M=odace_benign,odace_benign_n1,lsse_geo_e2

# --- smoke gate: tiny real nude-count relabel on one model (no generation) ---
python models/fcf/eval_nude_counts.py --models odace_benign --limit 5 >/dev/null 2>&1 \
  || { echo "GATE FAIL -> abort" | tee -a "$ST"; touch models/fcf/PAPERFILL_DONE; exit 1; }
echo "[gate] ok $(date -u +%FT%TZ)" | tee -a "$ST"

# --- #9/#10 nude counts (relabel existing fullset images) ---
python models/fcf/eval_nude_counts.py --models "$M" 2>&1 | tee logs/pf_nude.log
echo "[nude_counts] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

# --- #14 FID-SD / KID (vs raw, over existing COCO images) ---
python models/fcf/eval_coco_kid.py --models "$M" 2>&1 | tee logs/pf_kid.log
echo "[coco_kid] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

# --- #12/#13 non-target style LPIPS_u/LPIPS_d (small non-VanGogh gen) ---
python eval/lpips_nontarget_style.py --models "$M" 2>&1 | tee logs/pf_styleud.log
echo "[style_ud] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python compare/build_live_gallery.py 2>&1 | tee logs/pf_gallery.log || true
echo "paperfill end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/PAPERFILL_DONE
