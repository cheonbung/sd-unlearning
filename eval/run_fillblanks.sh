#!/usr/bin/env bash
# Fill all blank gallery cells for the OOD-fix models (odace_benign, odace_benign_n1, lsse_geo_e2):
# COCO FID/CLIP (eval_coco.py), COCO-LPIPS (eval_coco_fid5k _lpips), Violence-Q16, VanGogh-style,
# style_lpips. Also fills raw_v14's missing COCO-LPIPS (the _lpips ref). All scripts merge
# incrementally (subset --models preserves other models) and are resumable. (Distinct from
# run_fill_cells.sh, which fills train-cost + MC full-set ASR for a different model/metric set.)
# tmux: tmux new-session -d -s fillblanks 'bash eval/run_fillblanks.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/FILLBLANKS_STATUS
rm -f models/fcf/FILLBLANKS_DONE
echo "fillblanks start $(date -u +%FT%TZ)" | tee "$ST"

M=odace_benign,odace_benign_n1,lsse_geo_e2

# ---- smoke gate: isolated tiny run (separate _smoke dir/json -> no corruption of real outputs) ----
python models/fcf/eval_coco_fid5k.py --tag _smoke --n 4 --models odace_benign >/dev/null 2>&1 \
  || { echo "GATE FAIL (smoke) -> abort" | tee -a "$ST"; touch models/fcf/FILLBLANKS_DONE; exit 1; }
echo "[gate] ok $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- COCO utility FID/CLIP (N=300, eval/outputs/<key>/coco_metrics.json) ----
python eval/eval_coco.py --models "$M" 2>&1 | tee logs/fb_coco.log
echo "[coco_fid_clip] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- COCO-LPIPS (N=100; include raw_v14 -> builds _lpips ref + fills raw's blank) ----
python models/fcf/eval_coco_fid5k.py --tag _lpips --n 100 --models "raw_v14,$M" 2>&1 | tee logs/fb_lpips.log
echo "[coco_lpips] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- Violence Q16 (I2P-viol 757 + RaB-viol 269) ----
python models/fcf/eval_violence_q16.py --models "$M" 2>&1 | tee logs/fb_viol.log
echo "[violence] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- VanGogh art-style retain ----
python models/fcf/eval_style_vangogh.py --models "$M" 2>&1 | tee logs/fb_style.log
echo "[style] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

# ---- style_lpips_f (over generated VG imgs; no generation) ----
python eval/lpips_style.py 2>&1 | tee logs/fb_slpips.log
echo "[style_lpips] rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$ST"

python compare/build_live_gallery.py 2>&1 | tee logs/fb_gallery.log || true
echo "fillblanks end $(date -u +%FT%TZ)" | tee -a "$ST"
touch models/fcf/FILLBLANKS_DONE
