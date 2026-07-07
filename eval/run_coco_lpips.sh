#!/usr/bin/env bash
# Fill the COCO-LPIPS column for ALL roster models. coco5k.json only had raw_v14/fcf_p/fcf_e (the
# paper Table-3 N=5000 fidelity run); COCO images for the other ~20 models were never retained, so
# this generates a fresh, INTERNALLY-CONSISTENT COCO-LPIPS pass for every model at a moderate N.
# LPIPS is a per-image mean -> robust to small N (unlike FID), so N=100 gives a stable coco_lpips
# while staying ~1.8h instead of the ~90h an N=5000 x 23-model run would take. We read ONLY coco_lpips
# from coco5k_lpips.json (FID/CLIP at N=100 are NOT used). fcf_p/fcf_e CAN be included here (inference
# eval on their checkpoints; the no-parent-edit constraint only blocks measuring their TRAINING cost).
#
# SMOKE-GATED (smoke-test-before-bg-jobs rule): tiny n=8 run first; abort if coco_lpips not captured.
# Queues behind the VRAM job (shared GPU). tmux: tmux new-session -d -s cocolpips 'bash eval/run_coco_lpips.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/COCOLPIPS_STATUS
rm -f models/fcf/COCOLPIPS_DONE
echo "=== COCO-LPIPS START $(date) ===" | tee "$ST"

# queue behind the VRAM re-measure job (single GPU)
if tmux has-session -t vram 2>/dev/null && [ ! -f models/fcf/VRAM_DONE ]; then
  echo "=== waiting for vram (models/fcf/VRAM_DONE) $(date) ===" | tee -a "$ST"
  while tmux has-session -t vram 2>/dev/null && [ ! -f models/fcf/VRAM_DONE ]; do sleep 120; done
  echo "=== vram cleared $(date) ===" | tee -a "$ST"
fi

# raw_v14 FIRST = LPIPS reference (eval_coco_fid5k uses raw_v14_coco5k<tag> as the ref).
ALL="raw_v14,raw_v15,sd21base,sld_medium,sld_strong,sld_max,safeclip,safe_neg,esd_u,fcf_p_official,fcf_e_official,vanilla_lsse,lsse_plu,lsse_plu_w2,sph_ot,dace_v2,dace_plu,odace_v3,odace_v15,odace_v2,lsse_mc_nvg,odace_mc,lsse_mc_nvg_v2"

# ---- SMOKE: n=8, comma --models (eval_coco_fid5k parses comma). Abort if coco_lpips not captured. ----
echo "=== SMOKE (n=8) $(date) ===" | tee -a "$ST"
python models/fcf/eval_coco_fid5k.py --models raw_v14,fcf_p_official --n 8 --tag _lpips_smoke 2>&1 | tee logs/cocolpips_smoke.log
SMOKE=$(python - <<'PY'
import json
try:
    d = json.load(open("models/fcf/coco5k_lpips_smoke.json"))["models"]
    print(d.get("fcf_p_official", {}).get("coco_lpips"))
except Exception:
    print("None")
PY
)
echo "=== SMOKE coco_lpips=$SMOKE ===" | tee -a "$ST"
if [ "$SMOKE" = "None" ] || [ -z "$SMOKE" ]; then
  echo "SMOKE FAILED: coco_lpips not captured -> ABORT (no full run)" | tee -a "$ST"
  touch models/fcf/COCOLPIPS_DONE; exit 1
fi

# ---- full pass: N=100, all 23 models -> coco5k_lpips.json ----
echo "=== full COCO-LPIPS (n=100, 23 models) $(date) ===" | tee -a "$ST"
python models/fcf/eval_coco_fid5k.py --models "$ALL" --n 100 --tag _lpips 2>&1 | tee logs/cocolpips_full.log
RC=${PIPESTATUS[0]}
echo "=== full END $(date) (rc=$RC) ===" | tee -a "$ST"

echo "=== rebuild gallery $(date) ===" | tee -a "$ST"
python compare/build_live_gallery.py 2>&1 | tee logs/cocolpips_gallery.log

echo "=== COCO-LPIPS DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add models/fcf/coco5k_lpips.json eval/run_coco_lpips.sh compare/build_live_gallery.py ; commit ; push" | tee -a "$ST"
touch models/fcf/COCOLPIPS_DONE
