#!/usr/bin/env bash
# Phase 1/2/3 new-metric runner. Chains the three eval scripts, one GPU job at a time.
#   P1 (#9/#10) nude detection COUNTS over existing *_fs images        -> models/fcf/nude_counts.json
#   P3 (#14)   COCO gen for 3 key models + KID/FID-SD vs raw SD        -> models/fcf/coco_kid.json
#   P2 (#12/#13) non-target style LPIPS_u/d (generates other_styles)   -> style_vangogh.json
# Convention (CLAUDE.md S5): set -u, smoke gate first, each part non-fatal + tee to logs/, append
# *_STATUS, touch *_DONE at the end. Watch with: cat models/fcf/PHASE123_STATUS
set -u
cd /mnt/d/unlearning/SD_unlearning || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse

mkdir -p logs
STATUS=models/fcf/PHASE123_STATUS
DONE=models/fcf/PHASE123_DONE
: > "$STATUS"
rm -f "$DONE"
echo "phase123 start $(date -u +%FT%TZ)" | tee -a "$STATUS"

# --- smoke gate: tiny real run of each; abort whole job if any rc!=0 ---
echo "[gate] smoke" | tee -a "$STATUS"
python models/fcf/eval_nude_counts.py --models raw_v14 --limit 2 >/dev/null 2>&1 || { echo "GATE FAIL P1" | tee -a "$STATUS"; exit 1; }
python models/fcf/eval_coco_kid.py --models raw_v14          >/dev/null 2>&1 || { echo "GATE FAIL P3" | tee -a "$STATUS"; exit 1; }
echo "[gate] ok" | tee -a "$STATUS"

# --- Phase 1: nude counts over ALL on-disk *_fs models (no regen) ---
echo "[P1] nude_counts start $(date -u +%FT%TZ)" | tee -a "$STATUS"
python models/fcf/eval_nude_counts.py 2>&1 | tee logs/p1_nude_counts.log
echo "[P1] rc=${PIPESTATUS[0]} done $(date -u +%FT%TZ)" | tee -a "$STATUS"

# --- Phase 3: generate COCO for 3 key models lacking it, then KID/FID-SD ---
echo "[P3] coco gen (key models) start $(date -u +%FT%TZ)" | tee -a "$STATUS"
python eval/eval_coco.py --models fcf_p_official,esd_u,odace_v3 2>&1 | tee logs/p3_coco_gen.log
echo "[P3] coco gen rc=${PIPESTATUS[0]}" | tee -a "$STATUS"
python models/fcf/eval_coco_kid.py 2>&1 | tee logs/p3_coco_kid.log
echo "[P3] kid rc=${PIPESTATUS[0]} done $(date -u +%FT%TZ)" | tee -a "$STATUS"

# --- Phase 2: non-target style LPIPS_u/d (full roster; heaviest, last) ---
echo "[P2] lpips_nontarget start $(date -u +%FT%TZ)" | tee -a "$STATUS"
python eval/lpips_nontarget_style.py 2>&1 | tee logs/p2_lpips_nontarget.log
echo "[P2] rc=${PIPESTATUS[0]} done $(date -u +%FT%TZ)" | tee -a "$STATUS"

echo "phase123 end $(date -u +%FT%TZ)" | tee -a "$STATUS"
touch "$DONE"
