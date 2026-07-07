#!/usr/bin/env bash
# Swap the gallery's P4D column from the pre-optim "P4D-sel" selection to the REAL optimized P4D set
# (zhiyichin/p4d union: p4dn_16 + p4dk_3 = 272 unique prompts, already written to
# models/fcf/data/eval/p4d_nudity.txt; old selection backed up to p4d_sel_nudity.txt).
#
# Both eval_fullset.py (raw/fcf_p/fcf_e -> fullset_eval.json) and eval_fullset_all.py (the other 12
# gallery models -> fullset_all.json) read p4d_nudity.txt and write images to eval/outputs/<m>_fs/p4d/.
# xeval.generate is resume-by-filename, so we CLEAR only the p4d/ image dirs first (the 4 unchanged
# attacks I2P/RaB/RaB(Re)/UDA stay on disk -> skipped instantly); only P4D (272 imgs x 15) regenerates.
# tmux: tmux new-session -d -s p4dreal 'bash eval/run_p4d_real.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/P4DREAL_STATUS
rm -f models/fcf/P4DREAL_DONE
echo "=== P4D REAL START $(date) ===" | tee "$ST"

# 15 gallery models (must match build_live_gallery.py MODELS). raw/fcf via eval_fullset.py; rest via eval_fullset_all.py.
ALL12="raw_v15,sd21base,sld_medium,sld_strong,sld_max,safeclip,safe_neg,esd_u,sph_ot,odace_benign,odace_benign_n1,lsse_geo_e2"

# --- clear ONLY the p4d image dirs so the new 272-prompt P4D regenerates (other attacks untouched) ---
echo "=== clearing p4d/ image dirs $(date) ===" | tee -a "$ST"
for d in eval/outputs/*_fs/p4d; do [ -d "$d" ] && rm -rf "$d"; done
echo "cleared p4d dirs" | tee -a "$ST"

# --- smoke gate: 3-prompt run on one model (writes to _fs_smoke + fullset_all_smoke.json, isolated) ---
echo "=== smoke (odace_benign_n1 --limit 3) START $(date) ===" | tee -a "$ST"
python models/fcf/eval_fullset_all.py --models odace_benign_n1 --limit 3 2>&1 | tee logs/p4dreal_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "SMOKE FAILED -> abort" | tee -a "$ST"; touch models/fcf/P4DREAL_DONE; exit 1; fi

# --- raw / fcf_p / fcf_e -> fullset_eval.json (P4D regen; 4 other attacks resume-skipped) ---
echo "=== eval_fullset.py (raw/fcf_p/fcf_e) START $(date) ===" | tee -a "$ST"
python models/fcf/eval_fullset.py 2>&1 | tee logs/p4dreal_rawfcf.log
echo "=== eval_fullset.py END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

# --- the other 12 gallery models -> fullset_all.json ---
echo "=== eval_fullset_all.py (12 models) START $(date) ===" | tee -a "$ST"
python models/fcf/eval_fullset_all.py --models "$ALL12" 2>&1 | tee logs/p4dreal_all12.log
echo "=== eval_fullset_all.py END $(date) (rc=${PIPESTATUS[0]}) ===" | tee -a "$ST"

# --- rebuild gallery (label already flipped P4D-sel -> P4D in source) ---
python compare/build_live_gallery.py 2>&1 | tee logs/p4dreal_gallery.log || true
echo "=== P4D REAL DONE $(date) ===" | tee -a "$ST"
touch models/fcf/P4DREAL_DONE
