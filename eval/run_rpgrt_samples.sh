#!/usr/bin/env bash
# RPG-RT SAMPLE-IMAGE fill for the live gallery's "RPG-RT attack samples" grid.
# Generates iter0 attack images for the registered Quantitative-results models that lack a sample set,
# WITHOUT clobbering the existing valid DPO curves in models/fcf/rpgrt_dpo.json:
#   - run rpgrt_dpo_attack.py --iters 1 into a THROWAWAY out dir (temp dpo_summary discarded),
#   - copy only iter0_*.png into eval/outputs/rpgrt_dpo/<t>/img/ (the layout the gallery reads),
#   - rpgrt_label_iter0.py writes nsfw_iter0.json per target (no GPU),
#   - rebuild gallery.
# Registered targets missing samples (eval/rpgrt_dpo_attack.py UNET_DIRS/TE_DIRS): sld_max, safeclip,
# fcf_e_official, odace_benign, odace_benign_n1, lsse_geo_e2. The 5 pure-baseline variants
# (raw_v15, sd21base, sld_medium, sld_strong, safe_neg) are NOT registered in the attack pipeline ->
# out of scope until registered. raw_v14/sph_ot/fcf_p_official/esd_u already have samples.
# HEAVY: vicuna-7b 4bit + SD per target, ~30-60 min/target. Single GPU, RPG-RT conda env, NO git.
# tmux: tmux new-session -d -s rpgrtsamp 'bash eval/run_rpgrt_samples.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RPG-RT
mkdir -p logs eval/outputs/rpgrt_dpo
RPG=/mnt/d/unlearning/RPG-RT
REPO=/mnt/d/unlearning/SD_unlearning
TMP=/mnt/d/unlearning/SD_unlearning/eval/outputs/rpgrt_samp_tmp

ST=models/fcf/RPGRT_SAMP_STATUS
rm -f models/fcf/RPGRT_SAMP_DONE
echo "=== RPG-RT SAMPLES START $(date) ===" | tee "$ST"

TARGETS="sld_max safeclip fcf_e_official odace_benign odace_benign_n1 lsse_geo_e2"

# --- smoke gate: tiny real iter0 gen on one target; abort whole job if rc!=0 ---
echo "=== smoke (safeclip 1iter 2x2) START $(date) ===" | tee -a "$ST"
( cd "$RPG" && python "$REPO"/eval/rpgrt_dpo_attack.py --target safeclip --iters 1 --n_train 2 \
    --n_eval 2 --group 2 --n_query 2 --out "$TMP"/_smoke ) 2>&1 | tee logs/rpgrt_samp_smoke.log
SRC=${PIPESTATUS[0]}
echo "=== smoke END $(date) (rc=$SRC) ===" | tee -a "$ST"
if [ "$SRC" -ne 0 ]; then echo "SMOKE FAILED -> abort" | tee -a "$ST"; touch models/fcf/RPGRT_SAMP_DONE; exit 1; fi

# --- per-target iter0 image gen (--iters 1: iter0 eval saves iter0_*.png before the single DPO step) ---
for T in $TARGETS; do
  echo "=== gen $T START $(date) ===" | tee -a "$ST"
  ( cd "$RPG" && python "$REPO"/eval/rpgrt_dpo_attack.py --target "$T" --iters 1 --n_train 20 \
      --n_eval 20 --group 6 --n_query 10 --out "$TMP"/"$T" ) 2>&1 | tee "logs/rpgrt_samp_$T.log"
  rc=${PIPESTATUS[0]}
  # copy ONLY iter0 images into the gallery dir; leave any existing dpo_summary untouched
  if [ "$rc" -eq 0 ]; then
    mkdir -p eval/outputs/rpgrt_dpo/"$T"/img
    cp "$TMP"/"$T"/img/iter0_*.png eval/outputs/rpgrt_dpo/"$T"/img/ 2>/dev/null
  fi
  echo "=== gen $T END $(date) (rc=$rc) ===" | tee -a "$ST"
done

# --- label iter0 images -> nsfw_iter0.json per target (no GPU) ---
echo "=== label iter0 $(date) ===" | tee -a "$ST"
python eval/rpgrt_label_iter0.py 2>&1 | tee logs/rpgrt_samp_label.log
echo "=== label rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$ST"

python compare/build_live_gallery.py 2>&1 | tee logs/rpgrt_samp_gallery.log || true
echo "=== RPG-RT SAMPLES DONE $(date) ===" | tee -a "$ST"
touch models/fcf/RPGRT_SAMP_DONE
