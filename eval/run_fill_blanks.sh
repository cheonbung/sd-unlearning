#!/usr/bin/env bash
# Fill the remaining EMPTY gallery cells for the novel LSSE CAP-CNP family + the violence-trained
# model (verified 2026-06-22). Every metric is merged INCREMENTALLY into its canonical JSON via the
# eval scripts' own load_result() (the other ~23 models are preserved). Each part is non-fatal: a
# failing part leaves its cell pending and the batch continues. Generation is resumable (skips PNGs).
#
# Cells filled (per the build_live_gallery diag):
#   A  COCO-LPIPS   lsse_capcnp / _zero / r2q_a / r2q_ab / r2q_violence   (raw_v14 ref FIRST)
#   B  VanGogh retain (style_img2raw) + forget (style_lpips_f)  same 5 models
#   C  Violence Q16  lsse_capcnp / _zero / r2q_a / r2q_ab        (off-target locality, ~raw expected)
#   D  Nudity full-set ASR (5 attacks)  lsse_r2q_violence        (off-target for the violence model)
#   E  aggregate_cost (picks up lsse_r2q_violence/train_cost.json -> Train/Params/VRAM) + gallery
#
# SMOKE-GATED (smoke-test-before-bg-jobs rule): a real n=2 violence gen+score runs FIRST; if it can't
# load a checkpoint and produce a number the whole job ABORTS before any heavy pass.
# Launch (WSL conda env lsse):  bash eval/run_fill_blanks.sh
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/FILLBLANKS_STATUS
rm -f models/fcf/FILLBLANKS_DONE
echo "=== FILL-BLANKS START $(date) ===" | tee "$ST"

NUDITY="lsse_capcnp,lsse_capcnp_zero,lsse_r2q_a,lsse_r2q_ab"
ALL5="lsse_capcnp,lsse_capcnp_zero,lsse_r2q_a,lsse_r2q_ab,lsse_r2q_violence"

# ---- SMOKE: cheapest real path (violence n=2 on one model). Abort all if it can't load+score. ----
echo "=== SMOKE violence (limit 2, lsse_capcnp) $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models lsse_capcnp --limit 2 2>&1 | tee logs/fillblanks_smoke.log
RC=${PIPESTATUS[0]}
echo "=== SMOKE rc=$RC ===" | tee -a "$ST"
if [ "$RC" -ne 0 ]; then
  echo "SMOKE FAILED (rc=$RC) -> ABORT, no heavy pass run" | tee -a "$ST"
  touch models/fcf/FILLBLANKS_DONE; exit 1
fi

# ---- A: COCO-LPIPS (raw_v14 FIRST = LPIPS reference imgs at tag _lpips) ----
echo "=== A COCO-LPIPS (n=100) $(date) ===" | tee -a "$ST"
python models/fcf/eval_coco_fid5k.py --models "raw_v14,$ALL5" --n 100 --tag _lpips \
  2>&1 | tee logs/fillblanks_cocolpips.log
echo "=== A end (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"

# ---- B: VanGogh gen + style_img2raw, then post-hoc LPIPS_f over the generated imgs ----
echo "=== B VanGogh gen $(date) ===" | tee -a "$ST"
python models/fcf/eval_style_vangogh.py --models "$ALL5" 2>&1 | tee logs/fillblanks_vangogh.log
echo "=== B gen end (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"
python eval/lpips_style.py 2>&1 | tee logs/fillblanks_vglpips.log
echo "=== B lpips_f end (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"

# ---- C: Violence Q16 (4 nudity models; off-target locality) ----
echo "=== C Violence Q16 $(date) ===" | tee -a "$ST"
python models/fcf/eval_violence_q16.py --models "$NUDITY" 2>&1 | tee logs/fillblanks_violence.log
echo "=== C end (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"

# ---- D: Nudity full-set ASR for the violence-trained model (off-target) ----
echo "=== D Nudity full-set (lsse_r2q_violence) $(date) ===" | tee -a "$ST"
python models/fcf/eval_fullset_all.py --models lsse_r2q_violence 2>&1 | tee logs/fillblanks_nudityfs.log
echo "=== D end (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"

# ---- E: aggregate cost (lsse_r2q_violence/train_cost.json) + rebuild gallery ----
echo "=== E aggregate_cost + gallery $(date) ===" | tee -a "$ST"
python eval/aggregate_cost.py 2>&1 | tee logs/fillblanks_aggregate.log
python compare/build_live_gallery.py 2>&1 | tee logs/fillblanks_gallery.log
echo "=== E end (rc=${PIPESTATUS[0]}) $(date) ===" | tee -a "$ST"

echo "=== FILL-BLANKS DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add models/fcf/{coco5k_lpips.json,style_vangogh.json,violence_q16.json,fullset_all.json,train_cost.json} eval/run_fill_blanks.sh compare/build_live_gallery.py ; commit ; push" | tee -a "$ST"
touch models/fcf/FILLBLANKS_DONE
