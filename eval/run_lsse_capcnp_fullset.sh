#!/usr/bin/env bash
# Full-set finalizer for the CAP-CNP win: real full-set ASR (1622 prompts x 5 attacks, NO --limit ->
# fullset_all.json) + COCO CLIP/FID for both frontier points:
#   lsse_capcnp      = S2 (contrastive_ortho/kv)      balanced win (canonical, already on disk)
#   lsse_capcnp_zero = R2 (contrastive_ortho/perlayer) max-forget variant (trained here)
# Also re-runs COCO for lsse_capcnp so coco_metrics.json matches the final S2 checkpoint (the sweep
# left R2's stale coco there). Results feed the gallery.
# tmux: tmux new-session -d -s lssefull 'bash eval/run_lsse_capcnp_fullset.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/LSSE_FULLSET_STATUS
rm -f models/fcf/LSSE_FULLSET_DONE
echo "=== LSSE FULLSET START $(date) ===" | tee "$ST"

# 1) train R2 max-forget variant -> outputs/lsse_capcnp_zero
echo "=== train R2 lsse_capcnp_zero (contrastive_ortho/perlayer) $(date) ===" | tee -a "$ST"
( cd models/lsse && python train_lsse.py --config configs/nudity_lsse_capcnp.yaml \
    --cap_dir_mode contrastive_ortho --cap_metric_mode perlayer \
    --output_dir outputs/lsse_capcnp_zero --no_diagnostics ) 2>&1 | tee logs/lsse_capcnp_zero_train.log
if [ ! -d models/lsse/outputs/lsse_capcnp_zero/final ]; then
  echo "R2 TRAIN FAIL -> will eval lsse_capcnp only" | tee -a "$ST"; fi

# 2) full-set ASR (no --limit -> fullset_all.json). Clear any partial _fs dirs (checkpoints changed).
echo "=== full-set ASR (1622 prompts x 5 attacks) $(date) ===" | tee -a "$ST"
rm -rf eval/outputs/lsse_capcnp_fs eval/outputs/lsse_capcnp_zero_fs
python models/fcf/eval_fullset_all.py --models lsse_capcnp,lsse_capcnp_zero 2>&1 | tee logs/lsse_capcnp_fullset.log

# 3) COCO CLIP/FID. Clear stale S2 coco (sweep left R2's images there) + new zero dir.
echo "=== COCO CLIP/FID $(date) ===" | tee -a "$ST"
rm -rf eval/outputs/lsse_capcnp/coco eval/outputs/lsse_capcnp/coco_metrics.json
rm -rf eval/outputs/lsse_capcnp_zero/coco eval/outputs/lsse_capcnp_zero/coco_metrics.json
( cd eval && python eval_coco.py --models lsse_capcnp,lsse_capcnp_zero ) 2>&1 | tee logs/lsse_capcnp_coco.log

# 4) summary
python - <<'PY' 2>&1 | tee -a "$ST"
import json
fa = json.load(open("models/fcf/fullset_all.json"))["models"]
for k in ["lsse_capcnp", "lsse_capcnp_zero"]:
    r = fa.get(k, {})
    try:
        cm = json.load(open("eval/outputs/%s/coco_metrics.json" % k))
    except Exception:
        cm = {}
    print("%-18s full-set ours8=%s fcf4_p03=%s | COCO CLIP=%s FID=%s"
          % (k, r.get("ours8_p03_mean"), r.get("fcf4_p03_mean"),
             cm.get("coco_clip"), cm.get("coco_fid")))
PY
echo "=== LSSE FULLSET DONE $(date) ===" | tee -a "$ST"
touch models/fcf/LSSE_FULLSET_DONE
