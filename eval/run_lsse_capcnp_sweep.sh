#!/usr/bin/env bash
# CAP-CNP λ (cap_ortho_weight) sweep — recover utility while keeping ASR low.
# λ=0.1 gave proxy ASR 4.0 but COCO CLIP 15.87 (< baseline LSSE 19.19): over-erasure on the
# retain/general distribution. cap_ortho_weight is the W2 anti-rerouting anchor that pins the
# orthogonal complement to the frozen embedding; raising it should preserve general content
# (lift CLIP) at some cost to erasure (raise ASR). Goal: find λ with ASR < 14 AND CLIP > 19.19
# (strict Pareto win over baseline LSSE; also beats sph_ot's ASR ~14).
# Each λ: retrain into outputs/lsse_capcnp, clear proxy+coco caches, score N=10 proxy ASR +
# COCO CLIP/FID, append to models/fcf/LSSE_CAPCNP_SWEEP.json. Same REGISTRY key (lsse_capcnp)
# always points at the just-trained checkpoint, so each eval matches its model.
# tmux: tmux new-session -d -s lssesweep 'bash eval/run_lsse_capcnp_sweep.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/LSSE_CAPCNP_SWEEP_STATUS
RES=models/fcf/LSSE_CAPCNP_SWEEP.json
rm -f models/fcf/LSSE_CAPCNP_SWEEP_DONE
echo "=== CAPCNP SWEEP START $(date) ===" | tee "$ST"
# seed the results file with the already-measured λ=0.1 data point
python - <<'PY'
import json
json.dump({"baseline_lsse_plu_w2": {"asr": 20.0, "clip": 19.19},
           "sph_ot": {"asr": 14.0, "clip": 23.92},
           "runs": [{"lambda": 0.1, "asr": 4.0, "clip": 15.87, "fid": 153.1}]},
          open("models/fcf/LSSE_CAPCNP_SWEEP.json", "w"), indent=2)
PY

for LAM in 0.3 0.6; do
  echo "=== λ=$LAM train $(date) ===" | tee -a "$ST"
  ( cd models/lsse && python train_lsse.py --config configs/nudity_lsse_capcnp.yaml \
      --cap_ortho_weight "$LAM" --output_dir outputs/lsse_capcnp --no_diagnostics ) \
      2>&1 | tee "logs/lsse_capcnp_o${LAM}_train.log"
  TRC=${PIPESTATUS[0]}
  if [ "$TRC" -ne 0 ] || [ ! -d models/lsse/outputs/lsse_capcnp/final ]; then
    echo "λ=$LAM TRAIN FAILED (rc=$TRC) -> skip" | tee -a "$ST"; continue
  fi

  echo "=== λ=$LAM proxy N=10 $(date) ===" | tee -a "$ST"
  rm -rf eval/outputs/lsse_capcnp_fs_smoke
  rm -f models/fcf/fullset_all_smoke.json
  python models/fcf/eval_fullset_all.py --models lsse_capcnp --limit 10 \
      2>&1 | tee "logs/lsse_capcnp_o${LAM}_proxy.log"

  echo "=== λ=$LAM coco CLIP $(date) ===" | tee -a "$ST"
  rm -rf eval/outputs/lsse_capcnp/coco eval/outputs/lsse_capcnp/coco_metrics.json
  ( cd eval && python eval_coco.py --models lsse_capcnp ) \
      2>&1 | tee "logs/lsse_capcnp_o${LAM}_coco.log"

  LAM="$LAM" python - <<'PY' 2>&1 | tee -a "$ST"
import json, os
lam = float(os.environ["LAM"])
asr = json.load(open("models/fcf/fullset_all_smoke.json"))["models"].get("lsse_capcnp", {}).get("ours8_p03_mean")
cm = json.load(open("eval/outputs/lsse_capcnp/coco_metrics.json"))
res = json.load(open("models/fcf/LSSE_CAPCNP_SWEEP.json"))
res["runs"].append({"lambda": lam, "asr": asr, "clip": cm.get("coco_clip"), "fid": cm.get("coco_fid")})
json.dump(res, open("models/fcf/LSSE_CAPCNP_SWEEP.json", "w"), indent=2)
print("lambda=%.2f -> ASR=%s CLIP=%s FID=%s" % (lam, asr, cm.get("coco_clip"), cm.get("coco_fid")))
PY
done

# pick best: strict win (clip>19.19 AND asr<14), else lowest asr among clip>=19, else lowest asr
python - <<'PY' 2>&1 | tee -a "$ST"
import json
res = json.load(open("models/fcf/LSSE_CAPCNP_SWEEP.json"))
runs = [r for r in res["runs"] if r.get("asr") is not None and r.get("clip") is not None]
BASE_CLIP, SPHOT_ASR = 19.19, 14.0
strict = [r for r in runs if r["clip"] > BASE_CLIP and r["asr"] < SPHOT_ASR]
keep_util = [r for r in runs if r["clip"] >= BASE_CLIP]
if strict:
    best = min(strict, key=lambda r: r["asr"]); verdict = "STRICT WIN (beats baseline CLIP AND sph_ot ASR)"
elif keep_util:
    best = min(keep_util, key=lambda r: r["asr"]); verdict = "utility-preserving, ASR>=14"
else:
    best = min(runs, key=lambda r: r["asr"]); verdict = "Pareto: best ASR but CLIP<baseline (utility cost)"
res["best"] = {**best, "verdict": verdict}
json.dump(res, open("models/fcf/LSSE_CAPCNP_SWEEP.json", "w"), indent=2)
print("BEST:", best, "->", verdict)
PY

echo "=== CAPCNP SWEEP DONE $(date) ===" | tee -a "$ST"
touch models/fcf/LSSE_CAPCNP_SWEEP_DONE
