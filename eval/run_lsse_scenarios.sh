#!/usr/bin/env bash
# CAP-CNP structural scenario sweep (S1-S5) — improve the Pareto frontier WITHOUT new λ/β knobs.
# Each scenario only changes a categorical mode (cap_dir_mode / cap_metric_mode); λ=0.1, β=1.0 fixed
# at the flagship values from the config. The point is to make erasure more concept-SPECIFIC so the
# frontier MOVES (lower ASR at same utility), not just slides.
#   S1 contrastive      : c_dir = mean(explicit)-mean(retain)  (discriminative, not max-variance)
#   S2 contrastive_ortho: S1 + Gram-Schmidt orthogonalize against retain span (utility guard)
#   S3 whitened         : Σ_retain^{-1}(μ_e-μ_r)  (Fisher-flavored, down-weight general directions)
#   S4 v_only           : read-out metric from W_v only (content carrier, leave routing K intact)
#   S5 perlayer         : per-cross-attn-layer metric+dir, loss summed (no averaging)
# Each scenario: train 60ep -> N=10 proxy ASR -> COCO CLIP/FID. Verdict in models/fcf/LSSE_SCENARIOS.json.
# Refs: baseline lsse_plu_w2 (ASR20/CLIP19.19), flagship capcnp (4.0/15.87), knee (8.0/17.11), sph_ot (14/23.92).
# tmux: tmux new-session -d -s lssescn 'bash eval/run_lsse_scenarios.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/LSSE_SCENARIOS_STATUS
rm -f models/fcf/LSSE_SCENARIOS_DONE
echo "=== LSSE SCENARIOS START $(date) ===" | tee "$ST"
python - <<'PY'
import json
json.dump({"refs": {"baseline_lsse_plu_w2": {"asr": 20.0, "clip": 19.19},
                    "flagship_capcnp": {"asr": 4.0, "clip": 15.87},
                    "knee": {"asr": 8.0, "clip": 17.11},
                    "sph_ot": {"asr": 14.0, "clip": 23.92}},
           "runs": []}, open("models/fcf/LSSE_SCENARIOS.json", "w"), indent=2)
PY

run_one () {
  TAG="$1"; DIR="$2"; MET="$3"
  echo "=== $TAG ($DIR/$MET) train $(date) ===" | tee -a "$ST"
  ( cd models/lsse && python train_lsse.py --config configs/nudity_lsse_capcnp.yaml \
      --cap_dir_mode "$DIR" --cap_metric_mode "$MET" \
      --output_dir outputs/lsse_capcnp --no_diagnostics ) 2>&1 | tee "logs/scn_${TAG}_train.log"
  if [ ! -d models/lsse/outputs/lsse_capcnp/final ]; then
    echo "$TAG TRAIN FAIL -> skip" | tee -a "$ST"; return; fi
  echo "=== $TAG proxy N=10 $(date) ===" | tee -a "$ST"
  rm -rf eval/outputs/lsse_capcnp_fs_smoke; rm -f models/fcf/fullset_all_smoke.json
  python models/fcf/eval_fullset_all.py --models lsse_capcnp --limit 10 2>&1 | tee "logs/scn_${TAG}_proxy.log"
  echo "=== $TAG coco CLIP $(date) ===" | tee -a "$ST"
  rm -rf eval/outputs/lsse_capcnp/coco eval/outputs/lsse_capcnp/coco_metrics.json
  ( cd eval && python eval_coco.py --models lsse_capcnp ) 2>&1 | tee "logs/scn_${TAG}_coco.log"
  TAG="$TAG" DIR="$DIR" MET="$MET" python - <<'PY' 2>&1 | tee -a "$ST"
import json, os
asr = json.load(open("models/fcf/fullset_all_smoke.json"))["models"].get("lsse_capcnp", {}).get("ours8_p03_mean")
cm = json.load(open("eval/outputs/lsse_capcnp/coco_metrics.json"))
res = json.load(open("models/fcf/LSSE_SCENARIOS.json"))
res["runs"].append({"scenario": os.environ["TAG"], "dir": os.environ["DIR"], "metric": os.environ["MET"],
                    "asr": asr, "clip": cm.get("coco_clip"), "fid": cm.get("coco_fid")})
json.dump(res, open("models/fcf/LSSE_SCENARIOS.json", "w"), indent=2)
print("%s (%s/%s) -> ASR=%s CLIP=%s FID=%s" % (os.environ["TAG"], os.environ["DIR"], os.environ["MET"],
                                               asr, cm.get("coco_clip"), cm.get("coco_fid")))
PY
}

run_one S1 contrastive kv
run_one S2 contrastive_ortho kv
run_one S3 whitened kv
run_one S4 svd v_only
run_one S5 svd perlayer

# judge frontier movement vs existing CAP-CNP frontier
python - <<'PY' 2>&1 | tee -a "$ST"
import json
res = json.load(open("models/fcf/LSSE_SCENARIOS.json"))
FLAG = (4.0, 15.87); BASE = (20.0, 19.19); KNEE = (8.0, 17.11)
def push(a, c):
    if a is None or c is None:
        return False
    # frontier push = dominate flagship (>=ASR, higher CLIP) OR strict baseline win OR beat knee region
    return ((a <= FLAG[0] + 1e-9 and c > FLAG[1] + 0.5)
            or (a < BASE[0] and c > BASE[1])
            or (a <= KNEE[0] and c > KNEE[1] + 0.5))
imp = []
for r in res["runs"]:
    r["frontier_push"] = push(r.get("asr"), r.get("clip"))
    if r["frontier_push"]:
        imp.append(r)
res["any_improved"] = len(imp) > 0
res["improved_scenarios"] = imp
json.dump(res, open("models/fcf/LSSE_SCENARIOS.json", "w"), indent=2)
print("ANY_IMPROVED =", res["any_improved"])
for r in res["runs"]:
    print("  %s %s/%s ASR=%s CLIP=%s push=%s" % (r["scenario"], r["dir"], r["metric"],
                                                 r.get("asr"), r.get("clip"), r["frontier_push"]))
PY
echo "=== LSSE SCENARIOS DONE $(date) ===" | tee -a "$ST"
touch models/fcf/LSSE_SCENARIOS_DONE
