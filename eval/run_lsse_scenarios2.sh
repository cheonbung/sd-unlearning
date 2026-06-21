#!/usr/bin/env bash
# CAP-CNP refinement: combine the WINNING retain-preserving direction (contrastive_ortho, S2:
# ASR10/CLIP22.04, strictly dominates baseline) with the aggressive metrics that gave ASR 0 but
# collapsed utility under the broad svd direction (S4 v_only CLIP12.86, S5 perlayer CLIP12.92).
# Hypothesis: contrastive_ortho keeps retain intact, so v_only/perlayer may now reach very low ASR
# WITHOUT the utility collapse -> a new frontier point (low ASR AND high CLIP). λ=0.1, β=1.0 fixed.
#   R1 contrastive_ortho + v_only
#   R2 contrastive_ortho + perlayer
# tmux: tmux new-session -d -s lssescn2 'bash eval/run_lsse_scenarios2.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/LSSE_SCENARIOS2_STATUS
rm -f models/fcf/LSSE_SCENARIOS2_DONE
echo "=== LSSE SCENARIOS2 START $(date) ===" | tee "$ST"
python - <<'PY'
import json
json.dump({"refs": {"baseline": {"asr": 20.0, "clip": 19.19},
                    "S2_contrastive_ortho_kv": {"asr": 10.0, "clip": 22.04},
                    "sph_ot": {"asr": 14.0, "clip": 23.92}},
           "runs": []}, open("models/fcf/LSSE_SCENARIOS2.json", "w"), indent=2)
PY

run_one () {
  TAG="$1"; DIR="$2"; MET="$3"
  echo "=== $TAG ($DIR/$MET) train $(date) ===" | tee -a "$ST"
  ( cd models/lsse && python train_lsse.py --config configs/nudity_lsse_capcnp.yaml \
      --cap_dir_mode "$DIR" --cap_metric_mode "$MET" \
      --output_dir outputs/lsse_capcnp --no_diagnostics ) 2>&1 | tee "logs/scn2_${TAG}_train.log"
  if [ ! -d models/lsse/outputs/lsse_capcnp/final ]; then
    echo "$TAG TRAIN FAIL -> skip" | tee -a "$ST"; return; fi
  echo "=== $TAG proxy N=10 $(date) ===" | tee -a "$ST"
  rm -rf eval/outputs/lsse_capcnp_fs_smoke; rm -f models/fcf/fullset_all_smoke.json
  python models/fcf/eval_fullset_all.py --models lsse_capcnp --limit 10 2>&1 | tee "logs/scn2_${TAG}_proxy.log"
  echo "=== $TAG coco CLIP $(date) ===" | tee -a "$ST"
  rm -rf eval/outputs/lsse_capcnp/coco eval/outputs/lsse_capcnp/coco_metrics.json
  ( cd eval && python eval_coco.py --models lsse_capcnp ) 2>&1 | tee "logs/scn2_${TAG}_coco.log"
  TAG="$TAG" DIR="$DIR" MET="$MET" python - <<'PY' 2>&1 | tee -a "$ST"
import json, os
asr = json.load(open("models/fcf/fullset_all_smoke.json"))["models"].get("lsse_capcnp", {}).get("ours8_p03_mean")
cm = json.load(open("eval/outputs/lsse_capcnp/coco_metrics.json"))
res = json.load(open("models/fcf/LSSE_SCENARIOS2.json"))
res["runs"].append({"scenario": os.environ["TAG"], "dir": os.environ["DIR"], "metric": os.environ["MET"],
                    "asr": asr, "clip": cm.get("coco_clip"), "fid": cm.get("coco_fid")})
json.dump(res, open("models/fcf/LSSE_SCENARIOS2.json", "w"), indent=2)
print("%s (%s/%s) -> ASR=%s CLIP=%s FID=%s" % (os.environ["TAG"], os.environ["DIR"], os.environ["MET"],
                                               asr, cm.get("coco_clip"), cm.get("coco_fid")))
PY
}

run_one R1 contrastive_ortho v_only
run_one R2 contrastive_ortho perlayer

python - <<'PY' 2>&1 | tee -a "$ST"
import json
res = json.load(open("models/fcf/LSSE_SCENARIOS2.json"))
S2 = (10.0, 22.04); BASE = (20.0, 19.19)
def better_than_s2(a, c):  # dominates S2 or non-dominated with clearly lower ASR
    if a is None or c is None: return False
    return (a <= S2[0] and c > S2[1]) or (a < S2[0] - 1 and c >= S2[1] - 1.0)
for r in res["runs"]:
    r["beats_S2"] = better_than_s2(r.get("asr"), r.get("clip"))
    r["strict_baseline_win"] = (r.get("asr") is not None and r.get("clip") is not None
                                and r["asr"] < BASE[0] and r["clip"] > BASE[1])
json.dump(res, open("models/fcf/LSSE_SCENARIOS2.json", "w"), indent=2)
for r in res["runs"]:
    print("  %s %s/%s ASR=%s CLIP=%s beats_S2=%s baseline_win=%s"
          % (r["scenario"], r["dir"], r["metric"], r.get("asr"), r.get("clip"),
             r["beats_S2"], r["strict_baseline_win"]))
PY
echo "=== LSSE SCENARIOS2 DONE $(date) ===" | tee -a "$ST"
touch models/fcf/LSSE_SCENARIOS2_DONE
