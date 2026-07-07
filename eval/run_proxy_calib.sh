#!/usr/bin/env bash
# Phase 0 of the DACE-improve plan: CALIBRATE a fast proxy eval against the full-set ground truth.
# eval_fullset_all.py --limit N generates only N prompts/attack (5 attacks) into fullset_all_smoke.json
# (deterministic: first-N prompts). We run a small N on 7 anchor models whose FULL-SET ours8 ASR we
# already know, then check Spearman(proxy, full) + key ordering. The smallest N with rho>=0.9 AND
# odace<sph_ot<dace_v2 ordering becomes the trusted per-iteration scorer for Phases 1-3.
# Anchors' full-set ours8 means (from fullset_eval/all.json): dace_v2 59.4, dace_plu 79.7, sph_ot 14.0,
#   vanilla_lsse 51.5, esd_u 23.0, fcf_p_official 16.9, odace_v3 5.2.
# tmux: tmux new-session -d -s proxycalib 'bash eval/run_proxy_calib.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/PROXY_CALIB_STATUS
rm -f models/fcf/PROXY_CALIB_DONE
echo "=== PROXY_CALIB START $(date) ===" | tee "$ST"

ANCHORS="dace_v2,dace_plu,sph_ot,vanilla_lsse,esd_u,fcf_p_official,odace_v3"

run_n() {  # $1 = N prompts/attack
  local N="$1"
  echo "=== proxy run N=$N (7 anchors x $N x 5 attacks) $(date) ===" | tee -a "$ST"
  rm -f models/fcf/fullset_all_smoke.json
  python models/fcf/eval_fullset_all.py --models "$ANCHORS" --include_done --limit "$N" \
    2>&1 | tee "logs/proxy_calib_N${N}.log"
  local RC=${PIPESTATUS[0]}
  cp -f models/fcf/fullset_all_smoke.json "models/fcf/proxy_calib_N${N}.json" 2>/dev/null
  echo "=== proxy run N=$N END (rc=$RC) $(date) ===" | tee -a "$ST"
}

for N in 10 20; do run_n "$N"; done

echo "=== correlate proxy vs full-set $(date) ===" | tee -a "$ST"
python - <<'PY' 2>&1 | tee -a "$ST"
import json, os
TRUTH = {"dace_v2":59.4,"dace_plu":79.7,"sph_ot":14.0,"vanilla_lsse":51.5,
         "esd_u":23.0,"fcf_p_official":16.9,"odace_v3":5.2}
def spearman(a, b):
    ks=list(a.keys())
    def ranks(d):
        order=sorted(ks, key=lambda k: d[k]); r={}
        for i,k in enumerate(order): r[k]=i+1
        return r
    ra, rb = ranks(a), ranks(b)
    n=len(ks); ds=sum((ra[k]-rb[k])**2 for k in ks)
    return 1 - 6*ds/(n*(n*n-1))
verdict={"candidates":{}}
best=None
for N in (10,20):
    p="models/fcf/proxy_calib_N%d.json"%N
    if not os.path.exists(p):
        print("N=%d: no json"%N); continue
    m=json.load(open(p)).get("models",{})
    proxy={k:(m.get(k,{}) or {}).get("ours8_p03_mean") for k in TRUTH}
    if any(v is None for v in proxy.values()):
        miss=[k for k,v in proxy.items() if v is None]
        print("N=%d: MISSING %s -> skip"%(N,miss)); continue
    rho=spearman(proxy, TRUTH)
    order_ok = proxy["odace_v3"] < proxy["sph_ot"] < proxy["dace_v2"]
    mae=sum(abs(proxy[k]-TRUTH[k]) for k in TRUTH)/len(TRUTH)
    ok = (rho>=0.9) and order_ok
    verdict["candidates"]["N%d"%N]={"rho":round(rho,3),"order_ok":order_ok,"mae":round(mae,1),
                                    "pass":ok,"proxy":{k:round(v,1) for k,v in proxy.items()}}
    print("N=%-3d rho=%.3f order_ok=%s mae=%.1f pass=%s"%(N,rho,order_ok,mae,ok))
    if ok and best is None: best=N
verdict["chosen_N"]=best
verdict["truth"]=TRUTH
verdict["next"]=("Phase1: per-token + adaptive-k DACE; score with N=%d proxy"%best) if best else \
                "ESCALATE: no N in {10,20} reached rho>=0.9 -> try N=40 / discriminative attacks"
json.dump(verdict, open("models/fcf/PROXY_CALIB.json","w"), indent=2)
print("CHOSEN_N =", best)
print("->", verdict["next"])
PY

echo "=== PROXY_CALIB DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/run_proxy_calib.sh models/fcf/PROXY_CALIB.json ; commit" | tee -a "$ST"
touch models/fcf/PROXY_CALIB_DONE
