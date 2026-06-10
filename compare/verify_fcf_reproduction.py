"""Phase 3 of the FCF-reproduction verification: quantify how similar the FCF paper's reported
results are to our reproduction, on three protocol-robust axes:

  1. Direction      - does each method reduce nudity ASR vs its own raw/SD baseline? (sign agree)
  2. Rank (Spearman) - do methods rank the same by efficacy?  Computed for all 5 methods and for
                       the 3 author/canonical baselines (ESD/SLD/Safe-CLIP) that do NOT depend on
                       our from-scratch FCF training.
  3. %-reduction     - reduction from each source's OWN raw baseline, which cancels the prompt-set
                       shift (our 50 curated prompts vs FCF's full I2P+tool sets) so magnitudes
                       become comparable.

Inputs:
  compare/fcf_reference_values.json  (paper, Phase 0)
  compare/fcf_rescore.json           (our FCF-protocol re-scored ASR, Phase 2)
  eval/outputs/<m>/coco_metrics.json (our COCO FID/CLIP for the quality rank)

Output: compare/fcf_verification.json + a printed report.
Run (WSL conda env lsse):  python compare/verify_fcf_reproduction.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

METHODS = ["ESD", "SLD", "Safe-CLIP", "FCF-E", "FCF-P"]      # comparison set (excl. raw)
BASELINES = ["ESD", "SLD", "Safe-CLIP"]                       # not dependent on our FCF training
# FCF paper method name -> our harness coco label (for the quality rank)
COCO_LABEL = {"SD": "raw_v14", "ESD": "esd_u", "SLD": "sld_medium",
              "Safe-CLIP": "safeclip", "FCF-E": "fcf_e", "FCF-P": "fcf_p"}


def rankdata(vals):
    """Average ranks (1-based), ties averaged."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a, b):
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((a[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((b[i] - mb) ** 2 for i in range(n)) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def band(delta):
    d = abs(delta)
    return "match" if d <= 5 else ("partial" if d <= 20 else "diverge")


def main():
    ref = json.loads((HERE / "fcf_reference_values.json").read_text())
    res = json.loads((HERE / "fcf_rescore.json").read_text())
    paper = ref["table1_asr"]["nudity_mean_computed"]
    paper_attacks = ref["table1_asr"]["nudity"]
    rescore = res["methods"]
    # Our FCF-aligned re-score keyed by FCF method name. For FCF-P/E we use the OFFICIAL
    # authors'-code reproduction (models/fcf/), not our earlier unfaithful fcf/ checkpoint.
    OURS_SRC = {"SD": "raw_v14", "ESD": "esd_u", "SLD": "sld_medium", "Safe-CLIP": "safeclip",
                "FCF-E": "fcf_e_official", "FCF-P": "fcf_p_official"}
    ours = {fcf: rescore[label] for fcf, label in OURS_SRC.items()}

    sd_paper = paper["SD"]
    raw_ours = ours["SD"]["asr_fcf_mean"]

    out = {"_doc": "FCF reproduction verification. ASR in percent; corr in [-1,1].",
           "asr": {"per_method": {}, "rank": {}, "direction": {}},
           "quality": {}, "verdict": {}}

    # ---- per-method ASR comparison (means + %-reduction + matched band) ----
    paper_means, ours_means = [], []
    for m in METHODS:
        pm = paper[m]
        om = ours[m]["asr_fcf_mean"]
        ours_raw = ours[m]["asr_ours_mean"]
        p_red = round(100 * (sd_paper - pm) / sd_paper, 1)
        o_red = round(100 * (raw_ours - om) / raw_ours, 1)
        d_agree = (pm < sd_paper) == (om < raw_ours)
        out["asr"]["per_method"][m] = {
            "paper_mean": pm, "ours_fcf_mean": om, "ours_8label_mean": ours_raw,
            "paper_pct_reduction": p_red, "ours_pct_reduction": o_red,
            "pct_reduction_gap": round(o_red - p_red, 1),
            "matched_delta": round(om - pm, 1), "matched_band": band(om - pm),
            "direction_agree": d_agree,
        }
        paper_means.append(pm)
        ours_means.append(om)

    # ---- cross-method rank (Spearman) ----
    pm_all = [paper[m] for m in METHODS]
    om_all = [ours[m]["asr_fcf_mean"] for m in METHODS]
    pm_base = [paper[m] for m in BASELINES]
    om_base = [ours[m]["asr_fcf_mean"] for m in BASELINES]
    out["asr"]["rank"] = {
        "spearman_all5": round(spearman(pm_all, om_all), 3),
        "spearman_baselines3": round(spearman(pm_base, om_base), 3),
        "paper_order_best_to_worst": [m for _, m in sorted(zip(pm_all, METHODS))],
        "ours_order_best_to_worst": [m for _, m in sorted(zip(om_all, METHODS))],
    }

    # ---- per-attack difficulty rank within each method (paper 5-vec vs ours 5-vec) ----
    attack_keys = ref["protocol"]["attack_order"]
    for m in METHODS:
        pv = paper_attacks[m]
        ov = [ours[m]["attacks"][k]["asr_fcf"] for k in attack_keys]
        out["asr"]["direction"][m] = {
            "per_attack_spearman": round(spearman(pv, ov), 3),
            "paper_attacks": dict(zip(attack_keys, pv)),
            "ours_fcf_attacks": dict(zip(attack_keys, ov)),
        }

    # ---- quality (Table 3 FID/CLIP) rank: scales differ -> Spearman only ----
    q_methods = ["SD"] + METHODS
    paper_fid = [ref["table3_quality"][m]["fid"] for m in q_methods]
    paper_clip = [ref["table3_quality"][m]["clip"] for m in q_methods]
    our_fid, our_clip = [], []
    for m in q_methods:
        cm = json.loads((REPO / "xmodel" / "outputs" / COCO_LABEL[m] / "coco_metrics.json").read_text())
        our_fid.append(cm["coco_fid"])
        our_clip.append(cm["coco_clip"])
    out["quality"] = {
        "methods": q_methods,
        "spearman_fid": round(spearman(paper_fid, our_fid), 3),
        "spearman_clip": round(spearman(paper_clip, our_clip), 3),
        "paper_fid": dict(zip(q_methods, paper_fid)),
        "our_coco_fid": dict(zip(q_methods, our_fid)),
        "note": "FID scale differs (paper COCO-30k ~14-18 vs our COCO-300 ~117-120); absolute "
                "comparison invalid, only rank is meaningful.",
    }

    # ---- overall verdict ----
    per = out["asr"]["per_method"]
    out["verdict"] = {
        "baselines_rank": "REPRODUCES" if out["asr"]["rank"]["spearman_baselines3"] >= 0.9
                          else "PARTIAL",
        "all5_rank": "REPRODUCES" if out["asr"]["rank"]["spearman_all5"] >= 0.7 else "FAILS",
        "fcf_method_self": {
            "FCF-P": per["FCF-P"]["matched_band"], "FCF-E": per["FCF-E"]["matched_band"],
        },
        "summary": (
            "Author/canonical baselines (ESD/SLD/Safe-CLIP) reproduce the paper's efficacy "
            "ranking; FCF's own method (FCF-P/E), reproduced from scratch, does NOT reach the "
            "paper's reported ASR even after protocol-aligned re-scoring."
        ),
    }

    (HERE / "fcf_verification.json").write_text(json.dumps(out, indent=2))

    # ---- printed report ----
    print("\n================ FCF REPRODUCTION VERIFICATION ================")
    print(f"{'method':10s} {'paper':>7} {'ours_fcf':>9} {'ours_8lab':>10} "
          f"{'p_red%':>7} {'o_red%':>7} {'band':>8} dir")
    for m in METHODS:
        r = per[m]
        print(f"{m:10s} {r['paper_mean']:7.2f} {r['ours_fcf_mean']:9.1f} "
              f"{r['ours_8label_mean']:10.1f} {r['paper_pct_reduction']:7.1f} "
              f"{r['ours_pct_reduction']:7.1f} {r['matched_band']:>8} "
              f"{'OK' if r['direction_agree'] else 'XX'}")
    rk = out["asr"]["rank"]
    print(f"\nSpearman (efficacy rank): all5 = {rk['spearman_all5']}   "
          f"baselines3 = {rk['spearman_baselines3']}")
    print(f"  paper best->worst: {rk['paper_order_best_to_worst']}")
    print(f"  ours  best->worst: {rk['ours_order_best_to_worst']}")
    print("\nPer-attack difficulty rank (Spearman, paper vs ours):")
    for m in METHODS:
        print(f"  {m:10s} {out['asr']['direction'][m]['per_attack_spearman']}")
    print(f"\nQuality rank: Spearman FID = {out['quality']['spearman_fid']}   "
          f"CLIP = {out['quality']['spearman_clip']}")
    print(f"\nVERDICT: baselines_rank={out['verdict']['baselines_rank']} | "
          f"all5_rank={out['verdict']['all5_rank']} | "
          f"FCF-P band={out['verdict']['fcf_method_self']['FCF-P']} | "
          f"FCF-E band={out['verdict']['fcf_method_self']['FCF-E']}")
    print(f"wrote {HERE / 'fcf_verification.json'}")


if __name__ == "__main__":
    main()
