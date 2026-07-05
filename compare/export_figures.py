"""Export paper-ready STATIC figures (PNG + PDF) from the canonical JSON tables.

The live gallery renders these interactively (JS/SVG), but a paper needs vector figures. This produces
the two quantitative centerpieces + the LSSE sweep, all from existing JSON (no GPU, no regen):

  fig_coherence_scatter : OOD coherence (Ring-A-Bell person_prob, x) vs 4-lab ASR (y) for all models,
                          with the shaded collapse zone. Bottom-right = honest erasure.
  fig_rpgrt_curves      : RPG-RT DPO adaptation curves (asr_query per iteration) per target.
  fig_lsse_sweep        : LSSE-family strength sweep (8-lab ASR vs ring) showing ASR<->collapse coupling.

Run (Windows anaconda python; needs matplotlib):
  C:/Users/USER/anaconda3/python.exe compare/export_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FCF = REPO / "models" / "fcf"
OUTDIR = REPO / "compare" / "figures"
OUTDIR.mkdir(exist_ok=True)

GROUP = {"ref": "#4a86c8", "baseline": "#8a929a", "novel": "#2fa877"}
LABELS = {
    "raw_v14": "Raw SD", "fcf_p_official": "FCF-P", "esd_u": "ESD-u", "safeclip": "Safe-CLIP",
    "sld_max": "SLD-Max", "sph_ot": "SLERP-OT", "odace_v3": "ODACE v3", "lsse_r2q_ab": "LSSE R2q-ab",
    "lsse_r2q_a": "LSSE R2q-a", "lsse_capcnp_zero": "LSSE capcnp-0", "odace_benign_n1": "ODACE benign-neg",
    "odace_benign": "ODACE benign", "lsse_geo_e2": "LSSE geo-e2", "lsse_geo_raw_e2": "LSSE geo-raw",
    "lsse_capcnp": "LSSE R2", "lsse_geo_raw_e3": "LSSE geo-raw3", "lsse_geo_tk1": "LSSE geo-tk1",
}
GROUP_OF = {"raw_v14": "ref", "fcf_p_official": "baseline", "esd_u": "baseline", "safeclip": "baseline",
            "sld_max": "baseline"}  # everything else defaults to "novel"


def L(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _asr4() -> dict:
    out = {}
    for k, d in (L(FCF / "fullset_eval.json").get("models", {})).items():
        if d.get("asr_fcf4_mean") is not None:
            out[k] = d["asr_fcf4_mean"]
    for k, d in (L(FCF / "fullset_all.json").get("models", {})).items():
        if isinstance(d, dict) and d.get("fcf4_p03_mean") is not None:
            out[k] = d["fcf4_p03_mean"]
    return out


def _asr8() -> dict:
    out = {}
    for k, d in (L(FCF / "fullset_all.json").get("models", {})).items():
        if isinstance(d, dict) and d.get("ours8_p03_mean") is not None:
            out[k] = d["ours8_p03_mean"]
    return out


def _ring() -> dict:
    return {k: (v.get("ring_a_bell") or {}).get("person_prob")
            for k, v in L(FCF / "coherence.json").get("models", {}).items()
            if isinstance(v, dict)}


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", OUTDIR / f"{name}.png")


# Method representatives for the main scatter (drop research-only LSSE variants -> those live in the sweep).
REPRESENTATIVE = ["raw_v14", "fcf_p_official", "esd_u", "safeclip", "sld_max", "sph_ot",
                  "odace_v3", "lsse_r2q_ab", "lsse_r2q_a", "lsse_capcnp_zero",
                  "odace_benign_n1", "odace_benign", "lsse_geo_e2"]


def fig_coherence_scatter() -> None:
    ring, asr4 = _ring(), _asr4()
    keys = [k for k in REPRESENTATIVE if k in ring and k in asr4 and ring[k] is not None]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axvspan(0, 40, color="#e05555", alpha=0.07)
    for k in keys:
        g = GROUP_OF.get(k, "novel")
        ax.scatter(ring[k] * 100, asr4[k], s=46, color=GROUP[g], edgecolor="#222", linewidth=0.4, zorder=3)
        ax.annotate(LABELS.get(k, k), (ring[k] * 100, asr4[k]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points", color="#333")
    ax.axvline(40, color="#e05555", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Ring-A-Bell person coherence (%)  ->  on-manifold")
    ax.set_ylabel("nudity ASR 4-lab (%)  (lower better)")
    ax.set_title("Coherence vs ASR - collapse zone (shaded) reaches low ASR dishonestly")
    ax.text(4, ax.get_ylim()[1] * 0.9, "collapse zone", color="#c0392b", fontsize=8)
    ax.grid(True, alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in GROUP.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    _save(fig, "fig_coherence_scatter")


def fig_rpgrt_curves() -> None:
    models = L(FCF / "rpgrt_dpo.json").get("models", {})
    order = ["raw", "esd_u", "fcf_p_official", "odace_v3", "lsse_geo_e2", "odace_benign",
             "odace_benign_n1", "sph_ot"]
    names = {"raw": "Raw SD", "esd_u": "ESD-u", "fcf_p_official": "FCF-P", "odace_v3": "ODACE v3",
             "lsse_geo_e2": "LSSE geo", "odace_benign": "ODACE benign", "odace_benign_n1": "ODACE benign-neg",
             "sph_ot": "SLERP-OT"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k in order:
        cur = (models.get(k, {}) or {}).get("curve") or []
        seq = [(int(p["iter"]), float(p["asr_query"])) for p in cur if p.get("asr_query") is not None]
        if not seq:
            continue
        xs, ys = zip(*seq)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.8, label=names.get(k, k))
    ax.set_xlabel("DPO iteration (0 = frozen attacker)")
    ax.set_ylabel("RPG-RT ASR (% queries NSFW)  (lower better)")
    ax.set_title("Adaptive red-team: redirect defenses stay flat & low")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, "fig_rpgrt_curves")


def fig_lsse_sweep() -> None:
    ring, asr8 = _ring(), _asr8()
    keys = [k for k in asr8 if (k.startswith("lsse") or k == "vanilla_lsse") and ring.get(k) is not None]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.axvspan(0, 40, color="#e05555", alpha=0.07)
    for k in keys:
        ax.scatter(ring[k] * 100, asr8[k], s=42, color="#2fa877", edgecolor="#222", linewidth=0.4, zorder=3)
        ax.annotate(LABELS.get(k, k.replace("lsse_", "")), (ring[k] * 100, asr8[k]), fontsize=6.5,
                    xytext=(4, 2), textcoords="offset points", color="#333")
    ax.set_xlabel("Ring-A-Bell person coherence (%)")
    ax.set_ylabel("nudity ASR 8-lab (%)")
    ax.set_title("LSSE strength sweep: low ASR is bought with collapse (low coherence)")
    ax.grid(True, alpha=0.25)
    _save(fig, "fig_lsse_sweep")


def fig_multiseed() -> None:
    ms = L(FCF / "multiseed.json").get("models", {})
    if not ms:
        print("skip fig_multiseed: no multiseed.json")
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axvspan(0, 40, color="#e05555", alpha=0.07)
    for k, r in ms.items():
        p, a = r.get("person", {}), r.get("asr4", {})
        if p.get("mean") is None:
            continue
        x, xe = p["mean"] * 100, (p.get("std") or 0) * 100
        y, ye = a["mean"], a.get("std") or 0
        col = "#c0392b" if x < 40 else "#2fa877"
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=7, color=col, ecolor=col,
                    elinewidth=1.2, capsize=3, zorder=3)
        ax.annotate(LABELS.get(k, k), (x, y), fontsize=7.5, xytext=(6, 4),
                    textcoords="offset points", color="#333")
    ax.axvline(40, color="#e05555", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Ring-A-Bell coherence (%)  (mean +/- std, 3 seeds)")
    ax.set_ylabel("Ring-A-Bell ASR 4-lab (%)  (mean +/- std)")
    ax.set_title("Multi-seed: collapse vs coherent separation is not seed noise")
    ax.grid(True, alpha=0.25)
    _save(fig, "fig_multiseed")


def main() -> None:
    fig_coherence_scatter()
    fig_rpgrt_curves()
    fig_lsse_sweep()
    fig_multiseed()
    print("FIGURES_DONE_OK")


if __name__ == "__main__":
    main()
