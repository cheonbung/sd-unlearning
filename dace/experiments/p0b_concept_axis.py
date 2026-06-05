"""P0b -- corrected concept-axis validation.

P0 refuted forget-vs-retain separability (lexically saturated, ~1.0 for all, anti-corr
with ASR). The salvageable hypothesis: the CONCEPT AXIS -- the embedding shift from a
concept-aligned NEUTRAL counterpart (same content, concept word removed) -- tracks ASR.
A good eraser makes adding "naked" barely move the embedding.

Metrics per checkpoint (concept-aligned pairs, no image generation):
  concept_shift   = || mean_i (z_explicit_i - z_neutral_i) ||         (net concept dir)
  pair_dist       = mean_i || z_explicit_i - z_neutral_i ||           (avg movement)
  paired_auc      = held-out AUC: does the mean concept dir separate explicit vs neutral

Run: python experiments/p0b_concept_axis.py
"""
from __future__ import annotations

import logging, re, sys
from pathlib import Path
import torch

DACE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DACE))
from methods.adversary import pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p0b")
MODEL_ID = "openai/clip-vit-large-patch14"

CKPTS = {
    "raw_sd":       (MODEL_ID, 62.0),
    "fcf_e":        (str(REPO / "fcf/outputs/fcf_e_nudity/final"), 61.2),
    "fcf_p":        (str(REPO / "fcf/outputs/fcf_p_nudity/final"), 52.8),
    "vanilla_lsse": (str(REPO / "lsse/outputs/sweep/baseline_seed42/final"), 46.0),
    "lsse_plu":     (str(REPO / "lsse/outputs/sweep/plu_seed42/final"), 21.6),
    "lsse_plu_w2":  (str(REPO / "lsse/outputs/sweep/stack_plu_w2_seed42/final"), 20.8),
    "sph_ot":       (str(REPO / "fcf-novel-methods/outputs/fcf_p_v2_nudity_spherical_ot/final"), 15.6),
}

# concept words -> removed to make the neutral (same content, concept stripped)
_KW = [r"\bnude\b", r"\bnaked\b", r"\bnudity\b", r"\btopless\b", r"\bbare-skinned\b",
       r"\bbare\b", r"\bexposed\b", r"\bundressing\b", r"\bwithout clothes\b",
       r"\bskin-baring\b", r"\bexplicit\b", r"\bsensual\b", r"\bskin-baring\b"]
_KWRE = re.compile("|".join(_KW), flags=re.IGNORECASE)


def neutralize(p: str) -> str:
    s = _KWRE.sub(" ", p)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(a|an)\s+(in|by|on|standing|$)", r"\2", s).strip()
    return s if s else "a photo"


def _spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v)
        for pos, i in enumerate(o): r[i] = pos
        return r
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx)/n, sum(ry)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    sx = sum((a-mx)**2 for a in rx)**0.5; sy = sum((b-my)**2 for b in ry)**0.5
    return cov/(sx*sy+1e-12)


@torch.no_grad()
def enc(model, tok, prompts, device, bs=8):
    out = []
    for i in range(0, len(prompts), bs):
        t = tok(prompts[i:i+bs], padding="max_length", max_length=77,
                truncation=True, return_tensors="pt").to(device)
        out.append(pool(model(t.input_ids).last_hidden_state, "mean").float().cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def paired_auc(de, dn, seed=0, n_splits=5):
    # de,dn: (N,D) explicit/neutral pooled. fit mean concept dir on train, AUC on held-out
    N, D = de.shape
    g = torch.Generator().manual_seed(seed); aucs = []
    for _ in range(n_splits):
        idx = torch.randperm(N, generator=g); h = max(1, N//2)
        tr, te = idx[:h], idx[h:]
        w = (de[tr] - dn[tr]).mean(0)
        se, sn = de[te] @ w, dn[te] @ w
        if se.numel() == 0 or sn.numel() == 0: continue
        diff = se.unsqueeze(1) - sn.unsqueeze(0)
        aucs.append(((diff > 0).float().sum() / (se.numel()*sn.numel())).item())
    return sum(aucs)/len(aucs) if aucs else 0.5


def load_prompts(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def main():
    from transformers import CLIPTextModel, CLIPTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CLIPTokenizer.from_pretrained(MODEL_ID)
    explicit = load_prompts(DACE / "data/prompts/nudity_explicit.txt")
    neutral = [neutralize(p) for p in explicit]
    logger.info("sample pairs:")
    for a, b in list(zip(explicit, neutral))[:4]:
        logger.info(f"   '{a}'  ->  '{b}'")

    rows, shifts, dists, aucs, asrs = [], [], [], [], []
    for name, (path, asr) in CKPTS.items():
        if path != MODEL_ID and not Path(path).exists():
            logger.warning(f"SKIP {name}"); continue
        m = CLIPTextModel.from_pretrained(path).to(device).eval()
        ze, zn = enc(m, tok, explicit, device), enc(m, tok, neutral, device)
        d = ze - zn
        shift = d.mean(0).norm().item()
        dist = d.norm(dim=1).mean().item()
        auc = paired_auc(ze, zn)
        rows.append((name, shift, dist, auc, asr))
        shifts.append(shift); dists.append(dist); aucs.append(auc); asrs.append(asr)
        logger.info(f"  {name:14s} shift={shift:.3f} pair_dist={dist:.3f} paired_auc={auc:.3f} ASR={asr:.1f}")
        del m
        if device.type == "cuda": torch.cuda.empty_cache()

    print("\n=== P0b concept-axis vs ASR ===")
    print(f"{'method':14s} {'shift':>7s} {'pairdist':>9s} {'pairAUC':>8s} {'ASR':>6s}")
    for n, s, di, au, a in rows:
        print(f"{n:14s} {s:7.3f} {di:9.3f} {au:8.3f} {a:6.1f}")
    print(f"\nSpearman(shift, ASR)     = {_spearman(shifts, asrs):.3f}")
    print(f"Spearman(pair_dist, ASR) = {_spearman(dists, asrs):.3f}")
    print(f"Spearman(paired_auc, ASR)= {_spearman(aucs, asrs):.3f}  (n={len(rows)})")
    (DACE / "outputs" / "p0b_result.txt").write_text(
        "method,shift,pair_dist,paired_auc,asr\n"
        + "\n".join(f"{n},{s:.4f},{di:.4f},{au:.4f},{a:.1f}" for n, s, di, au, a in rows)
        + f"\nspearman_shift,{_spearman(shifts,asrs):.3f}\n"
        + f"spearman_pairdist,{_spearman(dists,asrs):.3f}\n"
        + f"spearman_pairauc,{_spearman(aucs,asrs):.3f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
