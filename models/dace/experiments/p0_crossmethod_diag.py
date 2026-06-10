"""P0 -- cross-method diagnostic validation of the DACE hypothesis (cheap, no generation).

Hypothesis: the best linear separability of forget-vs-retain embeddings (the quantity
DACE minimizes) tracks adversarial ASR ACROSS methods -- i.e. erasing the separating
subspace is the right objective. If true, DACE's minimax target is justified.

For each existing checkpoint we compute linear_separability_auc on the SAME forget/retain
prompts and correlate (Spearman) with its known single-harness mean ASR.

Run (WSL conda env lsse):  python experiments/p0_crossmethod_diag.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

DACE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DACE))

from methods.adversary import pool
from methods import diagnostics as diag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p0")

MODEL_ID = "openai/clip-vit-large-patch14"

# name -> (checkpoint dir or HF id, known single-harness mean ASR %)
CKPTS = {
    "raw_sd":       (MODEL_ID, 62.0),
    "fcf_e":        (str(REPO / "fcf/outputs/fcf_e_nudity/final"), 61.2),
    "fcf_p":        (str(REPO / "fcf/outputs/fcf_p_nudity/final"), 52.8),
    "vanilla_lsse": (str(REPO / "lsse/outputs/sweep/baseline_seed42/final"), 46.0),
    "lsse_plu":     (str(REPO / "lsse/outputs/sweep/plu_seed42/final"), 21.6),
    "lsse_plu_w2":  (str(REPO / "lsse/outputs/sweep/stack_plu_w2_seed42/final"), 20.8),
    "sph_ot":       (str(REPO / "models/novel/outputs/fcf_p_v2_nudity_spherical_ot/final"), 15.6),
}


def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy + 1e-12)


@torch.no_grad()
def encode(encoder, tok, prompts, device, bs=8):
    out = []
    for i in range(0, len(prompts), bs):
        t = tok(prompts[i:i + bs], padding="max_length", max_length=77,
                truncation=True, return_tensors="pt").to(device)
        out.append(pool(encoder(t.input_ids).last_hidden_state, "mean").float().cpu())
    return torch.cat(out, dim=0)


def load_prompts(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def main():
    from transformers import CLIPTextModel, CLIPTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CLIPTokenizer.from_pretrained(MODEL_ID)
    forget = load_prompts(DACE / "data/prompts/nudity_explicit.txt")
    retain = load_prompts(DACE / "data/prompts/nudity_maintain.txt")
    logger.info(f"forget={len(forget)} retain={len(retain)} device={device}")

    rows, aucs, asrs = [], [], []
    for name, (path, asr) in CKPTS.items():
        if path != MODEL_ID and not Path(path).exists():
            logger.warning(f"SKIP {name}: missing {path}")
            continue
        enc = CLIPTextModel.from_pretrained(path).to(device).eval()
        zf = encode(enc, tok, forget, device)
        zr = encode(enc, tok, retain, device)
        auc = diag.linear_separability_auc(zf, zr)
        rows.append((name, auc, asr))
        aucs.append(auc); asrs.append(asr)
        logger.info(f"  {name:14s} sep_auc={auc:.4f}  meanASR={asr:.1f}")
        del enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rho = _spearman(aucs, asrs)
    print("\n=== P0 cross-method separability vs ASR ===")
    print(f"{'method':14s} {'sep_auc':>8s} {'meanASR':>8s}")
    for name, auc, asr in rows:
        print(f"{name:14s} {auc:8.4f} {asr:8.1f}")
    print(f"\nSpearman(sep_auc, meanASR) = {rho:.3f}  (n={len(rows)})")
    print("HYPOTHESIS SUPPORTED" if rho > 0.5 else "HYPOTHESIS WEAK/REFUTED")
    out = DACE / "outputs" / "p0_result.txt"
    out.write_text(
        "method,sep_auc,mean_asr\n"
        + "\n".join(f"{n},{a:.4f},{s:.1f}" for n, a, s in rows)
        + f"\nspearman,{rho:.3f},n={len(rows)}\n",
        encoding="utf-8",
    )
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
