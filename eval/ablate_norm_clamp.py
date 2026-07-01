"""Phase 0 (no-retrain GATE): inference norm-clamp ablation for the Ring-A-Bell OOD collapse.

Hypothesis: LSSE collapses on OOD gibberish prompts because the read-out-space erasure leaves the
conditioning (text-encoder last_hidden_state) off the CLIP manifold / with degenerate magnitude, so
the UNet decodes garbage. Test: keep the edited TE's DIRECTION but restore the RAW TE's per-token
magnitude -- rescale edited last_hidden_state_i to ||raw last_hidden_state_i||. If person-presence
recovers on Ring-A-Bell, magnitude/manifold is the cause (-> Phase 1 manifold-preserving erasure).

We monkeypatch the swapped TE forward in the te_swap pipe (eval/xeval.py:222) to run BOTH the edited
TE and a raw SD1.4 TE on the same input_ids and rescale. Images -> eval/outputs/lsse_r2q_ab_clamp_fs/
<sub>/ so the existing coherence probe can score the key directly.

Run (WSL conda env lsse):
  python eval/ablate_norm_clamp.py --limit 2     # smoke
  python eval/ablate_norm_clamp.py               # ring_a_bell 95 + i2p 50
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import CLIPTextModel

import xeval  # sibling (REGISTRY, build_pipe, generate, read_prompts)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("norm_clamp")

REPO = Path(__file__).resolve().parents[1]
FS_ROOT = REPO / "eval" / "outputs"
EVAL_DIR = REPO / "models" / "fcf" / "data" / "eval"
TARGET = "lsse_r2q_ab"
OUT_KEY = "lsse_r2q_ab_clamp"
ATTACKS = [("ring_a_bell", "ring_a_bell_nudity.txt", 95), ("i2p", "i2p_nudity.txt", 50)]
EPS = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap prompts per attack (smoke). 0 = per-ATTACK cap.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = xeval.REGISTRY[TARGET]
    pipe = xeval.build_pipe(spec, device)
    dtype = pipe.text_encoder.dtype

    raw_te = CLIPTextModel.from_pretrained(spec["base"], subfolder="text_encoder").to(
        device=device, dtype=dtype).eval()
    edited_forward = pipe.text_encoder.forward  # bound method of the edited TE

    def clamp_forward(*a, **kw):
        out = edited_forward(*a, **kw)            # edited TE output (keeps DIRECTION)
        with torch.no_grad():
            raw_out = raw_te(*a, **kw)            # raw TE output (donor of MAGNITUDE)
        e = out.last_hidden_state
        rn = raw_out.last_hidden_state.norm(dim=-1, keepdim=True)
        en = e.norm(dim=-1, keepdim=True)
        out.last_hidden_state = e * (rn / (en + EPS))
        return out

    pipe.text_encoder.forward = clamp_forward
    logger.info("norm-clamp wrapper installed on %s TE (donor=raw %s)", TARGET, spec["base"])

    for sub, pfile, cap in ATTACKS:
        prompts = xeval.read_prompts(EVAL_DIR / pfile)
        prompts = prompts[:(args.limit if args.limit else cap)]
        od = FS_ROOT / f"{OUT_KEY}_fs" / sub
        logger.info(">>> %s: %d prompts -> %s", sub, len(prompts), od)
        xeval.generate(pipe, prompts, str(od), neg_prompt=spec.get("neg_prompt"), sld_cfg=None)

    print("CLAMP_ABLATE_DONE_OK")


if __name__ == "__main__":
    main()
