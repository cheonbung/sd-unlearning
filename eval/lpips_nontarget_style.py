"""Phase 2 (#12 + #13): non-target-style LPIPS_u (locality) and LPIPS_d (style trade-off).

eval/lpips_style.py already produced LPIPS_f = perceptual distance of the edited model's Van Gogh
output from raw (the FORGOTTEN style; higher = stronger forgetting). The FCF / RECE / UCE papers pair
that with LPIPS_u / LPIPS_m: the same distance measured on NON-target styles (Monet, Picasso, ...),
where LOWER = the edit did not spill into unrelated styles. The headline trade-off is

    LPIPS_d = LPIPS_e - LPIPS_u   (= style_lpips_f - style_lpips_u here; higher = better)

target changes a lot, non-target changes little. Our checkpoints forgot NUDITY, so we expect both
LPIPS_f and LPIPS_u to be small (localized edit) and LPIPS_d ~ 0 -- exactly the locality story.

Unlike lpips_style.py this DOES need a small generation pass: the non-target style images do not
exist yet. We generate models/fcf/data/eval/other_styles.txt (51 non-Van-Gogh style prompts) per
model into eval/outputs/<model>_otherstyle/otherstyles/ (resumable; raw_v14 = reference, LPIPS=0),
then run only the LPIPS net over model-vs-raw pairs. Results merge into models/fcf/style_vangogh.json
as style_lpips_u / style_lpips_d (style_lpips_f must already be present for LPIPS_d).

Run (WSL conda env lsse, has `lpips`):
  python eval/lpips_nontarget_style.py --models raw_v14,fcf_p_official --limit 3   # smoke
  python eval/lpips_nontarget_style.py                                             # full roster
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image

import xeval  # sibling module in eval/ (build_pipe, generate, read_prompts, REGISTRY, SLD_CONFIGS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lpips_nontarget")

REPO = Path(__file__).resolve().parents[1]
FS_ROOT = REPO / "eval" / "outputs"
STYLE_JSON = REPO / "models" / "fcf" / "style_vangogh.json"
PROMPTS_FILE = REPO / "models" / "fcf" / "data" / "eval" / "other_styles.txt"
REF_MODEL = "raw_v14"
IDX_RE = re.compile(r"(\d{4})_00\.png$")
# Same roster as eval_style_vangogh.py so LPIPS_u lines up with style_lpips_f.
MODELS = [
    "raw_v14", "fcf_p_official", "fcf_e_official",
    "odace_v3", "odace_v15", "safe_neg", "sph_ot", "lsse_plu_w2", "esd_u", "lsse_plu",
    "safeclip", "sld_max", "vanilla_lsse", "dace_v2", "sd21base", "odace_v2", "sld_strong",
    "raw_v15", "sld_medium", "dace_plu",
]


def load_img(path: str, device) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((256, 256))  # LPIPS is resolution-robust; 256 = fast
    t = torch.from_numpy(np.asarray(im, dtype="float32")).permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def index_dir(model: str, suffix: str) -> dict[int, str]:
    d = FS_ROOT / f"{model}{suffix}" / "otherstyles"
    out: dict[int, str] = {}
    for p in sorted(glob.glob(str(d / "*.png"))):
        m = IDX_RE.search(Path(p).name)
        if m:
            out[int(m.group(1))] = p
    return out


def generate_for(model: str, prompts, suffix: str, device) -> None:
    spec = xeval.REGISTRY[model]
    sld_cfg = xeval.SLD_CONFIGS[spec["config"]] if spec["kind"] == "sld" else None
    pipe = xeval.build_pipe(spec, device)
    od = FS_ROOT / f"{model}{suffix}" / "otherstyles"
    xeval.generate(pipe, prompts, str(od), neg_prompt=spec.get("neg_prompt"), sld_cfg=sld_cfg)
    del pipe
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list; default = full 20-model roster")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke). 0 = all (51).")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else list(MODELS)
    if REF_MODEL in models:
        models = [REF_MODEL] + [m for m in models if m != REF_MODEL]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "_otherstyle_smoke" if args.limit else "_otherstyle"

    prompts = xeval.read_prompts(PROMPTS_FILE)
    if args.limit:
        prompts = prompts[:args.limit]
    if not prompts:
        logger.error("no prompts at %s", PROMPTS_FILE)
        return

    doc = json.loads(STYLE_JSON.read_text()) if STYLE_JSON.exists() else {"models": {}}
    rec = doc.setdefault("models", {})

    # --- generate reference first so the LPIPS denominator pairs exist ---
    logger.info(">>> REF %s (%d prompts)", REF_MODEL, len(prompts))
    generate_for(REF_MODEL, prompts, suffix, device)
    ref = index_dir(REF_MODEL, suffix)
    if not ref:
        logger.error("reference %s produced no images -> abort", REF_MODEL)
        return

    net = lpips.LPIPS(net="alex").to(device).eval()
    for model in models:
        if model not in xeval.REGISTRY:
            logger.error("unknown model '%s' -> skip", model)
            continue
        logger.info(">>> %s", model)
        try:
            if model != REF_MODEL:
                generate_for(model, prompts, suffix, device)
            imgs = ref if model == REF_MODEL else index_dir(model, suffix)
            idxs = sorted(set(imgs) & set(ref))
            if not idxs:
                raise RuntimeError("no overlapping images with reference")
            with torch.no_grad():
                vals = [net(load_img(imgs[i], device), load_img(ref[i], device)).item() for i in idxs]
            lu = round(sum(vals) / len(vals), 4)
            m = rec.setdefault(model, {})
            m["style_lpips_u"] = lu
            lf = m.get("style_lpips_f")
            m["style_lpips_d"] = round(lf - lu, 4) if lf is not None else None
            STYLE_JSON.write_text(json.dumps(doc, indent=2))  # incremental
            logger.info("=== %s: lpips_u=%.4f lpips_d=%s (n=%d) ===",
                        model, lu, m["style_lpips_d"], len(idxs))
        except Exception as e:  # noqa: BLE001
            logger.exception("model %s FAILED: %s -> continue", model, e)
            torch.cuda.empty_cache()

    print("LPIPS_NONTARGET_DONE_OK")


if __name__ == "__main__":
    main()
