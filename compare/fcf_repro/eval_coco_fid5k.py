"""Phase 4: COCO FID-5K -- put our FCF reproduction onto the paper's FID scale.

xmodel/eval_coco.py uses N_GEN=300 / N_REAL=600. That tiny sample size is exactly why its FID sits
at ~118 (small-N FID is heavily biased upward), so it is NOT comparable to the FCF paper's COCO-30K
Table 3 (FCF-P FID 15.07 / CLIP 31.03). This script raises N to 5000 (val2017 has exactly 5000
images) so FID lands on the paper's ~15 scale, then reports FID + CLIP(x100) for the official FCF
reproduction vs raw SD v1.4. CLIP is directly comparable (x100 ~31); FID matches the SCALE (absolute
value still depends on refset choice + clean-fid's inception, so we compare to the paper as a scale
check, not an exact-match claim).

Reuses xeval.build_pipe/generate (resumable), FIDCLIPEvaluator (clean-fid), and eval_coco's LPIPS /
CLIP-IQA helpers. Robust for a multi-hour run: per-model try/except, incremental JSON, resumable
generation + resumable real-image download.

Data: xmodel/data/coco/captions_val2017.json (already present) -> captions{N}.txt + real{N}/ images.
Gen -> xmodel/outputs/<model>_coco5k{tag}/ (gitignored). Result -> compare/fcf_repro/coco5k{tag}.json.

Run (WSL conda env lsse):
  python compare/fcf_repro/eval_coco_fid5k.py --prep_only                  # download 5000 real imgs
  python compare/fcf_repro/eval_coco_fid5k.py --n 24 --tag _smoke          # smoke test
  python compare/fcf_repro/eval_coco_fid5k.py                              # full FID-5K (P/E/raw)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent           # compare/fcf_repro
REPO = HERE.parent.parent                          # SD_unlearning
sys.path.insert(0, str(REPO / "xmodel"))
import xeval  # noqa: E402  (build_pipe, generate, read_prompts, REGISTRY, SLD_CONFIGS)
from eval_coco import compute_lpips, compute_clip_iqa  # noqa: E402  (dir-parametrized helpers)
from evaluation import FIDCLIPEvaluator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("coco_fid5k")

COCO = REPO / "xmodel" / "data" / "coco"
CAP_JSON = COCO / "captions_val2017.json"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"
MODELS = ["raw_v14", "fcf_p_official", "fcf_e_official"]   # raw first -> LPIPS ref exists


def coco_pairs(n: int):
    """Return n (image_id, caption) pairs, one caption per unique val2017 image."""
    if not CAP_JSON.exists():
        zpath = COCO / "annotations_trainval2017.zip"
        if not zpath.exists():
            logger.info("downloading COCO annotations (~250MB) ...")
            urllib.request.urlretrieve(ANN_URL, zpath)
        with zipfile.ZipFile(zpath) as z, z.open("annotations/captions_val2017.json") as src:
            CAP_JSON.write_bytes(src.read())
    data = json.loads(CAP_JSON.read_text())
    seen, pairs = set(), []
    for a in sorted(data["annotations"], key=lambda x: (x["image_id"], x["id"])):
        iid = a["image_id"]
        if iid in seen:
            continue
        seen.add(iid)
        pairs.append((iid, " ".join(a["caption"].split())))
        if len(pairs) >= n:
            break
    return pairs


def ensure_coco(n: int, tag: str):
    """Write captions{n}{tag}.txt and download n real images -> real{n}{tag}/. Resumable."""
    COCO.mkdir(parents=True, exist_ok=True)
    cap_file = COCO / f"captions{n}{tag}.txt"
    real_dir = COCO / f"real{n}{tag}"
    real_dir.mkdir(parents=True, exist_ok=True)
    pairs = coco_pairs(n)
    if not cap_file.exists():
        cap_file.write_text("\n".join(c for _, c in pairs) + "\n", encoding="utf-8")
        logger.info("wrote %d captions -> %s", len(pairs), cap_file)
    have = len(list(real_dir.glob("*.jpg")))
    if have < len(pairs):
        def fetch(iid):
            fp = real_dir / f"{iid:012d}.jpg"
            if fp.exists():
                return
            try:
                urllib.request.urlretrieve(IMG_URL.format(iid), fp)
            except Exception as e:  # noqa: BLE001
                logger.warning("img %s failed: %s", iid, e)
        ids = [iid for iid, _ in pairs]
        logger.info("downloading %d real COCO images (have %d) ...", len(ids), have)
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(fetch, ids))
    n_real = len(list(real_dir.glob("*.jpg")))
    logger.info("COCO ready: %d captions, %d real images", len(pairs), n_real)
    return cap_file, real_dir


def load_result(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"_doc": "Phase 4 COCO FID-5K. FID lower=better; CLIP x100 higher=better; LPIPS vs "
                    "same-base raw; IQ=CLIP-IQA. Paper Table 3 FCF-P FID 15.07 / CLIP 31.03.",
            "models": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="images/model and real refset size")
    ap.add_argument("--tag", default="", help="output suffix for smoke runs, e.g. _smoke")
    ap.add_argument("--models", default=None, help="comma list; default raw_v14,fcf_p/e_official")
    ap.add_argument("--prep_only", action="store_true")
    args = ap.parse_args()

    cap_file, real_dir = ensure_coco(args.n, args.tag)
    if args.prep_only:
        logger.info("prep done.")
        return

    models = [m.strip() for m in args.models.split(",")] if args.models else MODELS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_ev = FIDCLIPEvaluator(device=device)
    captions = xeval.read_prompts(cap_file)
    out_path = HERE / f"coco5k{args.tag}.json"
    out = load_result(out_path)

    for model in models:
        if model not in xeval.REGISTRY:
            logger.error("unknown model '%s' -> skip", model)
            continue
        logger.info(">>> START %s (n=%d)", model, args.n)
        try:
            spec = xeval.REGISTRY[model]
            out_dir = REPO / "xmodel" / "outputs" / f"{model}_coco5k{args.tag}"
            pipe = xeval.build_pipe(spec, device)
            sld_cfg = xeval.SLD_CONFIGS[spec["config"]] if spec["kind"] == "sld" else None
            xeval.generate(pipe, captions, str(out_dir), neg_prompt=spec.get("neg_prompt"),
                           sld_cfg=sld_cfg)
            clip = clip_ev.compute_clip_score(str(out_dir), captions)
            fid = clip_ev.compute_fid(str(out_dir), str(real_dir))
            del pipe
            torch.cuda.empty_cache()
            ref = REPO / "xmodel" / "outputs" / f"raw_v14_coco5k{args.tag}"
            lpips_v = compute_lpips(str(out_dir), str(ref), device) \
                if (model != "raw_v14" and ref.exists()) else None
            iq = compute_clip_iqa(str(out_dir), device)
            rec = {"n_gen": len(captions), "n_real": len(list(real_dir.glob("*.jpg"))),
                   "coco_fid": round(fid, 2), "coco_clip": round(clip, 2),
                   "coco_lpips": lpips_v, "coco_iq": iq}
            out["models"][model] = rec
            out_path.write_text(json.dumps(out, indent=2))   # incremental save
            logger.info("=== %s done: FID %.2f | CLIP %.2f | LPIPS %s | IQ %s (saved) ===",
                        model, fid, clip, lpips_v, iq)
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            logger.exception("model %s FAILED: %s -> continue", model, e)
            out["models"].setdefault(model, {"error": str(e)})
            out_path.write_text(json.dumps(out, indent=2))
            torch.cuda.empty_cache()

    logger.info("wrote %s", out_path)
    print("COCO_FID5K_DONE_OK")


if __name__ == "__main__":
    main()
