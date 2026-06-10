"""Standard locality protocol -- COCO retain FID + CLIP.

Replaces the discarded custom probes (CLIP_atk / people_clip / people_asr). This is the
field-standard specificity/locality axis used by concept-erasure work (ESD/UCE/SLD-style):
generate from COCO captions (no nudity intent -> no "removed-nudity vs off-target" confound)
and report FID (vs real COCO) + CLIP(image, caption). Efficacy (ASR) is read from the existing
xmodel metrics.json. Lower FID = closer to real distribution; higher CLIP = better caption
alignment; both should stay near raw if erasure is localized.

Data is fetched once from cocodataset.org (ungated): captions_val2017.json + N real val images.
300 captions/model generated (user-chosen "light" scope); FID vs 600 real images.

Run (WSL conda env lsse):
  python eval_coco.py --prep_only                 # download + parse COCO once
  python eval_coco.py --models raw_v14,raw_v15,safe_neg,odace_v3,odace_v15
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xeval  # noqa: E402  (build_pipe, generate, REGISTRY, read_prompts)
from evaluation import FIDCLIPEvaluator  # noqa: E402  (lsse on path via xeval)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_coco")

COCO = HERE / "data" / "coco"
CAP_FILE = COCO / "captions300.txt"
REAL_DIR = COCO / "real"

# --- LPIPS + CLIP-IQA (paper-grounded extra metrics) -----------------------------------------
# LPIPS (ESD/FCF): perceptual distance between the EDITED model and the same-base RAW model on the
#   SAME COCO caption+seed -> "edit drift" locality (lower = generation better preserved).
# CLIP-IQA (no-reference image quality): P(image is "Good photo." vs "Bad photo.") in CLIP space,
#   averaged over the generated set (higher = better quality). Uses a small CLIP, no new install.
_LPIPS = None
_IQA = None  # (model, processor)


def _ref_label_for(spec) -> str | None:
    base = str(spec.get("base", spec.get("model_id", "")))
    if "v1-5" in base or "v1.5" in base:
        return "raw_v15"
    if "2-1" in base or "v2-1" in base:
        return None                # no same-base raw available -> LPIPS undefined
    return "raw_v14"


@torch.no_grad()
def compute_lpips(gen_dir, ref_dir, device) -> float | None:
    global _LPIPS
    import numpy as np
    import lpips as lpips_lib
    from PIL import Image
    if _LPIPS is None:
        _LPIPS = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()

    def _load(p):
        im = Image.open(p).convert("RGB").resize((256, 256))
        t = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 127.5 - 1.0
        return t.unsqueeze(0).to(device)

    vals = []
    for g in sorted(Path(gen_dir).glob("*.png")):
        r = Path(ref_dir) / g.name
        if not r.exists():
            continue
        vals.append(_LPIPS(_load(g), _load(r)).item())
    return round(sum(vals) / len(vals), 4) if vals else None


@torch.no_grad()
def compute_clip_iqa(gen_dir, device) -> float | None:
    global _IQA
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    if _IQA is None:
        # use_safetensors avoids transformers 5.9 refusing .bin on torch<2.6; large-patch14 is the
        # same CLIP the FID/CLIP scorer already loads, so it is cached and has safetensors.
        cid = "openai/clip-vit-large-patch14"
        m = CLIPModel.from_pretrained(cid, use_safetensors=True).to(device).eval()
        p = CLIPProcessor.from_pretrained(cid)
        _IQA = (m, p)
    model, proc = _IQA
    prompts = ["Good photo.", "Bad photo."]   # CLIP-IQA antonym pair
    scores = []
    for ip in sorted(Path(gen_dir).glob("*.png")):
        inp = proc(text=prompts, images=Image.open(ip).convert("RGB"),
                   return_tensors="pt", padding=True).to(device)
        out = model(**inp)                    # CLIPOutput: image_embeds / text_embeds (projected)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        cos = (ie @ te.T)[0]                  # (2,) cosine to Good/Bad
        scores.append(cos.softmax(0)[0].item())
    return round(sum(scores) / len(scores), 4) if scores else None
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"
N_GEN = 300
N_REAL = 600


def ensure_coco():
    COCO.mkdir(parents=True, exist_ok=True)
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    have_caps = CAP_FILE.exists()
    have_real = len(list(REAL_DIR.glob("*.jpg"))) >= N_REAL
    if have_caps and have_real:
        logger.info("COCO data already present.")
        return

    capjson = COCO / "captions_val2017.json"
    if not capjson.exists():
        zpath = COCO / "annotations_trainval2017.zip"
        if not zpath.exists():
            logger.info("downloading annotations (~250MB) ...")
            urllib.request.urlretrieve(ANN_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            with z.open("annotations/captions_val2017.json") as src, open(capjson, "wb") as dst:
                dst.write(src.read())
    data = json.loads(capjson.read_text())
    seen, pairs = set(), []
    for a in sorted(data["annotations"], key=lambda x: (x["image_id"], x["id"])):
        iid = a["image_id"]
        if iid in seen:
            continue
        seen.add(iid)
        pairs.append((iid, " ".join(a["caption"].split())))
        if len(pairs) >= N_REAL:
            break

    if not have_caps:
        CAP_FILE.write_text("\n".join(c for _, c in pairs[:N_GEN]) + "\n", encoding="utf-8")
        logger.info(f"wrote {N_GEN} captions -> {CAP_FILE}")

    if not have_real:
        def fetch(iid):
            fp = REAL_DIR / f"{iid:012d}.jpg"
            if fp.exists():
                return
            try:
                urllib.request.urlretrieve(IMG_URL.format(iid), fp)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"img {iid} failed: {e}")
        ids = [iid for iid, _ in pairs[:N_REAL]]
        logger.info(f"downloading {len(ids)} real COCO images ...")
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(fetch, ids))
        logger.info(f"real images: {len(list(REAL_DIR.glob('*.jpg')))}")


def existing_asr(label):
    f = HERE / "outputs" / label / "metrics.json"
    if f.exists():
        return json.loads(f.read_text()).get("asr_mean")
    return None


def run(label, device, clip_ev):
    spec = xeval.REGISTRY[label]
    captions = xeval.read_prompts(CAP_FILE)
    out_dir = HERE / "outputs" / label / "coco"
    pipe = xeval.build_pipe(spec, device)
    sld_cfg = xeval.SLD_CONFIGS[spec["config"]] if spec["kind"] == "sld" else None
    xeval.generate(pipe, captions, str(out_dir), neg_prompt=spec.get("neg_prompt"), sld_cfg=sld_cfg)
    clip = clip_ev.compute_clip_score(str(out_dir), captions)
    fid = clip_ev.compute_fid(str(out_dir), str(REAL_DIR))
    del pipe
    torch.cuda.empty_cache()
    # LPIPS vs same-base raw (edit drift) + no-reference CLIP-IQA image quality.
    ref_label = _ref_label_for(spec)
    ref_coco = (HERE / "outputs" / ref_label / "coco") if ref_label else None
    lpips_v = compute_lpips(str(out_dir), str(ref_coco), device) \
        if (ref_coco and ref_coco.exists()) else None
    iq = compute_clip_iqa(str(out_dir), device)
    m = {"label": label, "n_gen": len(captions), "n_real": len(list(REAL_DIR.glob("*.jpg"))),
         "coco_clip": round(clip, 2), "coco_fid": round(fid, 2),
         "coco_lpips": lpips_v, "coco_iq": iq, "ref_label": ref_label,
         "asr_mean": existing_asr(label)}
    (HERE / "outputs" / label / "coco_metrics.json").write_text(json.dumps(m, indent=2))
    logger.info(f"=== {label} === ASR {m['asr_mean']} | FID {m['coco_fid']} | CLIP {m['coco_clip']} "
                f"| LPIPS {lpips_v} | IQ {iq}")
    torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--prep_only", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ensure_coco()
    if args.prep_only:
        logger.info("prep done.")
        return

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    clip_ev = FIDCLIPEvaluator(device=device)
    rows = {}
    for label in [s.strip() for s in args.models.split(",") if s.strip()]:
        if label not in xeval.REGISTRY:
            logger.error(f"unknown model {label}")
            continue
        rows[label] = run(label, device, clip_ev)
    print("\n=== COCO SUMMARY (ASR / FID / CLIP) ===")
    for label, m in rows.items():
        print(f"{label:12s} ASR={m['asr_mean']}  FID={m['coco_fid']}  CLIP={m['coco_clip']}")


if __name__ == "__main__":
    main()
