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
    xeval.generate(pipe, captions, str(out_dir), neg_prompt=spec.get("neg_prompt"))
    clip = clip_ev.compute_clip_score(str(out_dir), captions)
    fid = clip_ev.compute_fid(str(out_dir), str(REAL_DIR))
    m = {"label": label, "n_gen": len(captions), "n_real": len(list(REAL_DIR.glob("*.jpg"))),
         "coco_clip": round(clip, 2), "coco_fid": round(fid, 2), "asr_mean": existing_asr(label)}
    (HERE / "outputs" / label / "coco_metrics.json").write_text(json.dumps(m, indent=2))
    logger.info(f"=== {label} === ASR {m['asr_mean']} | COCO-FID {m['coco_fid']} | COCO-CLIP {m['coco_clip']}")
    del pipe
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
