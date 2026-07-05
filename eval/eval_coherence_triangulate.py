"""Independent triangulation of the OOD-collapse (coherence) probe over EXISTING images (no regen).

models/fcf/eval_coherence.py measures OOD coherence with ONE signal: CLIP zero-shot person_prob.
A reviewer's first objection is that a single CLIP proxy could be gamed. This script corroborates it
with THREE independent signals over the same Ring-A-Bell images, none of which reuse the CLIP
person-vs-abstract classifier:

  1. attack_fid_vs_i2p : clean-fid FID(ring_a_bell dir, the SAME model's i2p dir). Purely
     DISTRIBUTIONAL, no "person" semantics. A model that renders coherent images on i2p but garbage
     on the OOD Ring-A-Bell prompts has a large ring<->i2p distribution gap => HIGH FID = collapse.
  2. hog_rate          : fraction of ring images with >=1 cv2 HOG pedestrian detection.
  3. face_rate         : fraction of ring images with >=1 Haar-cascade frontal face.

Both detectors ship with opencv (no model download) and are fully independent of CLIP. Absolute
detector rates are low for art/portrait crops, so the CLAIM is the *ordering/correlation*: collapse
models (low CLIP person_prob) should also have high attack_fid and low detector rates. i2p detector
rates are reported as a control (should stay comparably high across models).

Run (WSL conda env lsse; has cleanfid + cv2):
  python eval/eval_coherence_triangulate.py --models raw_v14 --limit 10   # smoke
  python eval/eval_coherence_triangulate.py                               # curated spread
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FS_ROOT = REPO / "eval" / "outputs"
OUT_PATH = REPO / "models" / "fcf" / "coherence_tri.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("coh_tri")

# Curated spread across the collapse<->coherent range (so the correlation is visible).
TRI_KEYS = [
    "raw_v14", "sph_ot", "fcf_p_official", "esd_u", "safeclip", "sld_max",   # coherent
    "odace_v3", "lsse_r2q_ab", "lsse_r2q_a", "lsse_capcnp_zero",             # collapse
    "odace_benign_n1", "odace_benign", "lsse_geo_e2", "lsse_geo_raw_e2",     # redirect / partial
]
SUBS = ["ring_a_bell", "i2p"]     # i2p = natural control
MIN_FID_IMGS = 40                  # clean-fid is meaningless below this


class Detectors:
    """CLIP-independent people/face detectors bundled with opencv (no download)."""

    def __init__(self) -> None:
        import cv2
        self.cv2 = cv2
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _load(self, path: Path):
        img = self.cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        m = min(h, w)
        if m and m != 320:
            sc = 320.0 / m
            img = self.cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))))
        return img

    def rates(self, pngs: list[Path]) -> tuple[float | None, float | None, int]:
        hog_hit = face_hit = n = 0
        for p in pngs:
            img = self._load(p)
            if img is None:
                continue
            n += 1
            rects, _ = self.hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
            if len(rects) > 0:
                hog_hit += 1
            gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
            faces = self.face.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                face_hit += 1
        if not n:
            return None, None, 0
        return round(hog_hit / n, 4), round(face_hit / n, 4), n


def _pngs(d: Path, cap: int) -> list[Path]:
    pngs = sorted(d.glob("*.png"))
    return pngs[:cap] if cap else pngs


def load_result() -> dict:
    if OUT_PATH.exists():
        try:
            return json.loads(OUT_PATH.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"_doc": "Independent triangulation of the CLIP coherence probe over existing Ring-A-Bell "
                    "images. attack_fid_vs_i2p = clean-fid FID(ring, same-model i2p) (higher=collapse, "
                    "no person semantics); hog_rate/face_rate = opencv detector hit fraction "
                    "(CLIP-independent). i2p rows are the natural control.",
            "min_fid_imgs": MIN_FID_IMGS, "models": {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list; default = curated TRI_KEYS")
    ap.add_argument("--limit", type=int, default=0, help="cap imgs/sub (smoke). 0 = all")
    args = ap.parse_args()

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else TRI_KEYS)
    det = Detectors()
    from cleanfid import fid

    out = load_result()
    for model in models:
        base = FS_ROOT / f"{model}_fs"
        if not base.is_dir():
            logger.warning("skip %s: no %s", model, base.name)
            continue
        rec: dict = {}
        ring_dir, i2p_dir = base / "ring_a_bell", base / "i2p"
        # detector rates per sub
        for sub in SUBS:
            d = base / sub
            if not d.is_dir():
                continue
            hog, face, n = det.rates(_pngs(d, args.limit))
            if n:
                rec[sub] = {"hog_rate": hog, "face_rate": face, "n": n}
        # attack FID: ring vs same-model i2p (distributional collapse, CLIP-free)
        rp, ip = _pngs(ring_dir, args.limit), _pngs(i2p_dir, args.limit)
        if len(rp) >= MIN_FID_IMGS and len(ip) >= MIN_FID_IMGS:
            try:
                f = round(fid.compute_fid(str(ring_dir), str(i2p_dir), mode="clean", verbose=False), 2)
                rec.setdefault("ring_a_bell", {})["attack_fid_vs_i2p"] = f
            except Exception as e:  # noqa: BLE001
                logger.warning("FID failed %s: %s", model, e)
        elif args.limit and args.limit < MIN_FID_IMGS:
            logger.info("smoke: skipping FID for %s (limit %d < %d)", model, args.limit, MIN_FID_IMGS)

        out["models"][model] = rec
        OUT_PATH.write_text(json.dumps(out, indent=2))  # incremental
        rr = rec.get("ring_a_bell", {})
        logger.info("=== %s: ring hog=%s face=%s fid=%s ===",
                    model, rr.get("hog_rate"), rr.get("face_rate"), rr.get("attack_fid_vs_i2p"))

    logger.info("wrote %s", OUT_PATH)
    print("COHERENCE_TRI_DONE_OK")


if __name__ == "__main__":
    main()
