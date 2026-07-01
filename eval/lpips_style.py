"""Paper-aligned art-style LPIPS_f over the ALREADY-GENERATED Van Gogh images (no SD regeneration).

eval_style_vangogh.py generated Van Gogh images to eval/outputs/<model>_style/vangogh/NNNN_00.png and
scored a CLIP-cosine locality metric (style_img2raw). The FCF paper (Table 2) instead reports an LPIPS
art-style metric: LPIPS_f = perceptual distance of the unlearned model's output from the reference on
the FORGOTTEN style (higher = stronger forgetting). This script recovers LPIPS_f post-hoc by running
ONLY the LPIPS network over the retained image pairs (model_img_i vs raw_v14_img_i, same prompt+seed),
so no diffusion generation is needed. raw_v14 is the reference (LPIPS_f = 0 by definition).

NOTE: the paper also reports LPIPS_m (retention of NON-target styles). That needs a separate non-Van-Gogh
style prompt set we do not generate yet, so only LPIPS_f is filled here; LPIPS_m is left for a follow-up.

Run (WSL conda env lsse, has the `lpips` package):
  python eval/lpips_style.py
Merges "style_lpips_f" into each model record of models/fcf/style_vangogh.json.
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FS_ROOT = REPO / "eval" / "outputs"
STYLE_JSON = REPO / "models" / "fcf" / "style_vangogh.json"
REF_MODEL = "raw_v14"
IDX_RE = re.compile(r"(\d{4})_00\.png$")


def load_img(path: str, device) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((256, 256))  # LPIPS is resolution-robust; 256 = fast
    t = torch.from_numpy(np.asarray(im, dtype="float32")).permute(2, 0, 1) / 127.5 - 1.0  # -> [-1,1]
    return t.unsqueeze(0).to(device)


def index_dir(model: str) -> dict[int, str]:
    d = FS_ROOT / f"{model}_style" / "vangogh"
    out: dict[int, str] = {}
    if not d.exists():
        return out
    for p in sorted(glob.glob(str(d / "*.png"))):
        m = IDX_RE.search(Path(p).name)
        if m:
            out[int(m.group(1))] = p
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lpips.LPIPS(net="alex").to(device).eval()
    ref = index_dir(REF_MODEL)
    if not ref:
        print(f"[abort] no reference images at {REF_MODEL}_style/vangogh")
        return

    doc = json.loads(STYLE_JSON.read_text()) if STYLE_JSON.exists() else {"models": {}}
    models = doc.setdefault("models", {})
    # every model that has a JSON record OR an on-disk image dir (<model>_style/vangogh)
    disk = {Path(p).parent.name for p in glob.glob(str(FS_ROOT / "*_style"))}
    disk = {d[:-6] for d in disk if d.endswith("_style")}
    keys = sorted(set(models) | disk)
    for model in keys:
        imgs = ref if model == REF_MODEL else index_dir(model)
        idxs = sorted(set(imgs) & set(ref))
        if not idxs:
            print(f"[skip] {model}: no overlapping images")
            continue
        with torch.no_grad():
            vals = [net(load_img(imgs[i], device), load_img(ref[i], device)).item() for i in idxs]
        lf = round(sum(vals) / len(vals), 4)
        models.setdefault(model, {})["style_lpips_f"] = lf
        STYLE_JSON.write_text(json.dumps(doc, indent=2))  # incremental
        print(f"[done] {model}: style_lpips_f={lf} (n={len(idxs)})", flush=True)
    print("LPIPS_STYLE_DONE", flush=True)


if __name__ == "__main__":
    main()
