"""Art-style (Van Gogh) LOCALITY check -- companion to eval_violence_q16.py.

Our checkpoints forgot NUDITY, not artistic style. So this is NOT a style-erasure reproduction;
it is a specificity / locality probe: does nudity-unlearning preserve the model's ability to render
Van Gogh style? If the edit is localized, the Van Gogh images stay ~ identical to raw SD v1.4
(high image<->raw CLIP similarity) and keep a similar absolute style strength. A large drop would
mean the nudity edit spills into unrelated style rendering. Reported as locality, not erasure.

Metrics (self-contained CLIP ViT-L/14, same model Q16 uses; no parent/lsse code modified):
  - style_img2raw   PRIMARY. mean cos( CLIP(model_img_i), CLIP(raw_v14_img_i) ) over same
                    prompt+seed. higher = style/content preserved vs raw (1.0 = identical). raw_v14
                    is the reference (=1.0 by definition).
  - style_clip_text mean cos( CLIP(model_img_i), CLIP("an artwork in the style of Vincent van
                    Gogh") ). absolute style strength.
  - delta_text_vs_raw  style_clip_text(model) - style_clip_text(raw). ~0 = localized.

Prompt set: models/fcf/data/eval/vangogh_style.txt (50 named Van Gogh works -- canonical ESD/UCE
style set). Gen -> eval/outputs/<model>_style/vangogh/ (gitignored, resumable). Same deterministic
seed (SEED+i) as every other axis so images are comparable across models. Result ->
models/fcf/style_vangogh.json. raw_v14 is always embedded first to serve as the reference.

Run (WSL conda env lsse):
  python models/fcf/eval_style_vangogh.py --limit 3                 # smoke
  python models/fcf/eval_style_vangogh.py --models raw_v14,fcf_p_official
  python models/fcf/eval_style_vangogh.py                           # full locality check (20 models)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent           # models/fcf
REPO = HERE.parents[1]                          # SD_unlearning
sys.path.insert(0, str(REPO / "eval"))
import xeval  # noqa: E402  (build_pipe, generate, read_prompts, REGISTRY, SLD_CONFIGS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("style_vangogh")

CLIP_ID = "openai/clip-vit-large-patch14"
STYLE_TEXT = "an artwork in the style of Vincent van Gogh"
PROMPTS_FILE = HERE / "data" / "eval" / "vangogh_style.txt"
REF_MODEL = "raw_v14"                              # locality reference (img2raw denominator)
# Same Table-A roster as the violence axis -> one uniform cross-model style axis.
MODELS = [
    "raw_v14", "fcf_p_official", "fcf_e_official",
    "odace_v3", "odace_v15", "safe_neg", "sph_ot", "lsse_plu_w2", "esd_u", "lsse_plu",
    "safeclip", "sld_max", "vanilla_lsse", "dace_v2", "sd21base", "odace_v2", "sld_strong",
    "raw_v15", "sld_medium", "dace_plu",
]


class Clip:
    """Self-contained CLIP ViT-L/14 -> normalized 768-d joint-space embeddings for images and
    text. Robust to transformers 5.9 (get_image_features may return a ModelOutput, not a tensor)."""

    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_ID, use_safetensors=True).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)
        logger.info("CLIP loaded (self-contained) %s", CLIP_ID)

    @torch.no_grad()
    def text_embed(self, text: str) -> torch.Tensor:
        inputs = self.proc(text=[text], return_tensors="pt", padding=True).to(self.device)
        out = self.model.get_text_features(**inputs)
        t = out if torch.is_tensor(out) else getattr(out, "text_embeds", out.pooler_output)
        t = t.float()
        return (t / t.norm(dim=-1, keepdim=True)).squeeze(0)

    @torch.no_grad()
    def image_embed(self, image_path: str) -> torch.Tensor:
        from PIL import Image
        inputs = self.proc(images=Image.open(image_path).convert("RGB"),
                           return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        if torch.is_tensor(out):
            feats = out
        else:
            feats = getattr(out, "image_embeds", None)
            if feats is None:
                feats = out.pooler_output
        feats = feats.float()
        if feats.shape[-1] != self.model.config.projection_dim:   # raw pooled (1024) -> project
            feats = self.model.visual_projection(feats.to(self.model.dtype)).float()
        return (feats / feats.norm(dim=-1, keepdim=True)).squeeze(0)


def embed_dir(clip: Clip, d: Path) -> dict[int, torch.Tensor]:
    """index (from NNNN_00.png) -> normalized CLIP image embedding."""
    embeds = {}
    for p in sorted(d.glob("*.png")):
        try:
            idx = int(p.name[:4])
        except ValueError:
            continue
        try:
            embeds[idx] = clip.image_embed(str(p))
        except Exception as e:  # noqa: BLE001
            logger.warning("CLIP failed %s: %s", p, e)
    return embeds


def gen_and_embed(clip: Clip, model: str, prompts, suffix: str, device) -> dict[int, torch.Tensor]:
    spec = xeval.REGISTRY[model]
    sld_cfg = xeval.SLD_CONFIGS[spec["config"]] if spec["kind"] == "sld" else None
    pipe = xeval.build_pipe(spec, device)
    od = REPO / "eval" / "outputs" / f"{model}{suffix}" / "vangogh"
    xeval.generate(pipe, prompts, str(od), neg_prompt=spec.get("neg_prompt"), sld_cfg=sld_cfg)
    del pipe
    torch.cuda.empty_cache()
    return embed_dir(clip, od)


def load_result(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"_doc": "Van Gogh art-style LOCALITY check. style_img2raw: CLIP cos(model_img, raw_img) "
                    "same prompt+seed, higher=style preserved (1.0=identical, raw is reference). "
                    "style_clip_text: cos(img, 'van gogh style'). FCF/etc forgot NUDITY -> expect "
                    "img2raw~1 and delta_text~0 (no spill = localized edit).",
            "ref_model": REF_MODEL, "models": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list; default = full 20-model roster")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke). 0 = full (50).")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else list(MODELS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = Clip(device)
    suffix = "_style_smoke" if args.limit else "_style"
    out_path = HERE / ("style_vangogh_smoke.json" if args.limit else "style_vangogh.json")
    out = load_result(out_path)

    prompts = xeval.read_prompts(PROMPTS_FILE)
    if args.limit:
        prompts = prompts[:args.limit]
    if not prompts:
        logger.error("no prompts at %s", PROMPTS_FILE)
        return
    text_emb = clip.text_embed(STYLE_TEXT)

    # --- reference (raw_v14): always embed first; img2raw is measured against it ---
    logger.info(">>> REF %s (%d prompts)", REF_MODEL, len(prompts))
    ref_embeds = gen_and_embed(clip, REF_MODEL, prompts, suffix, device)
    if not ref_embeds:
        logger.error("reference %s produced no embeddings -> abort", REF_MODEL)
        return
    ref_text = torch.stack([e @ text_emb for e in ref_embeds.values()]).mean().item()

    for model in models:
        if model not in xeval.REGISTRY:
            logger.error("unknown model '%s' -> skip", model)
            continue
        logger.info(">>> START %s", model)
        try:
            embeds = ref_embeds if model == REF_MODEL else gen_and_embed(
                clip, model, prompts, suffix, device)
            idxs = sorted(set(embeds) & set(ref_embeds))
            if not idxs:
                raise RuntimeError("no overlapping images with reference")
            img2raw = torch.stack([embeds[i] @ ref_embeds[i] for i in idxs]).mean().item()
            text_cos = torch.stack([embeds[i] @ text_emb for i in idxs]).mean().item()
            rec = {"n": len(idxs),
                   "style_img2raw": round(img2raw, 4),
                   "style_clip_text": round(text_cos, 4),
                   "delta_text_vs_raw": round(text_cos - ref_text, 4)}
            out["models"][model] = rec
            out_path.write_text(json.dumps(out, indent=2))   # incremental save
            logger.info("=== %s done: img2raw=%.4f text=%.4f dtext=%+.4f (n=%d, saved) ===",
                        model, rec["style_img2raw"], rec["style_clip_text"],
                        rec["delta_text_vs_raw"], rec["n"])
        except Exception as e:  # noqa: BLE001
            logger.exception("model %s FAILED: %s -> continue", model, e)
            out["models"].setdefault(model, {"error": str(e)})
            out_path.write_text(json.dumps(out, indent=2))
            torch.cuda.empty_cache()

    logger.info("wrote %s", out_path)
    print("STYLE_VANGOGH_DONE_OK")


if __name__ == "__main__":
    main()
