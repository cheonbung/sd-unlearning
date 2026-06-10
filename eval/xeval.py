"""Cross-model nudity-erasure efficacy eval (ASR only).

Efficacy is measured with the field-standard ASR (NudeNet v3 nudity rate on adversarial attack
images, lower = safer). LOCALITY / specificity is NOT measured here -- it lives in the standard
COCO protocol (`eval_coco.py`: FID + CLIP on general COCO captions).

  NOTE: earlier revisions of this file also computed CLIP_atk / CLIP_neu and a custom benign
  "people" probe (people_clip / people_asr). Those were non-standard, hand-rolled metrics with a
  confound (CLIP on an attack prompt drops both when nudity is removed *and* when content is
  destroyed, so it cannot separate the two). They were discarded by user instruction; locality is
  now exclusively the COCO FID/CLIP axis. Do not reintroduce them here.

Models live in REGISTRY. Notes on safe baselines:
  - SD2.x (NSFW-filtered pretraining) moved around on HF; REGISTRY keeps the original
    Stability AI id plus public mirrors as fallbacks.
  - The diffusers 0.38 build here removed the SLD pipeline, so "safe_neg" approximates an
    inference-time safety baseline via a fixed nudity NEGATIVE PROMPT on SD-v1.5.

Images for already-evaluated models are reused via "attack_dir"; everything else is generated
fresh at 512 / 50 steps / guidance 7.5 / per-prompt deterministic seed (same latent across
models). Results -> eval/outputs/<label>/metrics.json. Gallery (Gate E) reads these dirs.

Run (WSL conda env lsse):
  python xeval.py --models raw_v14,odace_v3
  python xeval.py --models raw_v15,safe_neg
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

try:                                  # progress bars with ETA (optional dep)
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:  # noqa: BLE001
    _HAS_TQDM = False

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LSSE = REPO / "models" / "core" / "lsse"
sys.path.insert(0, str(LSSE)); sys.path.insert(0, str(REPO / "models" / "comparison"))
sys.path.insert(0, str(REPO))
from evaluation import ASREvaluator  # noqa: E402
from sld.sld_pipeline import sld_generate, SLD_CONFIGS  # noqa: E402
from safeclip.safeclip_loader import load_safeclip_text_encoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xeval")

GEN_RES = 512
GEN_STEPS = 50
GEN_GUIDANCE = 7.5
SEED = 42

# (canonical key, prompt file under lsse/data/eval, accepted subfolder names)
ATTACKS = [
    ("I2P",             "i2p_nudity.txt",            ["i2p"]),
    ("Ring-A-Bell",     "ring_a_bell_nudity.txt",    ["ring_a_bell"]),
    ("Ring-A-Bell(Re)", "ring_a_bell_re_nudity.txt", ["ring_a_bell_re", "ring_a_bellre"]),
    ("P4D",             "p4d_nudity.txt",            ["p4d"]),
    ("UnlearnDiffAtk",  "unlearnDiffAtk_nudity.txt", ["unlearndiffatk", "unlearnDiffAtk"]),
]

NEG_NUDITY = ("nudity, nude, naked, bare skin, breasts, nipples, genitals, "
              "explicit, sexual, NSFW, erotic, pornographic")


# label -> spec. kind: sd1 | sd2 | odace. model_id/base may be a str or list of fallbacks.
# neg_prompt: optional fixed negative prompt (inference-time safety baseline).
REGISTRY = {
    "raw_v14": {"kind": "sd1", "model_id": "CompVis/stable-diffusion-v1-4",
                "attack_dir": "models/core/lsse/outputs/eval/xharness_rawsd/images/fcf_nudity"},
    "odace_v3": {"kind": "odace", "base": "CompVis/stable-diffusion-v1-4",
                 "unet_dir": "models/comparison/odace/outputs/odace_v3/final",
                 "attack_dir": "models/comparison/odace/outputs/eval/odace_v3/images/fcf_nudity"},
    "raw_v15": {"kind": "sd1", "model_id": [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "sd-legacy/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"]},
    "safe_neg": {"kind": "sd1", "neg_prompt": NEG_NUDITY, "model_id": [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "sd-legacy/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"]},
    # SD2.1-base. Try the original id first, then public mirrors if it is gated/private.
    "sd21base": {"kind": "sd2", "model_id": [
        "stabilityai/stable-diffusion-2-1-base",
        "Manojb/stable-diffusion-2-1-base",
        "sd2-community/stable-diffusion-2-1-base",
    ]},
    "odace_v15": {"kind": "odace", "unet_dir": "models/comparison/odace/outputs/odace_v15/final", "base": [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "sd-legacy/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"]},
    # --- Reproduced reference baselines (Phase 1-4), all on SD v1.4 to match each paper. ---
    # ESD: trained UNET swapped in like odace. SLD: training-free 3-way safety guidance (config
    # selects the paper preset). Safe-CLIP: training-free CLIP text-encoder swap.
    "esd_u": {"kind": "esd", "base": "CompVis/stable-diffusion-v1-4",
              "unet_dir": "models/comparison/esd/outputs/esd_u/final"},
    "sld_medium": {"kind": "sld", "config": "medium", "model_id": "CompVis/stable-diffusion-v1-4"},
    "sld_strong": {"kind": "sld", "config": "strong", "model_id": "CompVis/stable-diffusion-v1-4"},
    "sld_max":    {"kind": "sld", "config": "max",    "model_id": "CompVis/stable-diffusion-v1-4"},
    "safeclip": {"kind": "safeclip", "model_id": "CompVis/stable-diffusion-v1-4",
                 "safeclip_id": "aimagelab/safeclip_vit-l_14"},
    # --- Text-encoder-family checkpoints (CLIPTextModel swap on SD v1.4) to fill empty COCO
    # cells. te_dir = local saved CLIPTextModel ("final" dir). Weights-only read of fcf/
    # models/comparison/novel/dace/lsse checkpoints (allowed by CLAUDE.md; no parent-code import). ---
    "sph_ot":       {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                     "te_dir": "models/comparison/novel/outputs/fcf_p_v2_nudity_spherical_ot/final"},
    "fcf_p":        {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                     "te_dir": "models/core/fcf/official_fcf_p/final"},
    "fcf_e":        {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                     "te_dir": "models/core/fcf/official_fcf_e/final"},
    "lsse_plu":     {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                     "te_dir": "models/core/lsse/outputs/sweep/plu_seed42/final"},
    "lsse_plu_w2":  {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                     "te_dir": "models/core/lsse/outputs/sweep/stack_plu_w2_seed42/final"},
    "vanilla_lsse": {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                     "te_dir": "models/core/lsse/outputs/sweep/baseline_seed42/final"},
    "dace_v2":      {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                     "te_dir": "models/core/dace/outputs/dace_nudity/final"},
    "dace_plu":     {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                     "te_dir": "models/core/dace/outputs/dace_nudity_plu/final"},
    "odace_v2":     {"kind": "odace", "base": "CompVis/stable-diffusion-v1-4",
                     "unet_dir": "models/comparison/odace/outputs/odace_nudity/final"},
    # --- Official FCF reproduction: trained with the AUTHORS' own code+data
    # (github.com/f-c-forgetting/FCF, data/train/nudity.csv) in our env, loaded via te_swap. ---
    "fcf_p_official": {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                       "te_dir": "models/core/fcf/official_fcf_p/final"},
    "fcf_e_official": {"kind": "te_swap", "base": "CompVis/stable-diffusion-v1-4",
                       "te_dir": "models/core/fcf/official_fcf_e/final"},
}


def _try_from_pretrained(cls, ids, **kw):
    ids = ids if isinstance(ids, list) else [ids]
    last = None
    for mid in ids:
        try:
            return cls.from_pretrained(mid, **kw), mid
        except Exception as exc:  # noqa: BLE001
            last = exc
            logger.warning(f"load failed {mid}: {exc}")
    raise RuntimeError(f"all model ids failed: {ids} ({last})")


def build_pipe(spec, device):
    from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, UNet2DConditionModel
    dtype = torch.float16
    base = spec.get("model_id", spec.get("base"))
    pipe, mid = _try_from_pretrained(StableDiffusionPipeline, base,
                                     torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    if spec["kind"] in ("odace", "esd"):  # both swap in a fine-tuned UNET
        unet_path = REPO / spec["unet_dir"]
        pipe.unet = UNet2DConditionModel.from_pretrained(unet_path).to(device=device, dtype=dtype)
    pipe = pipe.to(device)
    if spec["kind"] == "safeclip":        # swap the CLIP text encoder (after device move)
        load_safeclip_text_encoder(pipe, spec.get("safeclip_id"))
    if spec["kind"] == "te_swap":         # swap in a local saved CLIPTextModel checkpoint
        from transformers import CLIPTextModel
        te_path = REPO / spec["te_dir"]
        pipe.text_encoder = CLIPTextModel.from_pretrained(te_path).to(device=device, dtype=dtype)
        logger.info(f"swapped text encoder <- {spec['te_dir']}")
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    logger.info(f"built pipe kind={spec['kind']} id={mid}")
    return pipe


@torch.no_grad()
def generate(pipe, prompts, out_dir, neg_prompt=None, sld_cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    desc = "/".join(Path(out_dir).parts[-2:])          # e.g. "<label>/i2p" or "<label>/coco"
    it = enumerate(prompts)
    if _HAS_TQDM:
        it = tqdm(it, total=len(prompts), desc=f"gen {desc}", unit="img", leave=False)
    for i, p in it:
        fp = os.path.join(out_dir, f"{i:04d}_00.png")
        if os.path.exists(fp):
            continue
        g = torch.Generator(device=pipe.device).manual_seed(SEED + i)
        if sld_cfg is not None:  # SLD: custom 3-way safety-guided denoise loop
            img = sld_generate(pipe, p, sld_cfg, generator=g, steps=GEN_STEPS,
                               guidance_scale=GEN_GUIDANCE, height=GEN_RES, width=GEN_RES)
        else:
            img = pipe(p, negative_prompt=neg_prompt, num_inference_steps=GEN_STEPS,
                       guidance_scale=GEN_GUIDANCE, height=GEN_RES, width=GEN_RES,
                       generator=g).images[0]
        img.save(fp)
    return out_dir


def read_prompts(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def find_sub(base_abs: Path, variants):
    for v in variants:
        if (base_abs / v).is_dir():
            return base_abs / v
    return None


def run(label, device, asr_ev, n_attack=50, regen=False):
    spec = REGISTRY[label]
    out_base = HERE / "outputs" / label
    out_base.mkdir(parents=True, exist_ok=True)

    need_attacks = "attack_dir" not in spec or regen
    pipe = build_pipe(spec, device)
    neg = spec.get("neg_prompt")
    sld_cfg = SLD_CONFIGS[spec["config"]] if spec["kind"] == "sld" else None

    # --- attack images: reuse existing dir or generate fresh ---
    attack_root = (REPO / spec["attack_dir"]) if ("attack_dir" in spec and not regen) \
        else (out_base / "attacks")
    metrics = {"label": label, "kind": spec["kind"], "asr": {}, "asr_flagged": {}}
    for key, pfile, variants in ATTACKS:
        prompts = read_prompts(LSSE / "data/eval" / pfile)[:n_attack]
        if need_attacks:
            sub = attack_root / variants[0]
            generate(pipe, prompts, str(sub), neg_prompt=neg, sld_cfg=sld_cfg)
        else:
            sub = find_sub(attack_root, variants)
            if sub is None:
                logger.warning(f"{label}: no subfolder for {key} under {attack_root}")
                continue
        res = asr_ev.evaluate_directory(str(sub))
        metrics["asr"][key] = round((res.get("asr") or 0) * 100, 1)
        metrics["asr_flagged"][key] = res.get("flagged")

    asr_vals = [v for v in metrics["asr"].values()]
    metrics["asr_mean"] = round(sum(asr_vals) / len(asr_vals), 1) if asr_vals else None

    (out_base / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"=== {label} === ASR {metrics['asr_mean']}")
    del pipe
    torch.cuda.empty_cache()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of REGISTRY labels")
    ap.add_argument("--n_attack", type=int, default=50)
    ap.add_argument("--regen", action="store_true", help="regenerate even if attack_dir exists")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    asr_ev = ASREvaluator(concept_type="nudity")

    all_metrics = {}
    for label in args.models.split(","):
        label = label.strip()
        if label not in REGISTRY:
            logger.error(f"unknown model '{label}' (known: {list(REGISTRY)})")
            continue
        all_metrics[label] = run(label, device, asr_ev, n_attack=args.n_attack, regen=args.regen)
    print("\n=== SUMMARY (ASR) ===")
    for label, m in all_metrics.items():
        print(f"{label:12s} ASR={m['asr_mean']}")


if __name__ == "__main__":
    main()
