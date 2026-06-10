"""
evaluate.py – Full FCF evaluation pipeline.

Reproduces the main evaluation in the paper:
  - Table 1: ASR for nudity/violence on I2P + red-teaming attacks
  - Table 2: LPIPS for artistic style forgetting
  - Table 3: FID + CLIP score for generative quality

Usage:
  # Full evaluation for nudity FCF-P
  python evaluate.py \\
      --config configs/nudity_fcf_p.yaml \\
      --encoder_dir outputs/fcf_p_nudity/final \\
      --concept nudity

  # Evaluate artistic style (Van Gogh)
  python evaluate.py \\
      --config configs/vangogh_fcf_p.yaml \\
      --encoder_dir outputs/fcf_p_vangogh/final \\
      --concept vangogh \\
      --baseline_image_dir outputs/images/sd_baseline_vangogh

  # Run on existing image directories (skip generation)
  python evaluate.py \\
      --skip_generation \\
      --concept nudity \\
      --generated_dir outputs/images/fcf_p_nudity
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

from evaluation import ASREvaluator, LPIPSEvaluator, FIDCLIPEvaluator
from generate_images import build_pipeline, generate_images, load_prompts  # aliases defined in generate_images.py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_device(device_str: str = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
#  Table 1: ASR evaluation (nudity / violence)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_asr(
    encoder_dir: str,
    sd_model_id: str,
    eval_dirs: dict,        # {"I2P": path, "Ring-A-Bell": path, ...}
    concept_type: str,      # "nudity" or "violence"
    device: torch.device,
    # generation kwargs
    prompts_files: dict = None,   # {"I2P": file, ...}  if generation needed
    num_images: int = 50,
    gen_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    skip_generation: bool = False,
) -> dict:
    """Generate images (if needed) and compute ASR for all attack types."""

    evaluator = ASREvaluator(concept_type=concept_type)
    results = {}

    for label, img_dir in eval_dirs.items():
        # Generate images if directory doesn't exist yet
        if not skip_generation and prompts_files and label in prompts_files:
            prompts_file = prompts_files[label]
            if Path(prompts_file).exists():
                logger.info(f"\n--- Generating images: {label} ---")
                prompts = load_prompts(prompts_file)[:num_images]
                pipe = build_pipeline(
                    sd_model_id=sd_model_id,
                    encoder_dir=encoder_dir,
                    device=device,
                )
                os.makedirs(img_dir, exist_ok=True)
                generate_images(
                    pipe=pipe,
                    prompts=prompts,
                    output_dir=img_dir,
                    num_images_per_prompt=1,
                    num_inference_steps=gen_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
                del pipe
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                logger.warning(f"Prompts file not found for {label}: {prompts_file}")
                continue

        # Evaluate ASR
        if Path(img_dir).exists():
            results[label] = evaluator.evaluate_directory(img_dir)
        else:
            logger.warning(f"Image directory not found: {img_dir}")
            results[label] = {"asr": None}

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Table 2: LPIPS evaluation (artistic style forgetting)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_style(
    encoder_dir: str,
    sd_model_id: str,
    style_prompts_file: str,
    other_prompts_file: str,
    baseline_style_dir: str,
    baseline_other_dir: str,
    output_style_dir: str,
    output_other_dir: str,
    device: torch.device,
    num_images: int = 20,
    gen_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    skip_generation: bool = False,
) -> dict:
    """Generate style images and compute LPIPS_f, LPIPS_m, LPIPS_d."""

    if not skip_generation:
        pipe = build_pipeline(sd_model_id=sd_model_id, encoder_dir=encoder_dir, device=device)

        for prompts_file, out_dir in [
            (style_prompts_file, output_style_dir),
            (other_prompts_file, output_other_dir),
        ]:
            if Path(prompts_file).exists():
                prompts = load_prompts(prompts_file)[:num_images]
                generate_images(pipe=pipe, prompts=prompts, output_dir=out_dir,
                                num_inference_steps=gen_steps,
                                guidance_scale=guidance_scale, seed=seed)
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    evaluator = LPIPSEvaluator(device=device)
    return evaluator.compute_style_forgetting_metrics(
        baseline_dir=baseline_style_dir,
        forgetting_dir=output_style_dir,
        baseline_other=baseline_other_dir,
        forgetting_other=output_other_dir,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Table 3: FID + CLIP score (generative quality)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_generative_quality(
    image_dir: str,
    prompts: list,
    reference_dir: str = None,
    device: torch.device = None,
) -> dict:
    evaluator = FIDCLIPEvaluator(device=device)
    return evaluator.evaluate(image_dir, prompts, reference_dir)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FCF Evaluation")
    parser.add_argument("--config",           type=str, default=None)
    parser.add_argument("--encoder_dir",      type=str, default=None,
                        help="Fine-tuned encoder dir (None = SD baseline)")
    parser.add_argument("--concept",          type=str, default="nudity",
                        choices=["nudity", "violence", "vangogh"],
                        help="Which concept was forgotten")
    parser.add_argument("--sd_model_id",      type=str,
                        default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_dir",       type=str, default="outputs/eval")
    parser.add_argument("--eval_type",        type=str, default="asr",
                        choices=["asr", "style", "quality", "all"],
                        help="Which evaluation to run")
    parser.add_argument("--skip_generation",  action="store_true",
                        help="Skip image generation, use existing images")
    parser.add_argument("--device",           type=str, default=None)
    parser.add_argument("--num_images",       type=int, default=50)
    parser.add_argument("--seed",             type=int, default=42)
    # For style evaluation
    parser.add_argument("--baseline_dir",     type=str, default=None,
                        help="Baseline (SD) style image directory for LPIPS_f")
    parser.add_argument("--baseline_other_dir", type=str, default=None,
                        help="Baseline (SD) other-style image directory for LPIPS_m")
    # Multi-GPU
    parser.add_argument("--rank",             type=int, default=0,
                        help="GPU rank for this process (0-based)")
    parser.add_argument("--world_size",       type=int, default=1,
                        help="Total number of parallel processes")
    args = parser.parse_args()
    if args.encoder_dir in ("None", "none", ""):
        args.encoder_dir = None

    base_dir = str(Path(__file__).parent)

    # rank가 지정되면 해당 GPU 사용 (--device 명시 우선)
    if args.device:
        device = get_device(args.device)
    elif torch.cuda.is_available() and args.world_size > 1:
        device = torch.device(f"cuda:{args.rank}")
    else:
        device = get_device(None)
    logger.info(f"Device: {device}  [rank {args.rank}/{args.world_size}]")

    # Load config if provided
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    sd_model_id = cfg.get("sd_model_id", args.sd_model_id)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # ── ASR Evaluation (Table 1) ──────────────────────────────────────────────
    if args.eval_type in ("asr", "all") and args.concept in ("nudity", "violence"):
        concept_key = args.concept
        label = f"fcf_{args.concept}"
        prefix = os.path.join(args.output_dir, "images", label)

        eval_dirs = {
            "I2P":             os.path.join(prefix, "i2p"),
            "Ring-A-Bell":     os.path.join(prefix, "ring_a_bell"),
            "Ring-A-Bell(Re)": os.path.join(prefix, "ring_a_bell_re"),
            "P4D":             os.path.join(prefix, "p4d"),
            "UnlearnDiffAtk":  os.path.join(prefix, "unlearnDiffAtk"),
        }
        prompts_files = {
            "I2P":             os.path.join(base_dir, f"data/eval/i2p_{concept_key}.txt"),
            "Ring-A-Bell":     os.path.join(base_dir, f"data/eval/ring_a_bell_{concept_key}.txt"),
            "Ring-A-Bell(Re)": os.path.join(base_dir, f"data/eval/ring_a_bell_re_{concept_key}.txt"),
            "P4D":             os.path.join(base_dir, f"data/eval/p4d_{concept_key}.txt"),
            "UnlearnDiffAtk":  os.path.join(base_dir, f"data/eval/unlearnDiffAtk_{concept_key}.txt"),
        }

        # Multi-GPU: 공격 유형 목록을 rank로 분할
        if args.world_size > 1:
            all_labels = list(eval_dirs.keys())
            my_labels  = all_labels[args.rank :: args.world_size]
            eval_dirs     = {k: eval_dirs[k]     for k in my_labels}
            prompts_files = {k: prompts_files[k] for k in my_labels if k in prompts_files}
            logger.info(f"[rank {args.rank}] 담당 공격 유형: {my_labels}")

        logger.info(f"\n{'='*60}")
        logger.info(f"ASR Evaluation — {args.concept.upper()}")
        logger.info(f"{'='*60}")

        asr_results = evaluate_asr(
            encoder_dir=args.encoder_dir,
            sd_model_id=sd_model_id,
            eval_dirs=eval_dirs,
            concept_type=concept_key,
            device=device,
            prompts_files=prompts_files,
            num_images=args.num_images,
            gen_steps=cfg.get("eval_steps", 50),
            guidance_scale=cfg.get("eval_guidance_scale", 7.5),
            seed=args.seed,
            skip_generation=args.skip_generation,
        )
        all_results["asr"] = asr_results

        logger.info("\n── ASR Results ─────────────────────────────────")
        for prompt_set, res in asr_results.items():
            asr_val = res.get("asr")
            if asr_val is not None:
                logger.info(f"  {prompt_set:20s}: ASR = {asr_val:.2%}")

    # ── LPIPS Evaluation (Table 2) ────────────────────────────────────────────
    if args.eval_type in ("style", "all") and args.concept == "vangogh":
        logger.info(f"\n{'='*60}")
        logger.info("LPIPS Evaluation — Van Gogh Style Forgetting")
        logger.info(f"{'='*60}")

        prefix = os.path.join(args.output_dir, "images", "fcf_vangogh")
        style_results = evaluate_style(
            encoder_dir=args.encoder_dir,
            sd_model_id=sd_model_id,
            style_prompts_file=os.path.join(base_dir, "data/prompts/vangogh_explicit.txt"),
            other_prompts_file=os.path.join(base_dir, "data/prompts/vangogh_maintain.txt"),
            baseline_style_dir=args.baseline_dir or os.path.join(
                args.output_dir, "images", "sd_baseline_vangogh_style"),
            baseline_other_dir=args.baseline_other_dir or os.path.join(
                args.output_dir, "images", "sd_baseline_vangogh_other"),
            output_style_dir=os.path.join(prefix, "style"),
            output_other_dir=os.path.join(prefix, "other"),
            device=device,
            num_images=args.num_images,
            gen_steps=cfg.get("eval_steps", 50),
            guidance_scale=cfg.get("eval_guidance_scale", 7.5),
            seed=args.seed,
            skip_generation=args.skip_generation,
        )
        all_results["lpips"] = style_results
        logger.info(f"\n── LPIPS Results ────────────────────────────────")
        for k, v in style_results.items():
            logger.info(f"  {k}: {v:.4f}")

    # ── FID + CLIP Score (Table 3) ────────────────────────────────────────────
    if args.eval_type in ("quality", "all"):
        logger.info(f"\n{'='*60}")
        logger.info("Generative Quality — FID & CLIP Score")
        logger.info(f"{'='*60}")

        quality_img_dir = os.path.join(args.output_dir, "images", "quality")
        maintain_prompts_file = os.path.join(
            base_dir, cfg.get("maintain_prompts_file", "data/prompts/nudity_maintain.txt")
        )
        if not args.skip_generation and Path(maintain_prompts_file).exists():
            prompts = load_prompts(maintain_prompts_file)[:args.num_images]
            pipe = build_pipeline(
                sd_model_id=sd_model_id,
                encoder_dir=args.encoder_dir,
                device=device,
            )
            generate_images(pipe=pipe, prompts=prompts,
                            output_dir=quality_img_dir,
                            num_inference_steps=cfg.get("eval_steps", 50),
                            guidance_scale=cfg.get("eval_guidance_scale", 7.5),
                            seed=args.seed)
            del pipe
        else:
            prompts = []
            if Path(maintain_prompts_file).exists():
                prompts = load_prompts(maintain_prompts_file)[:args.num_images]

        quality_results = evaluate_generative_quality(
            image_dir=quality_img_dir,
            prompts=prompts,
            reference_dir=cfg.get("coco_dir"),   # set in config if available
            device=device,
        )
        all_results["quality"] = quality_results
        logger.info(f"\n── Quality Results ───────────────────────────────")
        for k, v in quality_results.items():
            logger.info(f"  {k}: {v:.4f}")

    # ── Save Results ──────────────────────────────────────────────────────────
    result_path = os.path.join(args.output_dir, "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
