"""
generate_images.py – Generate images with a fine-tuned FCF text encoder.

Aligned with the original FCF repository's fcf-generate.py:
  - Uses LMSDiscreteScheduler (original repo default)
  - Supports CSV input with columns: case_number, prompt, evaluation_seed
  - Supports loading .pt state_dict checkpoints (original repo format)
  - Also supports plain-text prompt files and HuggingFace encoder dirs

Usage:
  # CSV prompts + .pt checkpoint (original repo format)
  python generate_images.py \\
      --model_path outputs/fcf_p_nudity/final.pt \\
      --prompts_path data/eval/i2p_nudity.csv \\
      --save_path outputs/images \\
      --model_name fcf_p_nudity

  # HuggingFace encoder dir + plain text prompts
  python generate_images.py \\
      --encoder_dir outputs/fcf_p_nudity/final \\
      --prompts_file data/eval/i2p_nudity.txt \\
      --output_dir outputs/images/fcf_p_nudity_i2p

  # SD baseline (no forgetting, original encoder)
  python generate_images.py \\
      --prompts_file data/eval/i2p_nudity.txt \\
      --output_dir outputs/images/sd_baseline_i2p
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"


# ─────────────────────────────────────────────────────────────────────────────
#  Input loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts_txt(path: str) -> list:
    """Load prompts from plain-text file (one per line, # = comment)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    lines = p.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def load_prompts_csv(path: str):
    """
    Load prompts from CSV with columns: case_number, prompt, evaluation_seed.
    Returns list of dicts.  Matches original fcf-generate.py input format.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for CSV loading: pip install pandas")
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df.to_dict("records")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline construction
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline_from_components(
    sd_model_id: str,
    model_path: str = None,
    device: torch.device = None,
) -> tuple:
    """
    Build SD pipeline components individually (original repo style).
    If model_path (.pt) is provided, load its state_dict into the text encoder.

    Returns (vae, text_encoder, tokenizer, unet, scheduler).
    """
    logger.info(f"Loading SD components from: {sd_model_id}")
    vae          = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_model_id, subfolder="text_encoder")
    unet         = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet")

    if model_path is not None:
        logger.info(f"Loading fine-tuned encoder weights from: {model_path}")
        text_encoder.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )

    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    return vae, text_encoder, tokenizer, unet, scheduler


def build_pipeline_hf(
    sd_model_id: str,
    encoder_dir: str = None,
    device: torch.device = None,
    use_fp16: bool = True,
) -> StableDiffusionPipeline:
    """
    Build a StableDiffusionPipeline with optional HuggingFace encoder directory.
    Uses LMSDiscreteScheduler to match original repo.
    """
    dtype = torch.float16 if (use_fp16 and str(device) != "cpu") else torch.float32

    logger.info(f"Loading SD pipeline: {sd_model_id}  (dtype={dtype})")
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if encoder_dir is not None:
        logger.info(f"Loading fine-tuned encoder from: {encoder_dir}")
        pipe.text_encoder = CLIPTextModel.from_pretrained(encoder_dir).to(dtype)

    # Use LMSDiscreteScheduler (matching original fcf-generate.py)
    pipe.scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
#  Generation: CSV mode (matches original repo style)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_from_csv(
    vae, text_encoder, tokenizer, unet, scheduler,
    prompts_csv: str,
    save_path: str,
    model_name: str,
    device: torch.device,
    guidance_scale: float = 7.5,
    image_size: int = 512,
    ddim_steps: int = 50,
    num_samples: int = 1,
    from_case: int = 0,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Generate images from a CSV with case_number/prompt/evaluation_seed columns.
    Matches fcf-generate.py :: generate_images() exactly.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(prompts_csv, encoding="ISO-8859-1")
    if world_size > 1:
        df = df.iloc[rank::world_size].reset_index(drop=True)
        logger.info(f"  [rank {rank}] CSV rows: {len(df)}개")
    folder_path = os.path.join(save_path, model_name)
    os.makedirs(folder_path, exist_ok=True)

    scheduler.set_timesteps(ddim_steps)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        case_number = int(row.case_number)
        if case_number < from_case:
            continue

        prompt = [str(row.prompt)] * num_samples
        seed   = int(row.evaluation_seed)

        text_input = tokenizer(
            prompt, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

        max_len = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * num_samples, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (num_samples, unet.config.in_channels, image_size // 8, image_size // 8),
            generator=generator,
        ).to(device)
        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        for num, img_arr in enumerate(images):
            pil_img = Image.fromarray(img_arr)
            pil_img.save(os.path.join(folder_path, f"{case_number}_{num}.png"))

    logger.info(f"Done. Images saved to: {folder_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Generation: plain-text mode (pipeline API)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_images(
    pipe: StableDiffusionPipeline,
    prompts: list,
    output_dir: str,
    num_images_per_prompt: int = 1,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    image_size: int = 512,
    seed: int = 42,
    global_indices: list = None,
):
    """Generate images for each prompt and save to output_dir.

    global_indices: if provided (multi-GPU mode), use these as file name prefixes
                    so ranks writing to the same directory don't overwrite each other.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        f"Generating {len(prompts)} prompts × {num_images_per_prompt} images → {output_dir}"
    )

    for local_idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        file_idx = global_indices[local_idx] if global_indices is not None else local_idx
        generator = torch.Generator(device=pipe.device).manual_seed(seed + file_idx)
        output = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=image_size,
            width=image_size,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
        )
        for img_idx, img in enumerate(output.images):
            img.save(os.path.join(output_dir, f"{file_idx:04d}_{img_idx:02d}.png"))

    logger.info(f"Done. Images saved to: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FCF Image Generation",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Original repo interface ───────────────────────────────────────────────
    parser.add_argument("--model_name",   type=str, default=None,
                        help="Subfolder name for saving (CSV mode)")
    parser.add_argument("--prompts_path", type=str, default=None,
                        help="CSV file: case_number, prompt, evaluation_seed (original repo format)")
    parser.add_argument("--save_path",    type=str, default=None,
                        help="Root save directory (CSV mode)")
    parser.add_argument("--model_path",   type=str, default=None,
                        help="Path to .pt state_dict checkpoint (original repo format)")
    parser.add_argument("--from_case",    type=int, default=0,
                        help="Start from this case_number (CSV mode)")
    parser.add_argument("--num_samples",  type=int, default=1)
    parser.add_argument("--ddim_steps",   type=int, default=50)

    # ── HuggingFace / plain-text interface ────────────────────────────────────
    parser.add_argument("--encoder_dir",           type=str, default=None,
                        help="HuggingFace encoder dir (fine-tuned FCF encoder)")
    parser.add_argument("--prompts_file",          type=str, default=None,
                        help="Plain-text prompts file (one per line)")
    parser.add_argument("--output_dir",            type=str, default=None,
                        help="Output directory (plain-text mode)")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--no_fp16",               action="store_true")

    # ── Shared ────────────────────────────────────────────────────────────────
    parser.add_argument("--sd_model_id",    type=str, default=SD_MODEL_ID)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size",     type=int, default=512)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--device",         type=str, default=None)
    # ── Multi-GPU ─────────────────────────────────────────────────────────────
    parser.add_argument("--rank",       type=int, default=0,
                        help="이 프로세스가 담당할 GPU rank (0-based)")
    parser.add_argument("--world_size", type=int, default=1,
                        help="총 병렬 프로세스 수 (default 1 = single GPU)")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        # rank가 지정된 경우 해당 GPU 사용
        gpu_id = args.rank if args.world_size > 1 else 0
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found – inference will be slow on CPU.")
    logger.info(f"Device: {device}  [rank {args.rank}/{args.world_size}]")

    # ── Route to CSV mode or plain-text mode ──────────────────────────────────
    if args.prompts_path:
        # Original repo style: CSV + component loading
        if not args.save_path or not args.model_name:
            parser.error("--save_path and --model_name are required with --prompts_path")

        vae, text_encoder, tokenizer, unet, scheduler = build_pipeline_from_components(
            sd_model_id = args.sd_model_id,
            model_path  = args.model_path,
            device      = device,
        )
        generate_from_csv(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler,
            prompts_csv    = args.prompts_path,
            save_path      = args.save_path,
            model_name     = args.model_name,
            device         = device,
            guidance_scale = args.guidance_scale,
            image_size     = args.image_size,
            ddim_steps     = args.ddim_steps,
            num_samples    = args.num_samples,
            from_case      = args.from_case,
            rank           = args.rank,
            world_size     = args.world_size,
        )

    elif args.prompts_file:
        # Plain-text + HuggingFace pipeline mode
        if not args.output_dir:
            parser.error("--output_dir is required with --prompts_file")

        all_prompts = load_prompts_txt(args.prompts_file)
        logger.info(f"Loaded {len(all_prompts)} prompts from {args.prompts_file}")

        # Multi-GPU: 인터리브 방식으로 프롬프트 분할 (global index 보존)
        if args.world_size > 1:
            my_global_indices = list(range(args.rank, len(all_prompts), args.world_size))
            prompts = [all_prompts[i] for i in my_global_indices]
            global_indices = my_global_indices
            logger.info(f"  [rank {args.rank}] 담당 프롬프트: {len(prompts)}개 "
                        f"(전체 {len(all_prompts)} 중 rank {args.rank} 인터리브, "
                        f"global idx {my_global_indices[0]}~{my_global_indices[-1]})")
        else:
            prompts = all_prompts
            global_indices = None

        pipe = build_pipeline_hf(
            sd_model_id = args.sd_model_id,
            encoder_dir = args.encoder_dir,
            device      = device,
            use_fp16    = not args.no_fp16,
        )
        generate_images(
            pipe                   = pipe,
            prompts                = prompts,
            output_dir             = args.output_dir,
            num_images_per_prompt  = args.num_images_per_prompt,
            guidance_scale         = args.guidance_scale,
            num_inference_steps    = args.ddim_steps,
            image_size             = args.image_size,
            seed                   = args.seed,
            global_indices         = global_indices,
        )

    else:
        parser.error("Provide either --prompts_path (CSV) or --prompts_file (plain text)")


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience aliases (used by evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

#: evaluate.py calls build_pipeline(sd_model_id, encoder_dir, device) — maps to HF pipeline.
build_pipeline = build_pipeline_hf

#: evaluate.py calls load_prompts(path) with plain-text .txt files.
load_prompts = load_prompts_txt


if __name__ == "__main__":
    main()
