"""scripts/learn_ot_noise.py — CLI entry point for N6 (OT-derived noise vocabulary).

Learns a noise-prompt vocabulary that maximizes Wasserstein distance from the
explicit-prompt embedding distribution. Saves the result as a JSON file that
train.py can load via `--ot_noise_file <path>`.

Example:
    python scripts/learn_ot_noise.py \\
        --explicit_file data/prompts/nudity_explicit.txt \\
        --output outputs/ot_noise/nudity_learned.json \\
        --n_candidates 500 --n_select 100 --metric sliced
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTokenizer

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods.ot_noise import OTNoiseConfig, OTNoiseLearner               # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_lines(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="N6 OT-noise vocabulary learner")
    parser.add_argument("--explicit_file", type=str, required=True)
    parser.add_argument("--clip_model_id", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--n_candidates",  type=int, default=500)
    parser.add_argument("--n_select",      type=int, default=100)
    parser.add_argument("--max_len",       type=int, default=5)
    parser.add_argument("--n_projections", type=int, default=50)
    parser.add_argument("--metric",        type=str, default="sliced",
                        choices=["sliced", "sinkhorn"])
    parser.add_argument("--sinkhorn_eps",  type=float, default=0.1)
    parser.add_argument("--sinkhorn_iter", type=int, default=30)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--device",        type=str, default=None)
    parser.add_argument("--output",        type=str, default="outputs/ot_noise/learned.json")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading CLIP (frozen): {args.clip_model_id}")
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_id)
    encoder = CLIPTextModel.from_pretrained(args.clip_model_id).to(device).eval()
    encoder.requires_grad_(False)

    explicit_prompts = _load_lines(args.explicit_file)
    logger.info(f"Loaded {len(explicit_prompts)} explicit prompts.")

    cfg = OTNoiseConfig(
        n_candidates=args.n_candidates,
        n_select=args.n_select,
        max_len=args.max_len,
        n_projections=args.n_projections,
        seed=args.seed,
        metric=args.metric,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iter=args.sinkhorn_iter,
    )

    learner = OTNoiseLearner(
        frozen_encoder=encoder,
        tokenizer=tokenizer,
        device=device,
        config=cfg,
    )

    logger.info(
        f"Learning OT noise: n_candidates={cfg.n_candidates}, "
        f"n_select={cfg.n_select}, metric={cfg.metric}"
    )
    result = learner.learn(explicit_prompts)

    output_path = str(_PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else args.output
    result.save(output_path)
    logger.info(f"OT-learned noise vocabulary saved -> {output_path}")
    logger.info(f"Final Wasserstein distance: {result.wasserstein_distance:.4f}")
    logger.info(f"First 5 selected prompts: {result.noise_prompts[:5]}")


if __name__ == "__main__":
    main()
