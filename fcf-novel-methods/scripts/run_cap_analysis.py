"""scripts/run_cap_analysis.py — CLI entry point for N1 (Causal Activation Patching).

Runs layer-level causal mediation analysis on a frozen CLIP text encoder to
discover which transformer layers are most responsible for encoding a target
concept (e.g., "nudity"). Writes the per-layer heatmap to a JSON file.

Example:
    python scripts/run_cap_analysis.py \\
        --concept nudity \\
        --explicit_file data/prompts/nudity_explicit.txt \\
        --output outputs/cap/nudity_heatmap.json \\
        --n_samples 8

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

from methods.cap_analyzer import (                                       # noqa: E402
    CAPConfig,
    CausalActivationPatcher,
)
from core.noise_utils import generate_noise_prompts                      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_lines(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="N1 Causal Activation Patching analysis")
    parser.add_argument("--concept",       type=str, required=True,
                        help="Target concept (e.g. 'nudity'). Used to build noise prompts.")
    parser.add_argument("--explicit_file", type=str, required=True,
                        help="File with explicit prompts (one per line).")
    parser.add_argument("--noise_file",    type=str, default=None,
                        help="Optional file with noise prompts. If omitted, paper-spec random "
                             "noise is generated from explicit_file.")
    parser.add_argument("--clip_model_id", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--n_samples",     type=int, default=32,
                        help="Number of prompts to use from each file.")
    parser.add_argument("--pool",          type=str, default="mean",
                        choices=["mean", "cls", "eos"])
    parser.add_argument("--score_norm",    type=str, default="l2",
                        choices=["l2", "l1", "cos_dist"])
    parser.add_argument("--device",        type=str, default=None)
    parser.add_argument("--output",        type=str, default="outputs/cap/heatmap.json")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading CLIP: {args.clip_model_id}")
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_id)
    text_encoder = CLIPTextModel.from_pretrained(args.clip_model_id)

    explicit_all = _load_lines(args.explicit_file)
    explicit = explicit_all[: args.n_samples]
    logger.info(f"Loaded {len(explicit_all)} explicit prompts; using first {len(explicit)}.")

    if args.noise_file:
        noise_all = _load_lines(args.noise_file)
        noise = noise_all[: args.n_samples]
        if len(noise) < len(explicit):
            raise ValueError(
                f"noise_file has fewer prompts ({len(noise)}) than n_samples ({len(explicit)})"
            )
    else:
        logger.info(f"Generating noise prompts (paper-spec 5-char) for concept='{args.concept}'")
        noise = generate_noise_prompts(explicit, args.concept)

    patcher = CausalActivationPatcher(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        config=CAPConfig(pool=args.pool, score_norm=args.score_norm),
    )

    logger.info(f"Running CAP analysis across {patcher.n_layers} layers...")
    result = patcher.analyze(explicit_prompts=explicit, noise_prompts=noise)

    output_path = str(_PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else args.output
    result.save(output_path)
    logger.info(f"CAP result saved -> {output_path}")
    logger.info(f"Top-3 causally implicated layers: {result.top_k_layers(3)}")
    logger.info(f"Per-layer scores: {[round(s, 4) for s in result.layer_scores]}")


if __name__ == "__main__":
    main()
