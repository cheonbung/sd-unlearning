"""
Ablation study runner – reproduces Table 6 from the paper.

Tests four configurations for each method variant:
  (1) SD baseline:     ECFP=False, ICFP=False
  (2) Explicit only:   ECFP=True,  ICFP=False
  (3) Implicit only:   ECFP=False, ICFP=True
  (4) Full FCF:        ECFP=True,  ICFP=True

Usage:
  python analysis/ablation_study.py \\
      --config configs/nudity_fcf_p.yaml \\
      --output_dir outputs/ablation_nudity_fcf_p
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from fcf import FCFTrainer, FCFDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


ABLATION_CONFIGS = [
    {"name": "ECFP=False_ICFP=False", "explicit": False, "implicit": False},
    {"name": "ECFP=True_ICFP=False",  "explicit": True,  "implicit": False},
    {"name": "ECFP=False_ICFP=True",  "explicit": False, "implicit": True},
    {"name": "ECFP=True_ICFP=True",   "explicit": True,  "implicit": True},
]


def train_ablation_config(
    ablation_cfg: dict,
    cfg: dict,
    dataset: FCFDataset,
    base_dir: str,
    output_dir: str,
    device: torch.device,
) -> str:
    """Train one ablation variant and return the saved encoder directory."""

    clip_id = cfg.get("clip_model_id", "openai/clip-vit-large-patch14")
    tokenizer    = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModel.from_pretrained(clip_id)

    trainer = FCFTrainer(
        text_encoder  = text_encoder,
        tokenizer     = tokenizer,
        device        = device,
        learning_rate = cfg["learning_rate"],
        eta           = cfg["eta"],
        mu_p          = cfg["mu_p"],
        mu_e          = cfg["mu_e"],
    )

    name = ablation_cfg["name"]
    save_dir = os.path.join(output_dir, name)
    logger.info(f"\n{'='*50}")
    logger.info(f"Ablation: {name}")
    logger.info(f"{'='*50}")

    num_epochs = cfg.get("num_epochs", 60)

    if ablation_cfg["explicit"]:
        logger.info("[Stage 1] Explicit forgetting...")
        trainer.train_explicit(
            dataset=dataset,
            num_epochs=num_epochs,
            log_every=cfg.get("log_every", 10),
        )

    if ablation_cfg["implicit"]:
        method = cfg.get("method", "fcf_p").lower()
        logger.info(f"[Stage 2] Implicit forgetting ({method.upper()})...")
        if method == "fcf_p":
            trainer.train_projection_implicit(
                dataset=dataset,
                num_epochs=num_epochs,
                log_every=cfg.get("log_every", 10),
            )
        elif method == "fcf_e":
            experience = trainer.compute_experience(dataset)
            trainer.train_empirical_implicit(
                dataset=dataset,
                experience=experience,
                num_epochs=num_epochs,
                log_every=cfg.get("log_every", 10),
            )

    trainer.save(save_dir)
    logger.info(f"Saved ablation encoder → {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/ablation")
    parser.add_argument("--device",     type=str, default=None)
    args = parser.parse_args()

    base_dir = str(Path(__file__).parent.parent)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    def abspath(rel):
        return str(Path(base_dir) / rel)

    dataset = FCFDataset.from_files(
        explicit_prompts_file  = abspath(cfg["explicit_prompts_file"]),
        implicit_concepts_file = abspath(cfg["implicit_concepts_file"]),
        maintain_prompts_file  = abspath(cfg["maintain_prompts_file"]),
        explicit_concepts_file = abspath(cfg["explicit_concepts_file"]),
        target_concept         = cfg["target_concept"],
    )

    # Train all ablation variants
    encoder_dirs = {}
    for ablation_cfg in ABLATION_CONFIGS:
        save_dir = train_ablation_config(
            ablation_cfg=ablation_cfg,
            cfg=cfg,
            dataset=dataset,
            base_dir=base_dir,
            output_dir=args.output_dir,
            device=device,
        )
        encoder_dirs[ablation_cfg["name"]] = save_dir

    # Save manifest
    manifest = {
        "config": args.config,
        "ablation_dirs": encoder_dirs,
        "next_step": (
            "Run evaluate.py with each encoder_dir to compute ASR for each variant. "
            "See Table 6 in the paper."
        ),
    }
    manifest_path = os.path.join(args.output_dir, "ablation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nAblation manifest saved → {manifest_path}")
    logger.info(
        f"\nNext: run evaluate.py for each variant to compute ASR.\n"
        f"Example:\n"
        f"  python evaluate.py --encoder_dir {list(encoder_dirs.values())[0]} "
        f"--concept {cfg['target_concept']}"
    )


if __name__ == "__main__":
    main()
