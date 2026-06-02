"""
Hyperparameter sweep – reproduces the parameter sensitivity analysis from the paper.

The paper tests:
  - η=0.6,  μ_p=0.85  → nudity -1.00%  but FID -4.68%
  - η=0.01, μ_p=0.3   → nudity +35.80% but FID +4.47%

Usage:
  python analysis/hyperparameter_sweep.py \\
      --config configs/nudity_fcf_p.yaml \\
      --output_dir outputs/sweep_nudity
"""

import argparse
import json
import logging
import os
import sys
from itertools import product
from pathlib import Path

import torch
import yaml
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from fcf import FCFTrainer, FCFDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Default sweep grid (customize as needed)
ETA_VALUES   = [0.01, 0.1, 0.25, 0.6]
MU_P_VALUES  = [0.3,  0.5, 0.7,  0.85]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/sweep")
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument(
        "--eta_values",  type=float, nargs="+", default=ETA_VALUES
    )
    parser.add_argument(
        "--mu_p_values", type=float, nargs="+", default=MU_P_VALUES
    )
    args = parser.parse_args()

    base_dir = str(Path(__file__).parent.parent)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.output_dir, exist_ok=True)

    def abspath(rel):
        return str(Path(base_dir) / rel)

    dataset = FCFDataset.from_files(
        explicit_prompts_file  = abspath(cfg["explicit_prompts_file"]),
        implicit_concepts_file = abspath(cfg["implicit_concepts_file"]),
        maintain_prompts_file  = abspath(cfg["maintain_prompts_file"]),
        explicit_concepts_file = abspath(cfg["explicit_concepts_file"]),
        target_concept         = cfg["target_concept"],
    )

    clip_id = cfg.get("clip_model_id", "openai/clip-vit-large-patch14")
    results = []

    for eta, mu_p in product(args.eta_values, args.mu_p_values):
        run_name = f"eta{eta}_mu_p{mu_p}"
        logger.info(f"\n[Sweep] {run_name}")

        tokenizer    = CLIPTokenizer.from_pretrained(clip_id)
        text_encoder = CLIPTextModel.from_pretrained(clip_id)

        trainer = FCFTrainer(
            text_encoder  = text_encoder,
            tokenizer     = tokenizer,
            device        = device,
            learning_rate = cfg["learning_rate"],
            eta           = eta,
            mu_p          = mu_p,
            mu_e          = cfg["mu_e"],
        )

        trainer.train_explicit(dataset=dataset,
                               num_epochs=cfg.get("num_epochs", 60),
                               log_every=10)
        trainer.train_projection_implicit(dataset=dataset,
                                          num_epochs=cfg.get("num_epochs", 60),
                                          log_every=10)

        save_dir = os.path.join(args.output_dir, run_name)
        trainer.save(save_dir)

        results.append({
            "eta": eta, "mu_p": mu_p,
            "encoder_dir": save_dir,
            "run_name": run_name,
        })

    manifest_path = os.path.join(args.output_dir, "sweep_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSweep complete. Manifest → {manifest_path}")
    logger.info(
        "Next: evaluate each encoder_dir with evaluate.py to see "
        "how ASR and FID change across parameter settings."
    )


if __name__ == "__main__":
    main()
