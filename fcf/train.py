"""
train_fcf.py – Main entry point for Fortified Concept Forgetting (FCF) training.

Aligned with the official FCF repository structure:
  Stage 1  ←  concept_forgetting_train.py
  Stage 2  ←  features_forgetting_P.py  /  features_forgetting_E.py

Usage examples:
  # Train FCF-P for nudity forgetting (default config)
  python train_fcf.py --config configs/nudity_fcf_p.yaml

  # Train FCF-E for nudity forgetting
  python train_fcf.py --config configs/nudity_fcf_e.yaml

  # Load CSV training data (original repo format)
  python train_fcf.py --config configs/nudity_fcf_p.yaml --train_csv data/train/nudity.csv

  # Override specific hyperparameters
  python train_fcf.py --config configs/nudity_fcf_p.yaml --eta 0.3 --mu_p 0.8

  # Train only Stage 1
  python train_fcf.py --config configs/nudity_fcf_p.yaml --skip_implicit

Reference:
    Fan et al., "Fortified Concept Forgetting for text-to-image generative models
    by machine unlearning on CLIP", CSI 97 (2026) 104142
"""

import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import CLIPTextModel, CLIPTokenizer

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core import FCFTrainer, FCFDataset

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int, deterministic: bool = True):
    """Seed every randomness source for reproducible training.

    Args:
        seed: Master seed.
        deterministic: If True, force cuDNN into deterministic mode.
            Slightly slower but eliminates run-to-run kernel variability.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_git_sha() -> str | None:
    """Return short git SHA of HEAD, or None if not a git repo / git missing."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _save_run_metadata(
    path: str,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
):
    """Dump environment + resolved config to `path` so the run is reproducible.

    Schema:
        timestamp_iso, git_sha, device, torch_version, cuda_available,
        cuda_version, python_version, platform, config, args
    """
    metadata = {
        "timestamp_iso":  datetime.now().isoformat(timespec="seconds"),
        "git_sha":        _get_git_sha(),
        "device":         str(device),
        "torch_version":  torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version":   torch.version.cuda if torch.cuda.is_available() else None,
        "python_version": sys.version.split()[0],
        "platform":       platform.platform(),
        "config":         cfg,
        "args":           {k: v for k, v in vars(args).items() if v is not None},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Run metadata saved → {path}")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI argument overrides on top of YAML config."""
    for key in ["eta", "mu_p", "mu_e", "learning_rate", "num_epochs", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
            logger.info(f"  Override: {key} = {val}")
    return cfg


def build_dataset(cfg: dict, base_dir: str, train_csv: str = None) -> FCFDataset:
    """Load FCFDataset from CSV (primary) or text files (fallback)."""

    def abspath(rel: str) -> str:
        return str(Path(base_dir) / rel)

    # Implicit groups from config (list of lists), e.g.:
    #   implicit_groups: [["male","boy","man"], ["female","girl","woman"]]
    implicit_groups_cfg = cfg.get("implicit_groups", None)

    # Explicit concept name variants for FCF-P projection direction
    explicit_concepts_file = cfg.get("explicit_concepts_file")
    explicit_concepts = []
    if explicit_concepts_file:
        p = Path(abspath(explicit_concepts_file))
        if p.exists():
            lines = [l.strip() for l in p.read_text("utf-8").splitlines()]
            explicit_concepts = [l for l in lines if l and not l.startswith("#")]

    csv_path = train_csv or cfg.get("train_csv")
    if csv_path:
        csv_path = abspath(csv_path) if not Path(csv_path).is_absolute() else csv_path
        logger.info(f"Loading dataset from CSV: {csv_path}")

        if not explicit_concepts:
            raise ValueError("explicit_concepts_file required when using CSV format")
        if not implicit_groups_cfg:
            raise ValueError("implicit_groups required in config when using CSV format")

        return FCFDataset.from_csv(
            train_csv         = csv_path,
            implicit_groups   = implicit_groups_cfg,
            explicit_concepts = explicit_concepts,
            target_concept    = cfg["target_concept"],
            seed              = cfg.get("seed", 42),
        )

    # Fallback: plain text files
    logger.info("Loading dataset from text files")
    return FCFDataset.from_files(
        explicit_prompts_file  = abspath(cfg["explicit_prompts_file"]),
        implicit_concepts_file = abspath(cfg["implicit_concepts_file"]),
        maintain_prompts_file  = abspath(cfg["maintain_prompts_file"]),
        explicit_concepts_file = abspath(cfg["explicit_concepts_file"]),
        target_concept         = cfg["target_concept"],
        implicit_groups_config = implicit_groups_cfg,
        seed                   = cfg.get("seed", 42),
    )


def save_history(history: list, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="FCF Training Script")
    parser.add_argument("--config",       type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--train_csv",    type=str, default=None,
                        help="CSV with prompt_r/prompt_n/prompt_f columns (original repo format)")
    parser.add_argument("--eta",          type=float, default=None)
    parser.add_argument("--mu_p",         type=float, default=None)
    parser.add_argument("--mu_e",         type=float, default=None)
    parser.add_argument("--learning_rate",type=float, default=None)
    parser.add_argument("--num_epochs",   type=int,   default=None,
                        help="Epochs for both Stage 1 and Stage 2 (paper default: 60)")
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--skip_explicit",action="store_true",
                        help="Skip stage 1 (explicit forgetting)")
    parser.add_argument("--skip_implicit",action="store_true",
                        help="Skip stage 2 (implicit forgetting)")
    parser.add_argument("--resume_from",  type=str,   default=None,
                        help="Path to a saved fine-tuned encoder (HuggingFace dir or .pt file)")
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=4,
                        help="Stage 1 배치 크기 (default 4, GPU 메모리에 따라 조정)")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    base_dir = str(Path(__file__).parent)
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    logger.info("=" * 60)
    logger.info(f"Experiment : {cfg['experiment_name']}")
    logger.info(f"Method     : {cfg['method'].upper()}")
    logger.info(f"Concept    : {cfg['target_concept']}")
    logger.info(f"η={cfg['eta']}  μ_p={cfg['mu_p']}  μ_e={cfg['mu_e']}  lr={cfg['learning_rate']}")
    logger.info("=" * 60)

    # ── Validate flag combinations ────────────────────────────────────────────
    if args.skip_explicit and not args.resume_from and not args.skip_implicit:
        logger.warning(
            "--skip_explicit set without --resume_from: Stage 2 will run on the "
            "raw pretrained encoder (no Stage-1 weights). Pass --resume_from to "
            "continue from a saved Stage-1 checkpoint, or use --skip_implicit "
            "to skip Stage 2 as well."
        )

    # ── Device & seed ─────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found – training will be slow on CPU.")
    logger.info(f"Device: {device}")
    set_seed(cfg.get("seed", 42))

    # ── Load CLIP encoder ─────────────────────────────────────────────────────
    clip_id = cfg.get("clip_model_id", "openai/clip-vit-large-patch14")
    logger.info(f"Loading CLIP: {clip_id}")
    tokenizer    = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModel.from_pretrained(clip_id)

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"--resume_from path does not exist: {args.resume_from}"
            )
        logger.info(f"Resuming encoder from: {args.resume_from}")
        if args.resume_from.endswith(".pt"):
            try:
                sd = torch.load(args.resume_from, map_location="cpu", weights_only=True)
            except (TypeError, RuntimeError) as e:
                logger.warning(
                    f"weights_only=True failed ({e}); falling back to legacy load. "
                    f"Only do this for checkpoints you trust."
                )
                sd = torch.load(args.resume_from, map_location="cpu")
            text_encoder.load_state_dict(sd)
        else:
            text_encoder = CLIPTextModel.from_pretrained(args.resume_from)

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Loading prompt datasets...")
    dataset = build_dataset(cfg, base_dir, train_csv=args.train_csv)
    logger.info(f"  {dataset}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = FCFTrainer(
        text_encoder  = text_encoder,
        tokenizer     = tokenizer,
        device        = device,
        learning_rate = cfg["learning_rate"],
        eta           = cfg["eta"],
        mu_p          = cfg["mu_p"],
        mu_e          = cfg["mu_e"],
        batch_size    = args.batch_size,
    )

    output_dir = str(Path(base_dir) / cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    num_epochs = cfg.get("num_epochs", 60)

    # ── Mirror all log output to a per-run file ──────────────────────────────
    log_file = os.path.join(output_dir, "train.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"Logging to file: {log_file}")

    # ── Persist run metadata + config BEFORE training (survives crashes) ─────
    _save_run_metadata(
        path=os.path.join(output_dir, "training_metadata.json"),
        cfg=cfg,
        args=args,
        device=device,
    )
    with open(os.path.join(output_dir, "run_config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Stage 1: Explicit Concept Forgetting ──────────────────────────────────
    history_explicit = []
    if not args.skip_explicit:
        logger.info("\n[Stage 1] Explicit Concept Forgetting (Algorithm 1)")
        history_explicit = trainer.train_explicit(
            dataset    = dataset,
            num_epochs = num_epochs,
            log_every  = cfg.get("log_every", 1),
        )
        stage1_dir = os.path.join(output_dir, "stage1_explicit")
        trainer.save(stage1_dir)
        trainer.save_pt(os.path.join(output_dir, "stage1_explicit.pt"))
        save_history(history_explicit, os.path.join(output_dir, "history_explicit.json"))
        logger.info(f"Stage 1 checkpoint saved → {stage1_dir}")
    else:
        logger.info("[Stage 1] Skipped (--skip_explicit)")

    # ── Stage 2: Implicit Concept Forgetting ──────────────────────────────────
    history_implicit = []
    if not args.skip_implicit:
        method = cfg.get("method", "fcf_p").lower()
        logger.info(f"\n[Stage 2] Implicit Concept Forgetting ({method.upper()})")

        if method == "fcf_p":
            history_implicit = trainer.train_projection_implicit(
                dataset    = dataset,
                num_epochs = num_epochs,
                log_every  = cfg.get("log_every", 1),
            )
        elif method == "fcf_e":
            experience = trainer.compute_experience(dataset)
            # Optionally save experience tensor for reuse
            exp_path = os.path.join(output_dir, "experience.pt")
            torch.save(experience, exp_path)
            logger.info(f"Experience vector saved → {exp_path}")

            history_implicit = trainer.train_empirical_implicit(
                dataset    = dataset,
                experience = experience,
                num_epochs = num_epochs,
                log_every  = cfg.get("log_every", 1),
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'fcf_p' or 'fcf_e'.")

        stage2_dir = os.path.join(output_dir, "stage2_implicit")
        trainer.save(stage2_dir)
        trainer.save_pt(os.path.join(output_dir, "stage2_implicit.pt"))
        save_history(history_implicit, os.path.join(output_dir, "history_implicit.json"))
        logger.info(f"Stage 2 checkpoint saved → {stage2_dir}")
    else:
        logger.info("[Stage 2] Skipped (--skip_implicit)")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final")
    trainer.save(final_dir)
    trainer.save_pt(os.path.join(output_dir, "final.pt"))
    logger.info(f"\nFinal encoder saved → {final_dir}")

    logger.info("\nTraining complete.")
    logger.info(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
