"""train.py — entry point for the fcf-novel-methods sub-project.

This script is INDEPENDENT of the parent thesis package (`fcf/`). All training
code lives in the sibling `core/` (base FCF trainer) and `methods/` (novel
extensions: N5 spherical manifold, N6 OT noise) folders.

Usage:
    python train.py --config configs/nudity_v2.yaml

CLI overrides (take precedence over YAML):
    --manifold {euclidean,spherical}    select N5 manifold
    --ot_noise_file <path>              load N6 OT-learned noise vocabulary
    --eta / --mu_p / --mu_e / --learning_rate / --num_epochs / --seed
    --skip_explicit / --skip_implicit
    --resume_from <path>                continue from Stage-1 checkpoint
"""

from __future__ import annotations

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
from typing import Optional

import numpy as np
import torch
import yaml
from transformers import CLIPTextModel, CLIPTokenizer

# Add this sub-project root to sys.path so `core` / `methods` resolve as
# top-level packages without needing an installed wheel.
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core import FCFDataset                                              # noqa: E402
from methods import NovelFCFTrainer                                      # noqa: E402
from methods.ot_noise import OTNoiseResult                               # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _save_run_metadata(path: str, cfg: dict, args: argparse.Namespace, device: torch.device):
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
        "subproject":     "fcf-novel-methods",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Run metadata saved -> {path}")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, base_dir: str, train_csv: Optional[str] = None) -> FCFDataset:
    def abspath(rel: str) -> str:
        return str(Path(base_dir) / rel) if not Path(rel).is_absolute() else rel

    implicit_groups_cfg = cfg.get("implicit_groups", None)

    explicit_concepts_file = cfg.get("explicit_concepts_file")
    explicit_concepts: list = []
    if explicit_concepts_file:
        p = Path(abspath(explicit_concepts_file))
        if p.exists():
            lines = [l.strip() for l in p.read_text("utf-8").splitlines()]
            explicit_concepts = [l for l in lines if l and not l.startswith("#")]

    csv_path = train_csv or cfg.get("train_csv")
    if csv_path:
        csv_path = abspath(csv_path)
        logger.info(f"Loading dataset from CSV: {csv_path}")
        if not explicit_concepts:
            raise ValueError("explicit_concepts_file required when using CSV format")
        if not implicit_groups_cfg:
            raise ValueError("implicit_groups required when using CSV format")
        ds = FCFDataset.from_csv(
            train_csv=csv_path,
            implicit_groups=implicit_groups_cfg,
            explicit_concepts=explicit_concepts,
            target_concept=cfg["target_concept"],
            seed=cfg.get("seed", 42),
        )
    else:
        logger.info("Loading dataset from text files")
        ds = FCFDataset.from_files(
            explicit_prompts_file=abspath(cfg["explicit_prompts_file"]),
            implicit_concepts_file=abspath(cfg["implicit_concepts_file"]),
            maintain_prompts_file=abspath(cfg["maintain_prompts_file"]),
            explicit_concepts_file=abspath(cfg["explicit_concepts_file"]),
            target_concept=cfg["target_concept"],
            implicit_groups_config=implicit_groups_cfg,
            seed=cfg.get("seed", 42),
        )

    ot_noise_path = cfg.get("ot_noise_file")
    if ot_noise_path:
        ot_path_abs = abspath(ot_noise_path)
        if not Path(ot_path_abs).exists():
            raise FileNotFoundError(f"ot_noise_file not found: {ot_path_abs}")
        ot_result = OTNoiseResult.load(ot_path_abs)
        n_target = len(ds.explicit_prompts)
        pool = ot_result.noise_prompts
        if len(pool) < n_target:
            reps = (n_target + len(pool) - 1) // len(pool)
            extended = (pool * reps)[:n_target]
        else:
            extended = pool[:n_target]
        ds.noise_prompts = extended
        logger.info(
            f"[N6] Overrode noise_prompts with OT-learned vocabulary "
            f"(W-distance={ot_result.wasserstein_distance:.4f}, "
            f"{len(extended)} prompts)"
        )

    return ds


def save_history(history: list, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="fcf-novel-methods training (N5 + N6)")
    parser.add_argument("--config",        type=str, required=True)
    parser.add_argument("--train_csv",     type=str, default=None)
    parser.add_argument("--eta",           type=float, default=None)
    parser.add_argument("--mu_p",          type=float, default=None)
    parser.add_argument("--mu_e",          type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs",    type=int,   default=None)
    parser.add_argument("--seed",          type=int,   default=None)
    parser.add_argument("--manifold",      type=str,   default=None,
                        choices=["euclidean", "spherical"])
    parser.add_argument("--ot_noise_file", type=str,   default=None)
    parser.add_argument("--skip_explicit", action="store_true")
    parser.add_argument("--skip_implicit", action="store_true")
    parser.add_argument("--resume_from",   type=str,   default=None)
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=4)
    args = parser.parse_args()

    base_dir = str(_PROJECT_ROOT)
    cfg = load_config(args.config)

    for key in ["eta", "mu_p", "mu_e", "learning_rate", "num_epochs",
                "seed", "manifold", "ot_noise_file"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
            logger.info(f"  Override: {key} = {val}")

    logger.info("=" * 60)
    logger.info(f"Experiment : {cfg['experiment_name']} (fcf-novel-methods)")
    logger.info(f"Method     : {cfg['method'].upper()}")
    logger.info(f"Manifold   : {cfg.get('manifold', 'euclidean')}")
    logger.info(f"OT noise   : {cfg.get('ot_noise_file') or 'random (paper-spec)'}")
    logger.info(f"Concept    : {cfg['target_concept']}")
    logger.info("=" * 60)

    if args.skip_explicit and not args.resume_from and not args.skip_implicit:
        logger.warning(
            "--skip_explicit set without --resume_from: Stage 2 will run on the "
            "raw pretrained encoder."
        )

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found - training will be slow on CPU.")
    logger.info(f"Device: {device}")
    set_seed(cfg.get("seed", 42))

    clip_id = cfg.get("clip_model_id", "openai/clip-vit-large-patch14")
    logger.info(f"Loading CLIP: {clip_id}")
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModel.from_pretrained(clip_id)

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_from path does not exist: {args.resume_from}")
        logger.info(f"Resuming encoder from: {args.resume_from}")
        if args.resume_from.endswith(".pt"):
            try:
                sd = torch.load(args.resume_from, map_location="cpu", weights_only=True)
            except (TypeError, RuntimeError):
                sd = torch.load(args.resume_from, map_location="cpu")
            text_encoder.load_state_dict(sd)
        else:
            text_encoder = CLIPTextModel.from_pretrained(args.resume_from)

    logger.info("Loading prompt datasets...")
    dataset = build_dataset(cfg, base_dir, train_csv=args.train_csv)
    logger.info(f"  {dataset}")

    trainer = NovelFCFTrainer(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        learning_rate=cfg["learning_rate"],
        eta=cfg["eta"],
        mu_p=cfg["mu_p"],
        mu_e=cfg["mu_e"],
        batch_size=args.batch_size,
        manifold=cfg.get("manifold", "euclidean"),
    )

    output_dir = str(Path(base_dir) / cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    num_epochs = cfg.get("num_epochs", 60)

    log_file = os.path.join(output_dir, "train.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"Logging to file: {log_file}")

    _save_run_metadata(
        path=os.path.join(output_dir, "training_metadata.json"),
        cfg=cfg, args=args, device=device,
    )
    with open(os.path.join(output_dir, "run_config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    if not args.skip_explicit:
        logger.info("\n[Stage 1] Explicit Concept Forgetting (Algorithm 1)")
        history_explicit = trainer.train_explicit(
            dataset=dataset, num_epochs=num_epochs, log_every=cfg.get("log_every", 1),
        )
        stage1_dir = os.path.join(output_dir, "stage1_explicit")
        trainer.save(stage1_dir)
        trainer.save_pt(os.path.join(output_dir, "stage1_explicit.pt"))
        save_history(history_explicit, os.path.join(output_dir, "history_explicit.json"))
        logger.info(f"Stage 1 checkpoint saved -> {stage1_dir}")
    else:
        logger.info("[Stage 1] Skipped (--skip_explicit)")

    if not args.skip_implicit:
        method = cfg.get("method", "fcf_p").lower()
        logger.info(f"\n[Stage 2] Implicit Concept Forgetting ({method.upper()})")
        if method == "fcf_p":
            history_implicit = trainer.train_projection_implicit(
                dataset=dataset, num_epochs=num_epochs, log_every=cfg.get("log_every", 1),
                retain_full=cfg.get("retain_full", False),
            )
        elif method == "fcf_e":
            experience = trainer.compute_experience(dataset)
            torch.save(experience, os.path.join(output_dir, "experience.pt"))
            history_implicit = trainer.train_empirical_implicit(
                dataset=dataset, experience=experience, num_epochs=num_epochs,
                log_every=cfg.get("log_every", 1),
            )
        else:
            raise ValueError(f"Unknown method '{method}'")
        stage2_dir = os.path.join(output_dir, "stage2_implicit")
        trainer.save(stage2_dir)
        trainer.save_pt(os.path.join(output_dir, "stage2_implicit.pt"))
        save_history(history_implicit, os.path.join(output_dir, "history_implicit.json"))
        logger.info(f"Stage 2 checkpoint saved -> {stage2_dir}")
    else:
        logger.info("[Stage 2] Skipped (--skip_implicit)")

    final_dir = os.path.join(output_dir, "final")
    trainer.save(final_dir)
    trainer.save_pt(os.path.join(output_dir, "final.pt"))
    logger.info(f"\nFinal encoder saved -> {final_dir}")
    logger.info(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
