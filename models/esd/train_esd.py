"""Canonical ESD training entry.

Run (WSL conda env lsse):
  python models/esd/train_esd.py --config models/esd/configs/nudity_esd_u.yaml

Erases the "nudity" concept from SD v1.4 with the canonical ESD-u (noxattn) recipe and saves a
standalone U-Net to <output_dir>/final, which the xeval harness loads via kind="esd".
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from esd_trainer import ESDTrainer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from cost_utils import CostMeter  # noqa: E402  (env-aware training-cost logging)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_esd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--train_method", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    base = Path(__file__).parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path) if (Path.cwd() / cfg_path).exists() else (base / cfg_path.name)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for k in ["num_steps", "train_method", "output_dir"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    concepts = cfg.get("concepts") or [cfg.get("target_concept", "nudity")]
    logger.info(f"device={device} exp={cfg.get('experiment_name')} concepts={concepts}")

    trainer = ESDTrainer(
        sd_model_id=cfg["sd_model_id"], device=device,
        train_method=str(cfg.get("train_method", "noxattn")),
        eta=float(cfg.get("eta", 1.0)),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        ddim_steps=int(cfg.get("ddim_steps", 50)),
        start_guidance=float(cfg.get("start_guidance", 3.0)),
        negative_guidance=float(cfg.get("negative_guidance", 1.0)),
        seed=int(cfg.get("seed", 42)))

    out_dir = Path(cfg.get("output_dir", "models/esd/outputs/esd_u"))
    if not out_dir.is_absolute():
        out_dir = base.parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with CostMeter(str(cfg.get("experiment_name", "esd")), str(out_dir),
                   steps=int(cfg.get("num_steps", 1000))):
        history = trainer.train(concepts, num_steps=int(cfg.get("num_steps", 1000)),
                                log_every=int(cfg.get("log_every", 25)))
    trainer.save(str(out_dir / "final"))
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"DONE -> {out_dir}/final ; history.json ({len(history)} steps)")


if __name__ == "__main__":
    main()
