"""DACE training entry point. Independent of fcf/ and lsse/.

Usage (WSL conda env lsse):
  python train_dace.py --config configs/nudity_dace.yaml
Optional overrides: --num_epochs --subspace_k --adv_every --alpha --gamma --beta --seed
Produces: <output_dir>/final/ (HF text encoder, eval via lsse/evaluate.py) + history.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))  # make methods/ and core/ importable

from core.dataset import DACEDataset
from core.trainer import DACETrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_dace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_epochs", type=int, default=None)
    ap.add_argument("--subspace_k", type=int, default=None)
    ap.add_argument("--adv_every", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    base = Path(__file__).parent
    with open(base / args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ["num_epochs", "subspace_k", "adv_every", "alpha", "gamma", "beta", "seed", "output_dir"]:
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} seed={seed} cfg={cfg.get('experiment_name')}")

    from transformers import CLIPTextModel, CLIPTokenizer
    model_id = cfg["clip_model_id"]
    text_encoder = CLIPTextModel.from_pretrained(model_id)
    tokenizer = CLIPTokenizer.from_pretrained(model_id)

    dataset = DACEDataset.from_files(
        forget_file=str(base / cfg["forget_prompts_file"]),
        retain_file=str(base / cfg["retain_prompts_file"]),
        target_concept=cfg.get("target_concept", "nudity"),
        seed=seed,
    )
    logger.info(repr(dataset))

    trainer = DACETrainer(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        learning_rate=float(cfg.get("learning_rate", 2.5e-5)),
        alpha=float(cfg.get("alpha", 1.0)),
        gamma=float(cfg.get("gamma", 0.5)),
        beta=float(cfg.get("beta", 1.0)),
        subspace_k=int(cfg.get("subspace_k", 4)),
        adv_ridge=float(cfg.get("adv_ridge", 1e-3)),
        adv_every=int(cfg.get("adv_every", 10)),
        adv_sample=int(cfg.get("adv_sample", 64)),
        ema_decay=float(cfg.get("ema_decay", 0.0)),
        pool_mode=cfg.get("pool", "mean"),
        use_plu=bool(cfg.get("use_plu", False)),
        batch_size=int(cfg.get("batch_size", 8)),
    )

    out_dir = base / cfg.get("output_dir", "outputs/dace_nudity")
    out_dir.mkdir(parents=True, exist_ok=True)
    history = trainer.train(
        dataset,
        num_epochs=int(cfg.get("num_epochs", 30)),
        log_every=int(cfg.get("log_every", 5)),
        ckpt_dir=str(out_dir),
        save_every=int(cfg.get("save_every", 0)),
    )
    trainer.save(str(out_dir / "final"))
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"DONE -> {out_dir}/final ; history.json ({len(history)} epochs)")


if __name__ == "__main__":
    main()
