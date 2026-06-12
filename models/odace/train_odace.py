"""ODACE training entry. Run (WSL conda env lsse): python train_odace.py --config configs/nudity_odace.yaml"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path
import torch, yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))  # repo/eval for env-aware cost
from core.dataset import DACEDataset
from core.trainer import ODACETrainer
from cost_utils import CostMeter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_odace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    base = Path(__file__).parent
    with open(base / args.config) as f:
        cfg = yaml.safe_load(f)
    for k in ["num_steps", "alpha", "beta", "batch_size", "output_dir"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} exp={cfg.get('experiment_name')}")

    dataset = DACEDataset.from_files(
        forget_file=str(base / cfg["forget_prompts_file"]),
        retain_file=str(base / cfg["retain_prompts_file"]),
        target_concept=cfg.get("target_concept", "nudity"),
        seed=int(cfg.get("seed", 42)))
    logger.info(repr(dataset))

    trainer = ODACETrainer(
        sd_model_id=cfg["sd_model_id"], device=device,
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        alpha=float(cfg.get("alpha", 1.0)), beta=float(cfg.get("beta", 1.0)), eta=float(cfg.get("eta", 1.0)),
        ddim_steps=int(cfg.get("ddim_steps", 30)),
        sample_guidance=float(cfg.get("sample_guidance", 3.0)),
        xattn_full=bool(cfg.get("xattn_full", False)),
        batch_size=int(cfg.get("batch_size", 4)), seed=int(cfg.get("seed", 42)))

    out_dir = base / cfg.get("output_dir", "outputs/odace_nudity")
    out_dir.mkdir(parents=True, exist_ok=True)
    with CostMeter(cfg.get("experiment_name", "odace"), str(out_dir),
                   steps=int(cfg.get("num_steps", 400))) as meter:
        history = trainer.train(dataset, num_steps=int(cfg.get("num_steps", 400)),
                                log_every=int(cfg.get("log_every", 25)))
        meter.set_trainable_params(trainer.unet)
    trainer.save(str(out_dir / "final"))
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"DONE -> {out_dir}/final ; history.json ({len(history)} steps)")


if __name__ == "__main__":
    main()
