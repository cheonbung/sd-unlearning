"""train_lsse.py — LSSE 프레임워크 학습 엔트리포인트.

부모 프로젝트 또는 fcf-novel-methods와 완전히 독립.
평가는 부모 evaluate.py 재사용 (encoder_dir 인터페이스 호환).

Usage (PowerShell):
    & "c:\\Users\\vip\\anaconda3\\envs\\fcf\\python.exe" lsse\\train_lsse.py --config lsse\\configs\\nudity_lsse.yaml

CLI overrides (YAML보다 우선):
    --learning_rate, --alpha, --beta, --gamma, --temperature
    --num_epochs, --seed, --clm_top_k, --batch_size
    --cap_file <path>          N9 CLM용 CAP JSON
    --use_extended_csr         N8 확장 모드 (forget을 CSR negative로 추가)
    --device <cuda|cpu>
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

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core import LSSEDataset      # noqa: E402
from methods import LSSETrainer   # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, base_dir: str) -> LSSEDataset:
    def abspath(rel: str) -> str:
        return str(Path(base_dir) / rel) if not Path(rel).is_absolute() else rel

    return LSSEDataset.from_files(
        explicit_file=abspath(cfg["explicit_prompts_file"]),
        retain_file=abspath(cfg["retain_prompts_file"]),
        implicit_file=abspath(cfg["implicit_concepts_file"]),
        target_concept=cfg["target_concept"],
        seed=cfg.get("seed", 42),
    )


def save_history(history: list, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_metadata(path: str, cfg: dict, args: argparse.Namespace, device: torch.device):
    meta = {
        "timestamp_iso":  datetime.now().isoformat(timespec="seconds"),
        "git_sha":        _get_git_sha(),
        "device":         str(device),
        "torch_version":  torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python_version": sys.version.split()[0],
        "platform":       platform.platform(),
        "config":         cfg,
        "args":           {k: v for k, v in vars(args).items() if v is not None},
        "subproject":     "lsse",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"메타데이터 저장 -> {path}")


def main():
    parser = argparse.ArgumentParser(description="LSSE 학습 (N7+N8+N9)")
    parser.add_argument("--config",           type=str, required=True)
    parser.add_argument("--learning_rate",    type=float, default=None)
    parser.add_argument("--alpha",            type=float, default=None)
    parser.add_argument("--beta",             type=float, default=None)
    parser.add_argument("--gamma",            type=float, default=None)
    parser.add_argument("--temperature",      type=float, default=None)
    parser.add_argument("--num_epochs",       type=int,   default=None)
    parser.add_argument("--seed",             type=int,   default=None)
    parser.add_argument("--clm_top_k",        type=int,   default=None)
    parser.add_argument("--batch_size",       type=int,   default=None)
    parser.add_argument("--cap_file",         type=str,   default=None)
    parser.add_argument("--use_extended_csr", action="store_true")
    parser.add_argument("--use_ddf",          action="store_true")
    parser.add_argument("--use_macd",         action="store_true")
    parser.add_argument("--use_plu",          action="store_true")
    parser.add_argument("--plu_k1_frac",      type=float, default=None)
    parser.add_argument("--plu_k2_frac",      type=float, default=None)
    parser.add_argument("--use_multi_cnp",    action="store_true")
    parser.add_argument("--num_concept_dirs", type=int,   default=None)
    parser.add_argument("--use_ldlr",         action="store_true")
    parser.add_argument("--output_dir",       type=str,   default=None)
    parser.add_argument("--device",           type=str,   default=None)
    args = parser.parse_args()

    base_dir = str(_PROJECT_ROOT)
    cfg = load_config(args.config)

    for key in ["learning_rate", "alpha", "beta", "gamma", "temperature",
                "num_epochs", "seed", "clm_top_k", "batch_size", "cap_file"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
            logger.info(f"  Override: {key} = {val}")
    if args.use_extended_csr:
        cfg["use_extended_csr"] = True
    if args.use_ddf:
        cfg["use_ddf"] = True
    if args.use_macd:
        cfg["use_macd"] = True
    if args.use_plu:
        cfg["use_plu"] = True
    if args.plu_k1_frac is not None:
        cfg["plu_k1_frac"] = args.plu_k1_frac
    if args.plu_k2_frac is not None:
        cfg["plu_k2_frac"] = args.plu_k2_frac
    if args.use_ldlr:
        cfg["use_ldlr"] = True
    if args.use_multi_cnp:
        cfg["use_multi_cnp"] = True
    if args.num_concept_dirs is not None:
        cfg["num_concept_dirs"] = args.num_concept_dirs
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    logger.info("=" * 60)
    logger.info(f"실험명   : {cfg['experiment_name']} (lsse)")
    logger.info(f"개념     : {cfg['target_concept']}")
    logger.info(f"CLM top-K: {cfg.get('clm_top_k', 3)}")
    logger.info(f"CAP 파일 : {cfg.get('cap_file') or '없음 (uniform fallback)'}")
    logger.info("=" * 60)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        logger.warning("GPU 없음 — CPU 학습은 느립니다.")
    logger.info(f"Device: {device}")
    set_seed(cfg.get("seed", 42))

    clip_id = cfg.get("clip_model_id", "openai/clip-vit-large-patch14")
    logger.info(f"CLIP 로드 중: {clip_id}")
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModel.from_pretrained(clip_id)

    logger.info("데이터셋 로드 중...")
    dataset = build_dataset(cfg, base_dir)
    logger.info(f"  {dataset}")

    cap_file = cfg.get("cap_file")
    if cap_file and not Path(cap_file).is_absolute():
        cap_file = str(Path(base_dir) / cap_file)

    trainer = LSSETrainer(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        learning_rate=cfg.get("learning_rate", 2.5e-5),
        alpha=cfg.get("alpha", 1.0),
        beta=cfg.get("beta", 1.0),
        gamma=cfg.get("gamma", 0.5),
        temperature=cfg.get("temperature", 0.07),
        max_length=cfg.get("max_length", 77),
        batch_size=cfg.get("batch_size", 8),
        cap_json_path=cap_file,
        clm_top_k=cfg.get("clm_top_k", 3),
        use_extended_csr=cfg.get("use_extended_csr", False),
        use_ddf=cfg.get("use_ddf", False),
        use_macd=cfg.get("use_macd", False),
        use_plu=cfg.get("use_plu", False),
        plu_k1_frac=cfg.get("plu_k1_frac", 1/3),
        plu_k2_frac=cfg.get("plu_k2_frac", 2/3),
        use_multi_cnp=cfg.get("use_multi_cnp", False),
        num_concept_dirs=cfg.get("num_concept_dirs", 3),
        use_ldlr=cfg.get("use_ldlr", False),
    )

    output_dir = str(Path(base_dir) / cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    num_epochs = cfg.get("num_epochs", 60)

    log_file = os.path.join(output_dir, "train.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"로그 파일: {log_file}")

    save_metadata(
        path=os.path.join(output_dir, "training_metadata.json"),
        cfg=cfg, args=args, device=device,
    )
    with open(os.path.join(output_dir, "run_config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"\n[LSSE] 학습 시작 — {num_epochs} epochs")
    history = trainer.train(
        dataset=dataset,
        num_epochs=num_epochs,
        log_every=cfg.get("log_every", 10),
    )

    final_dir = os.path.join(output_dir, "final")
    trainer.save(final_dir)
    trainer.save_pt(os.path.join(output_dir, "final.pt"))
    save_history(history, os.path.join(output_dir, "history.json"))
    logger.info(f"\n최종 인코더 -> {final_dir}")
    logger.info(f"결과 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()
