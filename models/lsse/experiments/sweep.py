"""sweep.py — LSSE 실험 스윕 러너 (Phase 2 재현성 척추).

매트릭스 YAML의 각 cell을 여러 seed로 train→eval 순차 실행. 단일 GPU 가정.
resumable: 이미 eval_results.json 있으면 건너뜀. tmux에서 장시간 무인 실행 대상.

Usage (conda env 안에서):
    python experiments/sweep.py --matrix experiments/configs/improvements.yaml --seeds 42,1,2
    python experiments/sweep.py --matrix experiments/configs/improvements.yaml --seeds 42 --num_images 50 --dry_run

매트릭스 YAML 형식:
    base_config: configs/nudity_lsse.yaml
    cells:
      - name: baseline
        train_flags: []
      - name: w1_tokensel
        train_flags: ["--use_tokensel_dir"]

산출물:
    outputs/sweep/<name>_seed<N>/final  (학습된 인코더)
    outputs/eval/<name>_seed<N>/eval_results.json  (ASR/quality)
    experiments/sweep.log, experiments/sweep_status.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

_LSSE_ROOT = Path(__file__).resolve().parent.parent  # lsse/
_EXP_DIR = Path(__file__).resolve().parent            # lsse/experiments/

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_EXP_DIR / "sweep.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("sweep")


def load_matrix(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def update_status(status_path: Path, key: str, info: dict):
    data = {}
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[key] = info
    status_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_cmd(cmd: List[str], log_prefix: str) -> bool:
    logger.info(f"[{log_prefix}] $ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_LSSE_ROOT))
    dt = time.time() - t0
    ok = proc.returncode == 0
    logger.info(f"[{log_prefix}] {'OK' if ok else 'FAIL rc=' + str(proc.returncode)} ({dt:.0f}s)")
    return ok


def main():
    ap = argparse.ArgumentParser(description="LSSE 스윕 러너")
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--seeds", default="42", help="쉼표 구분 seed 목록")
    ap.add_argument("--python", default=sys.executable, help="학습/평가용 python 경로")
    ap.add_argument("--num_images", type=int, default=50)
    ap.add_argument("--eval_types", default="asr", help="쉼표 구분: asr,quality")
    ap.add_argument("--concept", default="nudity")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    base_config = matrix.get("base_config", "configs/nudity_lsse.yaml")
    cells = matrix["cells"]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    eval_types = [e.strip() for e in args.eval_types.split(",") if e.strip()]
    status_path = _EXP_DIR / "sweep_status.json"

    total = len(cells) * len(seeds)
    logger.info(f"=== SWEEP start: {len(cells)} cells x {len(seeds)} seeds = {total} runs ===")
    logger.info(f"  base_config={base_config}  python={args.python}  eval_types={eval_types}")

    done, skipped, failed = 0, 0, 0
    for cell in cells:
        name = cell["name"]
        train_flags = cell.get("train_flags", [])
        for seed in seeds:
            run_key = f"{name}_seed{seed}"
            out_dir = f"outputs/sweep/{run_key}"
            eval_dir = f"outputs/eval/{run_key}"
            final_dir = f"{out_dir}/final"
            eval_json = _LSSE_ROOT / eval_dir / "eval_results.json"

            if eval_json.exists():
                logger.info(f"[{run_key}] SKIP (eval_results.json 존재)")
                skipped += 1
                continue

            update_status(status_path, run_key,
                          {"status": "running", "started": datetime.now().isoformat(timespec="seconds")})

            # 1. Train
            train_cmd = [args.python, "train_lsse.py", "--config", base_config,
                         "--output_dir", out_dir, "--seed", str(seed),
                         "--no_diagnostics"] + train_flags
            # 진단은 매트릭스 cell에 diagnostics:true 설정 시에만 켬 (비용)
            if cell.get("diagnostics"):
                train_cmd.remove("--no_diagnostics")

            if args.dry_run:
                logger.info(f"[{run_key}] DRY $ {' '.join(train_cmd)}")
                continue

            if not (_LSSE_ROOT / final_dir).exists():
                if not run_cmd(train_cmd, f"{run_key}/train"):
                    update_status(status_path, run_key, {"status": "train_failed"})
                    failed += 1
                    continue

            # 2. Eval
            all_ok = True
            for et in eval_types:
                eval_cmd = [args.python, "evaluate.py", "--encoder_dir", final_dir,
                            "--concept", args.concept, "--eval_type", et,
                            "--output_dir", eval_dir, "--num_images", str(args.num_images)]
                if not run_cmd(eval_cmd, f"{run_key}/eval:{et}"):
                    all_ok = False

            update_status(status_path, run_key,
                          {"status": "done" if all_ok else "eval_failed",
                           "finished": datetime.now().isoformat(timespec="seconds")})
            if all_ok:
                done += 1
            else:
                failed += 1

    logger.info(f"=== SWEEP end: done={done} skipped={skipped} failed={failed} / {total} ===")


if __name__ == "__main__":
    main()
