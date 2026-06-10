"""
prepare_eval_data.py – Download and prepare evaluation prompt files for FCF.

Required files (data/eval/):
  Nudity:
    i2p_nudity.txt               – I2P dataset (Schramowski et al., 2023)
    ring_a_bell_nudity.txt       – Ring-A-Bell adversarial prompts
    ring_a_bell_re_nudity.txt    – Ring-A-Bell re-generated with FCF model
    p4d_nudity.txt               – P4D (Chin et al., 2023)
    unlearnDiffAtk_nudity.txt    – UnlearnDiffAtk (Zhang et al., 2023)

  Violence:
    i2p_violence.txt
    ring_a_bell_violence.txt
    p4d_violence.txt
    unlearnDiffAtk_violence.txt

  Van Gogh style:
    vangogh_style.txt
    other_styles.txt

Usage:
  python scripts/prepare_eval_data.py --check        # check which files are missing
  python scripts/prepare_eval_data.py --download_i2p # auto-download I2P from HuggingFace
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
EVAL_DIR = BASE_DIR / "data" / "eval"

REQUIRED_FILES = {
    "nudity": [
        "i2p_nudity.txt",
        "ring_a_bell_nudity.txt",
        "ring_a_bell_re_nudity.txt",
        "p4d_nudity.txt",
        "unlearnDiffAtk_nudity.txt",
    ],
    "violence": [
        "i2p_violence.txt",
        "ring_a_bell_violence.txt",
        "p4d_violence.txt",
        "unlearnDiffAtk_violence.txt",
    ],
    "style": [
        "vangogh_style.txt",
        "other_styles.txt",
    ],
}

DOWNLOAD_INSTRUCTIONS = {
    "i2p_nudity.txt": (
        "I2P Dataset (Schramowski et al., CVPR 2023)\n"
        "  1. Download CSV: https://github.com/ml-research/i2p\n"
        "     or via HuggingFace: hf_hub_download('AIML-TUDA/i2p', 'unsafe.csv')\n"
        "  2. Filter rows where 'categories' contains 'nudity'\n"
        "  3. Write one prompt per line to data/eval/i2p_nudity.txt"
    ),
    "i2p_violence.txt": (
        "I2P Dataset – violence subset\n"
        "  Filter rows where 'categories' contains 'violence' from unsafe.csv"
    ),
    "ring_a_bell_nudity.txt": (
        "Ring-A-Bell (Tsai et al., ICLR 2024)\n"
        "  https://github.com/chiayi-hsu/Ring-A-Bell\n"
        "  Run with --concept nudity against your target model."
    ),
    "ring_a_bell_re_nudity.txt": (
        "Ring-A-Bell re-generation:\n"
        "  Run Ring-A-Bell against the FCF-fine-tuned model (not SD baseline)."
    ),
    "p4d_nudity.txt": (
        "P4D (Chin et al., 2023)\n"
        "  https://github.com/joycenerd/P4D\n"
        "  Generate adversarial prompts for nudity concept."
    ),
    "unlearnDiffAtk_nudity.txt": (
        "UnlearnDiffAtk (Zhang et al., 2023)\n"
        "  https://github.com/OPTML-Group/Unlearn-Diff\n"
        "  Generate adversarial prompts using UnlearnDiffAtk method."
    ),
    "vangogh_style.txt": (
        "Van Gogh style eval prompts – create manually:\n"
        "  Prompts like 'a painting in the style of Van Gogh', etc."
    ),
    "other_styles.txt": (
        "Other artist style prompts – create manually:\n"
        "  Prompts for Monet, Picasso, etc. (non-target styles)."
    ),
}


def check_files() -> dict:
    """Check which required files are present/missing."""
    status = {}
    for category, files in REQUIRED_FILES.items():
        for fname in files:
            path = EVAL_DIR / fname
            exists = path.exists() and path.stat().st_size > 0
            status[fname] = {"exists": exists, "category": category, "path": str(path)}
    return status


def print_status(status: dict):
    present = [f for f, s in status.items() if s["exists"]]
    missing = [f for f, s in status.items() if not s["exists"]]

    logger.info(f"\n{'='*60}")
    logger.info("Evaluation data file status")
    logger.info(f"{'='*60}")
    logger.info(f"  Present : {len(present)}/{len(status)}")
    logger.info(f"  Missing : {len(missing)}/{len(status)}")

    if present:
        logger.info("\n[PRESENT]")
        for f in present:
            logger.info(f"  ✓  {f}")

    if missing:
        logger.info("\n[MISSING]")
        for f in missing:
            logger.info(f"  ✗  {f}")
            if f in DOWNLOAD_INSTRUCTIONS:
                for line in DOWNLOAD_INSTRUCTIONS[f].splitlines():
                    logger.info(f"       {line}")
            logger.info("")


def download_i2p():
    """Auto-download I2P prompts from HuggingFace and extract nudity/violence subsets.

    The I2P benchmark CSV (i2p_benchmark.csv) uses the following category labels:
        'sexual'   → maps to nudity eval (i2p_nudity.txt)
        'violence' → maps to violence eval (i2p_violence.txt)
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas required: pip install pandas")
        sys.exit(1)

    logger.info("Downloading I2P dataset from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        csv_path = hf_hub_download(
            repo_id="AIML-TUDA/i2p",
            filename="i2p_benchmark.csv",
            repo_type="dataset",
        )
        logger.info(f"Downloaded to: {csv_path}")
    except Exception as e:
        logger.error(f"HuggingFace download failed: {e}")
        logger.error(
            "Manual download: https://huggingface.co/datasets/AIML-TUDA/i2p\n"
            "  or: https://github.com/ml-research/i2p"
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from I2P dataset")
    logger.info(f"Available categories: {sorted(df['categories'].dropna().str.split(',').explode().str.strip().unique().tolist())}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Nudity subset — I2P uses 'sexual' category for nudity content
    nudity_df = df[df["categories"].str.contains("sexual", case=False, na=False)]
    nudity_prompts = nudity_df["prompt"].dropna().tolist()
    out_nudity = EVAL_DIR / "i2p_nudity.txt"
    out_nudity.write_text("\n".join(nudity_prompts) + "\n", encoding="utf-8")
    logger.info(f"Saved {len(nudity_prompts)} nudity prompts (sexual category) → {out_nudity}")

    # Violence subset
    violence_df = df[df["categories"].str.contains("violence", case=False, na=False)]
    violence_prompts = violence_df["prompt"].dropna().tolist()
    out_violence = EVAL_DIR / "i2p_violence.txt"
    out_violence.write_text("\n".join(violence_prompts) + "\n", encoding="utf-8")
    logger.info(f"Saved {len(violence_prompts)} violence prompts → {out_violence}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FCF evaluation data")
    parser.add_argument("--check",        action="store_true",
                        help="Check which eval files are present/missing")
    parser.add_argument("--download_i2p", action="store_true",
                        help="Auto-download I2P dataset from HuggingFace")
    args = parser.parse_args()

    if not (args.check or args.download_i2p):
        # Default: just print status
        args.check = True

    if args.check:
        status = check_files()
        print_status(status)

    if args.download_i2p:
        download_i2p()


if __name__ == "__main__":
    main()
