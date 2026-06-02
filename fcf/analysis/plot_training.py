"""
Plot training loss curves from FCF training history JSON files.

Usage:
  python analysis/plot_training.py \\
      --explicit_log outputs/fcf_p_nudity/history_explicit.json \\
      --implicit_log outputs/fcf_p_nudity/history_implicit.json \\
      --output outputs/fcf_p_nudity/training_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_history(path: str) -> dict:
    """Load a list of per-step loss dicts from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    # Convert list of dicts -> dict of lists
    keys = data[0].keys()
    return {k: [d[k] for d in data] for k in keys}


def smooth(values: list, window: int = 20) -> list:
    """Simple moving average smoothing."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr.tolist()
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    # Pad front
    pad = arr[:window - 1].tolist()
    return pad + smoothed.tolist()


def plot_losses(
    explicit_history: dict = None,
    implicit_history: dict = None,
    output_path: str = "training_curves.png",
    title: str = "FCF Training Loss",
):
    """Plot L_maintain, L_forget, L_total for both stages."""

    n_stages = (explicit_history is not None) + (implicit_history is not None)
    fig, axes = plt.subplots(1, n_stages, figsize=(7 * n_stages, 4))
    if n_stages == 1:
        axes = [axes]

    ax_idx = 0
    for stage_name, history in [
        ("Stage 1: Explicit Forgetting", explicit_history),
        ("Stage 2: Implicit Forgetting", implicit_history),
    ]:
        if history is None:
            continue
        ax = axes[ax_idx]
        steps = list(range(1, len(history["L_total"]) + 1))

        for key, color, label in [
            ("L_total",    "tab:blue",   "L_total"),
            ("L_maintain", "tab:green",  "L_maintain"),
            ("L_forget",   "tab:orange", "L_forget"),
        ]:
            if key in history:
                raw = history[key]
                smo = smooth(raw, window=min(20, len(raw) // 5 + 1))
                ax.plot(steps, raw,  alpha=0.2, color=color)
                ax.plot(steps, smo,  alpha=1.0, color=color, label=label, linewidth=1.5)

        ax.set_title(stage_name)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit_log", type=str, default=None)
    parser.add_argument("--implicit_log", type=str, default=None)
    parser.add_argument("--output",       type=str, default="training_curves.png")
    parser.add_argument("--title",        type=str, default="FCF Training Loss")
    args = parser.parse_args()

    explicit_hist = load_history(args.explicit_log) if args.explicit_log else None
    implicit_hist = load_history(args.implicit_log) if args.implicit_log else None

    plot_losses(
        explicit_history=explicit_hist,
        implicit_history=implicit_hist,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
