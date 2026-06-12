"""Aggregate per-run train_cost.json files into models/fcf/train_cost.json for the gallery.

Environment-aware: each model's numbers come from whichever machine trained it (written by
cost_utils.CostMeter). A fresh git clone that re-trains regenerates the per-run files; re-running
this aggregator then refreshes the gallery's Train-cost column for that environment. No USD values.

Trained models: located via xeval.REGISTRY (te_dir / unet_dir -> run dir); we read
<run_dir>/train_cost.json or its parent. Models with no train_cost.json yet -> omitted (gallery
shows pending until re-trained on the local machine).

Training-free / external baselines (raw SD, SLD inference guidance, safe negative prompt, Safe-CLIP
external encoder, SD2.1 filtered pretraining) have no local training cost -- recorded statically.

Run (any env, CPU only):
  python eval/aggregate_cost.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # eval/
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import xeval  # noqa: E402  (REGISTRY)

OUT = REPO / "models" / "fcf" / "train_cost.json"

# Baselines with no *local* training step. gallery shows these as "free" (gpu_hours 0).
TRAINING_FREE = {
    "raw_v14": "raw SD v1.4 (no unlearning)",
    "raw_v15": "raw SD v1.5 (no unlearning)",
    "sd21base": "SD2.1-base (NSFW-filtered pretraining, not trained by us)",
    "safe_neg": "inference-time negative prompt (no training)",
    "sld_medium": "SLD inference-time guidance (no training)",
    "sld_strong": "SLD inference-time guidance (no training)",
    "sld_max": "SLD inference-time guidance (no training)",
    "safeclip": "external aimagelab Safe-CLIP encoder (not trained by us)",
}


def _run_dir(spec) -> Path | None:
    rel = spec.get("te_dir") or spec.get("unet_dir")
    return (REPO / rel) if rel else None


def _load_cost(run_dir: Path):
    for cand in (run_dir / "train_cost.json", run_dir.parent / "train_cost.json"):
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except Exception:  # noqa: BLE001
                pass
    return None


def main():
    models = {}
    for label, spec in xeval.REGISTRY.items():
        if label in TRAINING_FREE:
            models[label] = {"training_free": True, "gpu_hours": 0.0,
                             "trainable_params_M": 0.0, "note": TRAINING_FREE[label]}
            continue
        rd = _run_dir(spec)
        if rd is None:
            continue
        rec = _load_cost(rd)
        if rec is not None:
            models[label] = rec
    out = {"_doc": "Environment-aware training cost. gpu_hours = wall*gpu_count measured on the "
                   "training machine (see 'gpu'); re-train on another env -> re-run "
                   "aggregate_cost.py to refresh. training_free baselines have no local training. "
                   "No USD (hardware-neutral).",
           "models": models}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT} ({len(models)} models; "
          f"{sum(1 for m in models.values() if m.get('training_free'))} training-free)")


if __name__ == "__main__":
    main()
