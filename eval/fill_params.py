"""Fill trainable_params_M for the roster models whose trainers never recorded it (None).

Counts requires_grad params exactly like cost_utils.CostMeter (sum(p.numel() for trainable p)),
by loading the same module each method optimizes — NO training, just model load + count:
  - FCF-P / FCF-E  : Adam over the FULL CLIPTextModel (concept_forgetting_train.py:65,
                     model_ori frozen) -> full text-encoder param count.
  - SLERP-OT       : novel/core/trainer.py Adam over self.text_encoder.parameters() -> full TE.
  - ESD-u          : esd_trainer.set_trainable_esd(unet, "noxattn") returns the trainable count
                     (UNet minus cross-attention).
(og_* / DACE are not in the gallery roster, so skipped.)

Then patches each canonical run-dir train_cost.json (trainable_params_M only; everything else kept).
Run (WSL env lsse):  python eval/fill_params.py   [then aggregate_cost + gallery rebuild]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = "CompVis/stable-diffusion-v1-4"


def patch(dst: Path, val_m: float) -> None:
    d = json.loads(dst.read_text()) if dst.exists() else {}
    d["trainable_params_M"] = round(val_m, 2)
    d.setdefault("params_source", "fill_params (requires_grad count)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(d, indent=2))
    print(f"[patch] {dst} -> trainable_params_M={round(val_m, 2)}")


def main() -> None:
    # --- full text encoder (FCF-P/E, SLERP-OT all do Adam over the whole CLIPTextModel) ---
    from transformers import CLIPTextModel
    te = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder")
    full_te_m = sum(p.numel() for p in te.parameters()) / 1e6
    print(f"full CLIPTextModel params = {full_te_m:.2f}M")
    for rel in ("models/fcf/official_fcf_p/train_cost.json",
                "models/fcf/official_fcf_e/train_cost.json",
                "models/novel/outputs/fcf_p_v2_nudity_spherical_ot/train_cost.json"):
        try:
            patch(REPO / rel, full_te_m)
        except Exception as e:
            print(f"[skip] {rel}: {e}")

    # --- ESD-u: UNet minus cross-attn, via the trainer's own selector ---
    try:
        sys.path.insert(0, str(REPO / "models" / "esd"))
        from esd_trainer import set_trainable_esd
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet")
        n_esd = set_trainable_esd(unet, "noxattn")
        esd_m = float(n_esd) / 1e6
        print(f"ESD-u noxattn trainable params = {esd_m:.2f}M")
        patch(REPO / "models/esd/outputs/esd_u/train_cost.json", esd_m)
    except Exception as e:
        print(f"[skip] esd_u: {e}")


if __name__ == "__main__":
    main()
