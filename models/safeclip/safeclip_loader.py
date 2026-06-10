"""Safe-CLIP (Poppi et al., ECCV 2025) -- training-free text-encoder swap for SD v1.x.

Safe-CLIP fine-tunes a CLIP ViT-L/14 to redirect NSFW text/image embeddings to safe regions.
Public weights: `aimagelab/safeclip_vit-l_14` (a full transformers `CLIPModel`, text hidden_size
768 -> dimension-compatible with the SD v1.x text encoder). Phase 0 verified availability/dims.

We load that checkpoint and copy its fine-tuned text transformer into the SD pipeline's existing
`CLIPTextModel` (preserving the exact architecture/config the U-Net expects), i.e. the safety
intervention is purely at the CLIP text encoder; the U-Net and VAE are untouched. Inference-only.

Used by eval/xeval.py (kind="safeclip") and eval/eval_coco.py.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SAFECLIP_ID = "aimagelab/safeclip_vit-l_14"


def load_safeclip_text_encoder(pipe, safeclip_id: str = SAFECLIP_ID):
    """Replace `pipe.text_encoder`'s transformer weights with Safe-CLIP's fine-tuned text model.

    The checkpoint is a full CLIPModel; we copy only its `.text_model` (the CLIPTextTransformer)
    into the SD pipe's CLIPTextModel, which shares that submodule structure. Returns pipe.
    """
    import torch
    from transformers import CLIPModel

    safeclip = CLIPModel.from_pretrained(safeclip_id, torch_dtype=pipe.text_encoder.dtype)
    # The SD pipe's text_encoder is a CLIPTextModel; in this transformers build its state_dict is
    # flattened (no `.text_model` attribute / prefix) and matches CLIPModel.text_model key-for-key
    # (both: embeddings.* ... final_layer_norm.*). Load the fine-tuned text transformer directly.
    src = safeclip.text_model.state_dict()
    missing, unexpected = pipe.text_encoder.load_state_dict(src, strict=False)
    if missing or unexpected:
        logger.warning(f"[safeclip] load_state_dict non-strict: "
                       f"{len(missing)} missing, {len(unexpected)} unexpected keys")
        n_loaded = len(src) - len(unexpected)
        if n_loaded < 0.5 * len(src):
            raise RuntimeError(f"[safeclip] only {n_loaded}/{len(src)} text tensors matched "
                               f"-- checkpoint structure incompatible")
    pipe.text_encoder = pipe.text_encoder.to(device=pipe.device, dtype=pipe.text_encoder.dtype)
    del safeclip
    torch.cuda.empty_cache()
    logger.info(f"[safeclip] swapped SD text encoder with {safeclip_id} text model")
    return pipe
