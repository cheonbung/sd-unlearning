"""ODACE UNET cross-attention editing + output-grounded losses.

Output-grounded erasure (vs all text-proxy baselines): make ONLY the UNET cross-attention
K/V projections trainable (UCE's target, but optimized by gradient against an OUTPUT loss).
The forget-conditioned noise prediction is pulled toward a NEUTRAL target with ESD-style
negative-guidance amplification (eta) so the erasure signal is strong (raw forget->neutral
eps differences at a random timestep are tiny):

    target = eps_neutral + eta * (eps_neutral - eps_forget_frozen)
    L_forget = MSE(eps_forget, target.detach())   # push forget PAST neutral, away from concept
    L_retain = MSE(eps_retain, eps_retain_frozen.detach())

This optimizes the noise prediction that determines the generated image (the quantity ASR
measures), not a text-embedding summary (which this session proved underdetermines ASR).

Callers: core/trainer.py, tests/test_odace.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def set_trainable_cross_attn_kv(unet, include_q_out: bool = False) -> int:
    """Freeze all UNET params, then unfreeze cross-attention (attn2) projections.

    include_q_out=False (default): only to_k & to_v (UCE target / minimal edit).
    include_q_out=True: full cross-attn to_q, to_k, to_v, to_out (ESD-x; stronger edit).
    Returns the trainable param count.
    """
    for p in unet.parameters():
        p.requires_grad_(False)
    n = 0
    for name, module in unet.named_modules():
        is_cross = getattr(module, "is_cross_attention", None)
        if is_cross is None:
            is_cross = name.endswith("attn2")
        if is_cross and hasattr(module, "to_k") and hasattr(module, "to_v"):
            mods = [module.to_k, module.to_v]
            if include_q_out:
                if hasattr(module, "to_q"):
                    mods.append(module.to_q)
                if getattr(module, "to_out", None) is not None:
                    mods.append(module.to_out)
            for m in mods:
                for p in m.parameters():
                    p.requires_grad_(True); n += p.numel()
    return n


def eps_match_losses(eps_forget, eps_neutral_frozen, eps_forget_frozen,
                     eps_retain, eps_retain_frozen, eta: float = 1.0):
    """ESD-style negative-guidance output loss (see module docstring)."""
    en = eps_neutral_frozen.detach().float()
    ef = eps_forget_frozen.detach().float()
    target = en + eta * (en - ef)
    L_forget = F.mse_loss(eps_forget, target)
    L_retain = F.mse_loss(eps_retain, eps_retain_frozen.detach().float())
    return L_forget, L_retain
