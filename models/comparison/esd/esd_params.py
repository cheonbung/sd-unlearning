"""Canonical ESD (Gandikota et al., ICCV 2023) trainable-parameter selection.

ESD fine-tunes a SUBSET of the SD U-Net by `train_method`. The two configurations from the
paper:
  - ESD-u ("unconditional", train_method='noxattn'): tune every U-Net param that is NOT in a
    cross-attention block (attn2). Global, prompt-independent erasure -> the standard choice for
    NSFW / nudity removal.
  - ESD-x (train_method='xattn'): tune ONLY cross-attention (attn2) params. Prompt-local
    erasure -> the choice for named artistic styles.

This is the CANON recipe and is deliberately distinct from ODACE (odace/methods/unet_edit.py),
which tunes only cross-attn K/V with an amplified eta as a *strengthened* variant. Do not
conflate the two.

Callers: models/comparison/esd/esd_trainer.py, models/comparison/esd/tests/test_esd.py
"""
from __future__ import annotations

VALID_METHODS = ("noxattn", "xattn", "full", "selfattn")


def _is_cross_attn_param(name: str) -> bool:
    """Cross-attention in diffusers SD U-Net lives under `.attn2.` (attn1 = self-attention)."""
    return "attn2" in name


def _is_self_attn_param(name: str) -> bool:
    return "attn1" in name


def set_trainable_esd(unet, train_method: str = "noxattn") -> int:
    """Freeze the whole U-Net, then unfreeze the param subset for `train_method`.

    Returns the trainable parameter count. Raises ValueError on an unknown method.
    """
    if train_method not in VALID_METHODS:
        raise ValueError(f"train_method must be one of {VALID_METHODS}, got {train_method!r}")

    for p in unet.parameters():
        p.requires_grad_(False)

    n = 0
    for name, p in unet.named_parameters():
        train = False
        if train_method == "full":
            train = True
        elif train_method == "noxattn":           # ESD-u: everything except cross-attn
            train = not _is_cross_attn_param(name)
        elif train_method == "xattn":             # ESD-x: only cross-attn
            train = _is_cross_attn_param(name)
        elif train_method == "selfattn":          # only self-attn
            train = _is_self_attn_param(name)
        if train:
            p.requires_grad_(True)
            n += p.numel()
    return n
