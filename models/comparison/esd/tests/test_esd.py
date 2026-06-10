"""ESD unit tests (CPU, fast). Run: pytest models/comparison/esd/tests -v

Validates the trainable-parameter selection logic without loading SD weights (uses a synthetic
module that mimics the diffusers naming: attn1=self, attn2=cross).
"""
import sys
from pathlib import Path

import pytest
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esd_params import set_trainable_esd, VALID_METHODS  # noqa: E402


def _synthetic_unet():
    """Module with attn1 (self) and attn2 (cross) projections plus a non-attn conv, named like
    diffusers SD U-Net submodules."""
    m = nn.Module()
    m.down_attn1_to_k = nn.Linear(8, 8)   # self-attention
    m.down_attn2_to_k = nn.Linear(8, 8)   # cross-attention
    m.down_attn2_to_v = nn.Linear(8, 8)   # cross-attention
    m.mid_resnet_conv = nn.Conv2d(4, 4, 3)  # non-attention
    return m


def test_noxattn_excludes_cross_attn():
    m = _synthetic_unet()
    set_trainable_esd(m, "noxattn")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all("attn2" not in n for n in trainable)
    assert any("attn1" in n for n in trainable)      # self-attn trained
    assert any("resnet" in n for n in trainable)     # conv trained


def test_xattn_only_cross_attn():
    m = _synthetic_unet()
    set_trainable_esd(m, "xattn")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all("attn2" in n for n in trainable)


def test_full_trains_everything():
    m = _synthetic_unet()
    n = set_trainable_esd(m, "full")
    assert n == sum(p.numel() for p in m.parameters())


def test_noxattn_and_xattn_are_disjoint_and_complete():
    m = _synthetic_unet()
    nox = set_trainable_esd(m, "noxattn")
    xa = set_trainable_esd(m, "xattn")
    total = sum(p.numel() for p in m.parameters())
    assert nox + xa == total          # partition of the param space


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        set_trainable_esd(_synthetic_unet(), "bogus")
    assert "noxattn" in VALID_METHODS and "xattn" in VALID_METHODS
