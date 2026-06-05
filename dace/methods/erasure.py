"""DACE erasure losses (gradient flows through the trainable encoder).

Given the adversary subspace U (D,k) computed under no_grad from the CURRENT
embeddings, the eraser updates the encoder so that:
  - forget's projection onto U collapses to the retain centroid    (L_forget)
  - forget's component ORTHOGONAL to U stays at frozen              (L_ortho, anti-rerouting)
  - retain embeddings stay at frozen                               (L_retain)

The moving U (recomputed every few steps) + the pinned orthogonal complement make
rerouting unproductive: concept that leaks into a new direction is caught by the next
subspace refresh, and the non-concept subspace is held fixed, so there is nowhere to
hide. This directly attacks the empirically identified failure mode (residual_var flat
=> rerouting) that defeats static single-direction erasure (FCF-P / CNP).

Pure functions. PyTorch only.

Callers:
  - core/trainer.py
  - tests/test_dace.py
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .adversary import pool


def dace_losses(
    zf_cur: torch.Tensor,     # (Bf, L, D) current forget (grad)
    zf_frozen: torch.Tensor,  # (Bf, L, D) frozen forget (no grad)
    zr_cur: torch.Tensor,     # (Br, L, D) current retain (grad)
    zr_frozen: torch.Tensor,  # (Br, L, D) frozen retain (no grad)
    U: torch.Tensor,          # (D, k) orthonormal adversary subspace (detached)
    *,
    pool_mode: str = "mean",
    retain_centroid: Optional[torch.Tensor] = None,  # (D,) frozen retain centroid
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (L_forget, L_ortho, L_retain) scalars."""
    U = U.detach()
    pf = pool(zf_cur, pool_mode)                       # (Bf, D) grad
    pf_fz = pool(zf_frozen, pool_mode).detach()        # (Bf, D)
    pr_fz = pool(zr_frozen, pool_mode).detach()        # (Br, D)

    if retain_centroid is None:
        retain_centroid = pr_fz.mean(dim=0)            # (D,)
    retain_centroid = retain_centroid.detach()

    # (1) forget projection within concept subspace -> retain centroid
    proj_f = pf @ U                                    # (Bf, k)
    proj_c = retain_centroid @ U                       # (k,)
    L_forget = ((proj_f - proj_c.unsqueeze(0)) ** 2).sum(dim=1).mean()

    # (2) forget orthogonal complement pinned to frozen (anti-rerouting)
    of = pf - (pf @ U) @ U.T                           # (Bf, D)
    of_fz = pf_fz - (pf_fz @ U) @ U.T
    L_ortho = ((of - of_fz) ** 2).sum(dim=1).mean()

    # (3) retain preserved (full-sequence MSE -> minimal collateral, protects quality)
    L_retain = ((zr_cur - zr_frozen.detach()) ** 2).mean()

    return L_forget, L_ortho, L_retain


def dace_concept_losses(
    ze_cur: torch.Tensor,   # (B,L,D) explicit current (grad)
    ze_fz: torch.Tensor,    # (B,L,D) explicit frozen (no grad)
    zn_cur: torch.Tensor,   # (B,L,D) neutral (concept-stripped) current (grad)
    zn_fz: torch.Tensor,    # (B,L,D) neutral frozen (no grad)
    zr_cur: torch.Tensor,   # (Br,L,D) retain current (grad)
    zr_fz: torch.Tensor,    # (Br,L,D) retain frozen (no grad)
    U: torch.Tensor,        # (D,k) concept subspace (detached)
    *,
    pool_mode: str = "mean",
):
    """Concept-axis DACE losses (corrected after P0/P0b).

    L_forget: make adding the concept word NOT move the embedding inside U
              (drive concept_shift -> 0 within the live concept subspace).
    L_ortho : pin explicit's non-concept component to frozen (anti-rerouting).
    L_retain: preserve neutral content AND general retain (no content destruction).
    """
    U = U.detach()
    pe = pool(ze_cur, pool_mode)
    pn = pool(zn_cur, pool_mode)
    pe_fz = pool(ze_fz, pool_mode).detach()

    shift = pe - pn                                  # (B,D) concept shift (grad)
    proj = shift @ U                                 # (B,k)
    L_forget = (proj ** 2).sum(dim=1).mean()

    oe = pe - (pe @ U) @ U.T
    oe_fz = pe_fz - (pe_fz @ U) @ U.T
    L_ortho = ((oe - oe_fz) ** 2).sum(dim=1).mean()

    L_retain = ((zn_cur - zn_fz.detach()) ** 2).mean() + ((zr_cur - zr_fz.detach()) ** 2).mean()
    return L_forget, L_ortho, L_retain
