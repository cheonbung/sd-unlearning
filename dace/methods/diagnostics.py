"""DACE diagnostics (independent of lsse). Quantify separability the eraser must remove.

Key metric (training monitor + P0 cross-method test):
  linear_separability_auc -- HELD-OUT (cross-validated) AUC of the best linear probe
  separating forget vs retain in pooled embedding space. In-sample AUC overfits badly
  when D >> N (D=768, N~32 here) and saturates at ~1.0 for every checkpoint, so we fit
  the LDA direction on a random half and score the held-out half. 0.5 = indistinguishable
  (concept erased), 1.0 = perfectly separable.

Pure functions, PyTorch only.

Callers:
  - core/trainer.py
  - experiments/p0_crossmethod_diag.py
  - tests/test_dace.py
"""
from __future__ import annotations

import torch

from .adversary import pool


@torch.no_grad()
def linear_separability_auc(
    zf: torch.Tensor,   # (Nf, D) or (Nf, L, D)
    zr: torch.Tensor,   # (Nr, D) or (Nr, L, D)
    ridge: float = 1e-2,
    pool_mode: str = "mean",
    n_splits: int = 5,
    seed: int = 0,
) -> float:
    """Held-out cross-validated best-linear separability of forget vs retain (ROC-AUC)."""
    zf = pool(zf, pool_mode).float()
    zr = pool(zr, pool_mode).float()
    D = zf.shape[1]
    nf, nr = zf.shape[0], zr.shape[0]
    if nf < 2 or nr < 2:
        return 0.5
    g = torch.Generator(device="cpu").manual_seed(seed)
    eye = torch.eye(D, device=zf.device)
    aucs = []
    for _ in range(n_splits):
        pf = torch.randperm(nf, generator=g).to(zf.device)
        pr = torch.randperm(nr, generator=g).to(zr.device)
        hf, hr = max(1, nf // 2), max(1, nr // 2)
        f_tr, f_te = zf[pf[:hf]], zf[pf[hf:]]
        r_tr, r_te = zr[pr[:hr]], zr[pr[hr:]]
        if f_te.shape[0] == 0 or r_te.shape[0] == 0:
            continue
        mu_f, mu_r = f_tr.mean(0), r_tr.mean(0)
        ztr = torch.cat([f_tr - mu_f, r_tr - mu_r], 0)
        C = (ztr.T @ ztr) / max(1, ztr.shape[0] - 1) + ridge * eye
        w = torch.linalg.solve(C, mu_f - mu_r)
        sf, sr = f_te @ w, r_te @ w
        diff = sf.unsqueeze(1) - sr.unsqueeze(0)
        wins = (diff > 0).float().sum() + 0.5 * (diff == 0).float().sum()
        aucs.append((wins / (sf.numel() * sr.numel())).item())
    return sum(aucs) / len(aucs) if aucs else 0.5


@torch.no_grad()
def subspace_residual_energy(
    zf: torch.Tensor,   # (Nf, D) or (Nf, L, D)
    U: torch.Tensor,    # (D, k)
    pool_mode: str = "mean",
) -> float:
    """Fraction of forget variance left OUTSIDE the concept subspace U (high => rerouting)."""
    pf = pool(zf, pool_mode).float()
    pf = pf - pf.mean(dim=0, keepdim=True)
    total = (pf ** 2).sum().item() + 1e-12
    inside = ((pf @ U.float()) ** 2).sum().item()
    return max(0.0, (total - inside) / total)


@torch.no_grad()
def retain_drift(z_current: torch.Tensor, z_frozen: torch.Tensor) -> float:
    """Mean squared movement of retain embeddings vs frozen (lower = better preserved)."""
    return ((z_current.detach() - z_frozen.detach()) ** 2).mean().item()
