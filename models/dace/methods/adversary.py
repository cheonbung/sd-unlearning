"""DACE adversary: closed-form discriminative subspace + optional EMA tracking.

The "adversary" (inner max of the DACE minimax) finds the k-dim subspace U in the
pooled embedding space R^D that best separates forget from retain. Solved in CLOSED
FORM each refresh (no inner SGD): whiten by the retain within-class scatter, then take
the top-k directions of (between-class outer product + forget covariance). This is the
empirical separability diagnostic promoted to a live training target.

Pure functions + a light EMA tracker. PyTorch tensors only, no file I/O.

Callers:
  - methods/erasure.py
  - core/trainer.py
  - experiments/p0_crossmethod_diag.py
  - tests/test_dace.py
"""
from __future__ import annotations

import torch


def pool(z: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """(B, L, D) -> (B, D). 'mean' over tokens or 'last' token. (B,D) passes through."""
    if z.dim() == 2:
        return z
    if mode == "mean":
        return z.mean(dim=1)
    if mode == "last":
        return z[:, -1, :]
    raise ValueError(f"unknown pool mode: {mode}")


@torch.no_grad()
def discriminative_subspace(
    zf: torch.Tensor,   # (Nf, D) pooled forget (detached)
    zr: torch.Tensor,   # (Nr, D) pooled retain (detached)
    k: int = 4,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """Closed-form top-k subspace where forget separates from retain.

    M  = (mu_f - mu_r)(mu_f - mu_r)^T + Cov(forget)      # 1st + 2nd order concept structure
    W  = (Cov(retain) + ridge*I)^{-1/2}                  # whiten by retain scatter
    U  = top-k eigenvectors of (W M W), mapped back to original space + orthonormalized.

    Returns U: (D, k) orthonormal columns (the live concept subspace).
    """
    D = zf.shape[1]
    dtype = zf.dtype
    mu_f = zf.mean(dim=0)
    mu_r = zr.mean(dim=0)
    d = (mu_f - mu_r).unsqueeze(1)                          # (D,1)

    fc = zf - mu_f
    Cf = (fc.T @ fc) / max(1, zf.shape[0] - 1)             # (D,D)
    rc = zr - mu_r
    Cr = (rc.T @ rc) / max(1, zr.shape[0] - 1)             # (D,D)

    M = d @ d.T + Cf                                        # (D,D)
    eye = torch.eye(D, device=zf.device, dtype=dtype)
    Cr_reg = Cr + ridge * eye

    ev, evec = torch.linalg.eigh(Cr_reg)
    ev = ev.clamp_min(ridge)
    W = (evec * ev.rsqrt().unsqueeze(0)) @ evec.T          # (D,D) symmetric inverse sqrt

    Mw = W @ M @ W
    Mw = 0.5 * (Mw + Mw.T)                                  # symmetrize
    _, vecs = torch.linalg.eigh(Mw)                        # ascending eigenvalues
    k = min(k, D)
    topk = vecs[:, -k:]                                     # (D,k) whitened space
    U = W @ topk                                            # back to original space
    Q, _ = torch.linalg.qr(U)                              # orthonormalize
    return Q[:, :k]


class SubspaceTracker:
    """Optional EMA smoothing of the subspace across refreshes (stabilizes the pursuit).

    decay <= 0 -> use the fresh closed-form subspace as-is (no smoothing).
    decay  > 0 -> EMA on the projector P = U U^T, then re-extract top-k eigenvectors.
    """

    def __init__(self, decay: float = 0.0):
        self.decay = decay
        self._P = None  # (D,D) EMA projector

    @torch.no_grad()
    def update(self, U: torch.Tensor) -> torch.Tensor:
        if self.decay <= 0.0:
            return U
        P = U @ U.T
        if self._P is None:
            self._P = P
        else:
            self._P = self.decay * self._P + (1.0 - self.decay) * P
        k = U.shape[1]
        _, vecs = torch.linalg.eigh(0.5 * (self._P + self._P.T))
        return vecs[:, -k:]


@torch.no_grad()
def concept_subspace(d: torch.Tensor, k: int = 4, energy: float = 0.0,
                     k_max: int = 32) -> torch.Tensor:
    """Top-k principal directions of concept-shift vectors d (N, D) via SVD.

    d_i = z_explicit_i - z_neutral_i (the embedding move caused by the concept word).
    These directions -- NOT the forget-vs-retain axis -- are what the P0b experiment
    found to track ASR (Spearman 0.821). U spans the live concept axis to be erased.

    Direction-B (adaptive rank): when `energy` in (0,1) is given, the rank is chosen as
    the smallest k whose cumulative singular-value energy sum_{i<=k} s_i^2 / sum s_i^2 >=
    energy (capped at k_max), so the erased subspace captures (1-eps) of the concept-shift
    variance instead of a fixed, possibly-too-small k. energy<=0 -> fixed `k` (legacy).

    Returns U: (D, k) orthonormal columns.
    """
    D = d.shape[1]
    # center is intentionally NOT removed: the mean shift is itself concept signal
    _, S, Vt = torch.linalg.svd(d, full_matrices=False)
    if energy and 0.0 < energy < 1.0:
        e = (S ** 2).cumsum(0) / (S ** 2).sum().clamp_min(1e-12)
        k = int((e < energy).sum().item()) + 1          # smallest k reaching `energy`
        k = min(k, k_max)
    k = min(max(1, k), D, d.shape[0])
    return Vt[:k].T.contiguous()  # (D, k)
