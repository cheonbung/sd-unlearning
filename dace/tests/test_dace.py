"""DACE unit tests (synthetic, CPU, fast). Run: pytest dace/tests -v"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dace/

from methods.adversary import pool, discriminative_subspace, SubspaceTracker
from methods.erasure import dace_losses
from methods import diagnostics as diag


def test_pool_shapes():
    z = torch.randn(5, 7, 16)
    assert pool(z, "mean").shape == (5, 16)
    assert pool(z, "last").shape == (5, 16)
    assert pool(torch.randn(5, 16)).shape == (5, 16)


def test_subspace_orthonormal():
    torch.manual_seed(0)
    D, k = 16, 4
    zf = torch.randn(32, D) + 3.0
    zr = torch.randn(32, D)
    U = discriminative_subspace(zf, zr, k=k, ridge=1e-3)
    assert U.shape == (D, k)
    gram = U.T @ U
    assert torch.allclose(gram, torch.eye(k), atol=1e-4)


def test_separability_auc_extremes():
    torch.manual_seed(0)
    D = 16
    # well separated -> high AUC
    zf = torch.randn(40, D) + 5.0
    zr = torch.randn(40, D) - 5.0
    assert diag.linear_separability_auc(zf, zr) > 0.9
    # identical distribution -> ~0.5
    a = torch.randn(80, D)
    auc = diag.linear_separability_auc(a[:40], a[40:])
    assert 0.3 < auc < 0.7


def test_dace_losses_identical_retain_zero():
    torch.manual_seed(0)
    D, k = 16, 3
    zf = torch.randn(8, 1, D)
    zr = torch.randn(8, 1, D)
    U = discriminative_subspace(pool(zf), pool(zr), k=k)
    Lf, Lo, Lr = dace_losses(zf, zf, zr, zr, U)
    assert Lr.item() < 1e-6           # current==frozen retain -> 0
    assert Lf.item() >= 0 and Lo.item() < 1e-6


def test_gradient_reduces_separability():
    torch.manual_seed(0)
    D = 16
    base = torch.randn(40, D)
    concept = torch.zeros(D); concept[0] = 6.0
    zr = (base).unsqueeze(1)                      # (40,1,D) retain
    zf0 = (base + concept).unsqueeze(1)           # (40,1,D) forget (separable)
    zf = zf0.clone().requires_grad_(True)
    zr_fz = zr.clone(); zf_fz = zf0.clone()
    auc_before = diag.linear_separability_auc(pool(zf.detach()), pool(zr))
    opt = torch.optim.Adam([zf], lr=0.2)
    for _ in range(120):
        U = discriminative_subspace(pool(zf.detach()), pool(zr), k=4)
        Lf, Lo, Lr = dace_losses(zf, zf_fz, zr, zr_fz, U)
        loss = Lf + 0.5 * Lo
        opt.zero_grad(); loss.backward(); opt.step()
    auc_after = diag.linear_separability_auc(pool(zf.detach()), pool(zr))
    assert auc_after < auc_before - 0.1          # separability dropped
    assert auc_after < 0.75


def test_tracker_passthrough():
    U = torch.linalg.qr(torch.randn(16, 4))[0][:, :4]
    assert torch.allclose(SubspaceTracker(0.0).update(U), U)
    out = SubspaceTracker(0.9).update(U)
    assert out.shape == U.shape
