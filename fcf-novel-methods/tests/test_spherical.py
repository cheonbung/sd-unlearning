"""Unit tests for methods/spherical.py (N5)."""

from __future__ import annotations

import pytest
import torch

from methods.spherical import (
    euclidean_cleaned_target,
    manifold_cleaned_target,
    spherical_cleaned_target,
)


def test_spherical_preserves_shape(small_target, small_concept):
    cleaned = spherical_cleaned_target(small_target, small_concept, eta_clean=0.5)
    assert cleaned.shape == small_target.shape


def test_spherical_preserves_magnitude(small_target, small_concept):
    """Spherical step rescales output to original ||target_mean|| (per design)."""
    cleaned = spherical_cleaned_target(small_target, small_concept, eta_clean=0.5)
    assert torch.allclose(cleaned.norm(), small_target.norm(), atol=1e-4)


def test_spherical_moves_away_from_concept(small_target, small_concept):
    """After spherical step, cleaned should be FURTHER from concept than target was."""
    def cos(a, b):
        a = a.reshape(-1)
        b = b.reshape(-1)
        return (a * b).sum() / (a.norm() * b.norm() + 1e-9)

    cleaned = spherical_cleaned_target(small_target, small_concept, eta_clean=0.5)
    assert cos(cleaned, small_concept) < cos(small_target, small_concept)


def test_spherical_eta_zero_is_identity(small_target, small_concept):
    """eta_clean = 0 should be (approximately) identity."""
    cleaned = spherical_cleaned_target(small_target, small_concept, eta_clean=0.0)
    assert torch.allclose(cleaned, small_target, atol=1e-4)


def test_spherical_handles_zero_target():
    """Zero target tensor -> returned unchanged (no division by zero)."""
    z_target = torch.zeros(4, 8)
    concept = torch.randn(4, 8)
    cleaned = spherical_cleaned_target(z_target, concept, eta_clean=0.5)
    assert torch.allclose(cleaned, z_target)


def test_spherical_handles_parallel_vectors():
    """When target == concept, log map degenerates -> safely returns target."""
    t = torch.randn(4, 8)
    cleaned = spherical_cleaned_target(t, t.clone(), eta_clean=0.5)
    assert torch.allclose(cleaned, t, atol=1e-4)


def test_spherical_assert_shape_mismatch():
    t = torch.randn(4, 8)
    c = torch.randn(4, 9)
    with pytest.raises(AssertionError):
        spherical_cleaned_target(t, c, eta_clean=0.5)


def test_euclidean_matches_fcf_p_formula(small_target, small_concept):
    """The local euclidean_cleaned_target reproduces FCF-P's Eq. 6."""
    cleaned = euclidean_cleaned_target(small_target, small_concept, eta_clean=0.7)

    # Re-derive the same answer manually using the paper formula.
    cn = small_concept / small_concept.norm()
    proj = (small_target * cn).sum() * cn
    expected = small_target - 0.7 * proj
    assert torch.allclose(cleaned, expected, atol=1e-5)


def test_manifold_dispatch(small_target, small_concept):
    sph = manifold_cleaned_target(small_target, small_concept, 0.5, manifold="spherical")
    euc = manifold_cleaned_target(small_target, small_concept, 0.5, manifold="euclidean")
    # The two methods should not produce identical outputs (different formulas).
    assert not torch.allclose(sph, euc, atol=1e-3)


def test_manifold_unknown_raises():
    t = torch.randn(4, 8)
    c = torch.randn(4, 8)
    with pytest.raises(ValueError, match="Unknown manifold"):
        manifold_cleaned_target(t, c, 0.5, manifold="hyperbolic")
