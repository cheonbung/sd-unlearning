"""Unit tests for methods/ot_noise.py (N6)."""

from __future__ import annotations

import json
import random

import pytest
import torch

from methods.ot_noise import (
    OTNoiseConfig,
    OTNoiseResult,
    _random_noise_text,
    sinkhorn_divergence,
    sliced_wasserstein,
)


def test_sinkhorn_zero_distance_when_identical():
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    d = sinkhorn_divergence(x, x.clone(), epsilon=0.1, n_iter=30)
    assert d.item() < 0.5


def test_sinkhorn_positive_when_different():
    """Sinkhorn divergence is positive when distributions differ.

    Use a moderate shift and matching epsilon so the kernel doesn't underflow
    (exp(-large_cost/small_eps) -> 0 numerically).
    """
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = torch.randn(8, 4) + 1.0  # moderate shift
    d = sinkhorn_divergence(x, y, epsilon=1.0, n_iter=30)
    assert d.item() > 0.0


def test_sliced_wasserstein_zero_when_identical():
    torch.manual_seed(0)
    x = torch.randn(16, 8)
    d = sliced_wasserstein(x, x.clone(), n_projections=20)
    assert d.item() < 1e-6


def test_sliced_wasserstein_handles_unequal_sizes():
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = torch.randn(12, 4)
    d = sliced_wasserstein(x, y, n_projections=10)
    assert d.item() >= 0.0


def test_random_noise_text_length():
    rng = random.Random(42)
    s = _random_noise_text(5, rng)
    assert len(s) == 5


def test_random_noise_text_no_repeats():
    rng = random.Random(42)
    for _ in range(20):
        s = _random_noise_text(5, rng)
        assert len(set(s)) == len(s)


def test_random_noise_text_mixed_types():
    import string
    sym = "!@#$%^&*+-="
    rng = random.Random(42)
    for _ in range(20):
        s = _random_noise_text(5, rng)
        assert any(c in sym for c in s)
        assert any(c in string.ascii_letters for c in s)
        assert any(c in string.digits for c in s)


def test_ot_result_round_trip(tmp_path):
    cfg = OTNoiseConfig(n_candidates=10, n_select=3, max_len=5)
    original = OTNoiseResult(
        noise_prompts=["abc12", "x9$Bz", "Q&7eW"],
        explicit_prompts=["explicit 1", "explicit 2"],
        wasserstein_distance=0.1234,
        candidate_scores=[0.1, 0.2, 0.3],
        config=cfg,
    )

    path = tmp_path / "ot.json"
    original.save(str(path))
    assert path.exists()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["noise_prompts"] == original.noise_prompts
    assert data["wasserstein_distance"] == pytest.approx(0.1234, abs=1e-9)

    reloaded = OTNoiseResult.load(str(path))
    assert reloaded.noise_prompts == original.noise_prompts
    assert reloaded.explicit_prompts == original.explicit_prompts
    assert reloaded.wasserstein_distance == pytest.approx(original.wasserstein_distance)
    assert reloaded.candidate_scores == original.candidate_scores
    assert reloaded.config.n_candidates == cfg.n_candidates
    assert reloaded.config.metric == cfg.metric
