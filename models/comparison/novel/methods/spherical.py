"""N5 — Riemannian Geodesic Forgetting (RG-FCF).

Replaces FCF-P's Euclidean projection (Eq. 6 in the FCF paper) with a
spherical geodesic step on the unit hypersphere.

Theoretical motivation:
  CLIP text embeddings live on a narrow cone of the embedding space
  (Ethayarajh, EMNLP 2019). A Euclidean subtraction can push the cleaned
  target *outside* this cone, producing OOD inputs to the UNet. A spherical
  geodesic step stays on the manifold.

Callers:
  - methods/__init__.py (re-export)
  - methods/trainer.py (NovelFCFTrainer._compute_cleaned_projection_target)
  - tests/test_spherical.py

Data formats: PyTorch tensors only, no file I/O.

Reference: research idea N5 in .claude/plan/fcf-research-ideas-v2.md
"""

from __future__ import annotations

import torch


@torch.no_grad()
def spherical_cleaned_target(
    target_mean: torch.Tensor,
    concept_mean: torch.Tensor,
    eta_clean: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Spherical analog of FCF-P's cleaned target.

    Treats the flattened (L, D) tensor as a single vector on the unit sphere.

    Steps:
      1. Normalize both to unit length (treat as points on S^{LD-1})
      2. Compute log map at target pointing toward concept (tangent vector)
      3. Take a step of -eta_clean along this tangent (away from concept)
      4. Project back via exp map
      5. Rescale to original magnitude

    Args:
        target_mean:  Implicit-concept mean embedding, shape (L, D)
        concept_mean: Explicit-concept mean embedding, shape (L, D)
        eta_clean:    Step size (analog of mu_p in FCF-P; 0..1 typical)
        eps:          Numerical stability threshold

    Returns:
        cleaned: shape (L, D), same magnitude scale as target_mean
    """
    assert target_mean.shape == concept_mean.shape, (
        f"shape mismatch: target={target_mean.shape} concept={concept_mean.shape}"
    )

    original_shape = target_mean.shape
    v_t = target_mean.reshape(-1)
    v_c = concept_mean.reshape(-1)

    t_mag = v_t.norm()
    c_mag = v_c.norm()
    if t_mag < eps or c_mag < eps:
        return target_mean

    u_t = v_t / t_mag
    u_c = v_c / c_mag

    cos_angle = (u_t * u_c).sum().clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_angle)
    sin_theta = torch.sin(theta)

    if sin_theta.abs() < eps:
        return target_mean

    # Log map: tangent vector at u_t pointing toward u_c.
    log_uc = (u_c - cos_angle * u_t) * (theta / sin_theta)

    # Take a step of -eta_clean along the log direction (move AWAY from concept).
    step = -eta_clean * log_uc
    step_norm = step.norm()
    if step_norm < eps:
        return target_mean

    step_dir = step / step_norm

    # Exp map: u_t moved by `step` on the sphere.
    new_u = torch.cos(step_norm) * u_t + torch.sin(step_norm) * step_dir

    # Rescale to original target magnitude (preserves embedding norm scale).
    cleaned_flat = new_u * t_mag

    return cleaned_flat.reshape(original_shape)


@torch.no_grad()
def euclidean_cleaned_target(
    target_mean: torch.Tensor,
    concept_mean: torch.Tensor,
    eta_clean: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference Euclidean projection (matches core.trainer FCF-P).

    Provided for unit-test comparison with the spherical variant.
    """
    if concept_mean.norm() < eps:
        return target_mean
    concept_norm = concept_mean / concept_mean.norm()
    proj_length = (target_mean * concept_norm).sum()
    projection = proj_length * concept_norm
    return target_mean - eta_clean * projection


def manifold_cleaned_target(
    target_mean: torch.Tensor,
    concept_mean: torch.Tensor,
    eta_clean: float,
    manifold: str = "spherical",
) -> torch.Tensor:
    """Dispatch by manifold name (used by NovelFCFTrainer)."""
    if manifold == "spherical":
        return spherical_cleaned_target(target_mean, concept_mean, eta_clean)
    if manifold == "euclidean":
        return euclidean_cleaned_target(target_mean, concept_mean, eta_clean)
    raise ValueError(
        f"Unknown manifold: {manifold!r}. Choose 'spherical' or 'euclidean'."
    )
