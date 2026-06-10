"""NovelFCFTrainer — subclass of the local FCFTrainer with N5/N6 hooks.

Why a subclass (not a re-implementation):
  The base FCFTrainer in core/ already implements Algorithms 1-3 exactly as
  in the paper. NovelFCFTrainer overrides ONLY the single method whose
  formula we want to change for N5 (manifold-aware projection).

Independence:
  This module imports from the LOCAL `core/` package, NOT from the parent
  thesis package `fcf/`. The base trainer is shipped inside this sub-project.

Backward compatibility:
  - manifold='euclidean' (default): identical behavior to FCFTrainer
  - manifold='spherical': Riemannian geodesic step (N5)
"""

from __future__ import annotations

import logging
from typing import List

import torch

from core.trainer import FCFTrainer

from .spherical import manifold_cleaned_target

logger = logging.getLogger(__name__)


class NovelFCFTrainer(FCFTrainer):
    """Drop-in replacement for FCFTrainer with experimental manifold support."""

    def __init__(self, *args, manifold: str = "euclidean", **kwargs):
        super().__init__(*args, **kwargs)
        if manifold not in {"euclidean", "spherical"}:
            raise ValueError(
                f"Unknown manifold: {manifold!r}. Choose 'euclidean' or 'spherical'."
            )
        self.manifold = manifold
        logger.info(f"[NovelFCFTrainer] manifold = {manifold}")

    @torch.no_grad()
    def _compute_cleaned_projection_target(
        self,
        implicit_concepts: List[str],
        concept_texts: List[str],
        eta_clean: float,
    ) -> torch.Tensor:
        """Override of FCF-P projection (Eq. 6).

        Euclidean path delegates to parent (identical behavior).
        Spherical path uses geodesic step (N5).
        """
        if self.manifold == "euclidean":
            return super()._compute_cleaned_projection_target(
                implicit_concepts, concept_texts, eta_clean
            )

        concept_feats = self._encode(concept_texts, encoder=self.frozen_encoder)
        concept_mean = concept_feats.mean(dim=0)

        target_feats = self._encode(implicit_concepts, encoder=self.frozen_encoder)
        target_mean = target_feats.mean(dim=0)

        return manifold_cleaned_target(
            target_mean=target_mean,
            concept_mean=concept_mean,
            eta_clean=eta_clean,
            manifold=self.manifold,
        )
