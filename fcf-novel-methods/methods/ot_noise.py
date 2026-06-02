"""N6 — Optimal-Transport-derived Noise Prompts (OT-FCF).

Replaces the ad-hoc 5-character random noise (paper appendix spec) with
*learned* noise prompts that maximize a Wasserstein distance from the
(frozen-encoded) explicit-prompt embedding distribution.

Theoretical motivation: there is no a priori reason a random 5-char string
is the optimal "forgetting destination". OT gives a principled choice of
endpoint in embedding space.

This module deliberately avoids the external POT library — it uses a
self-contained Sinkhorn iteration plus a sliced-Wasserstein loss so the
sub-project has zero extra dependencies.

Callers:
  - train.py (loads OTNoiseResult from JSON to override dataset.noise_prompts)
  - scripts/learn_ot_noise.py (instantiates OTNoiseLearner, saves OTNoiseResult)
  - tests/test_ot_noise.py

Data formats:
  JSON schema (OTNoiseResult.save / OTNoiseResult.load):
    {
      "noise_prompts":        ["a7$Bx", "Q1#mZ", ...],
      "explicit_prompts":     [...],
      "wasserstein_distance": 0.123,
      "candidate_scores":     [float, ...],
      "config":               { ... OTNoiseConfig dict ... }
    }
  No date fields.

Reference: research idea N6 in .claude/plan/fcf-research-ideas-v2.md
"""

from __future__ import annotations

import json
import os
import random
import string
from dataclasses import dataclass
from typing import List, Optional

import torch


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    n_iter: int = 50,
) -> torch.Tensor:
    """Sinkhorn divergence between two empirical distributions."""
    N, M = x.shape[0], y.shape[0]
    cost = torch.cdist(x, y, p=2) ** 2

    a = torch.full((N,), 1.0 / N, device=x.device, dtype=x.dtype)
    b = torch.full((M,), 1.0 / M, device=x.device, dtype=x.dtype)

    K = torch.exp(-cost / epsilon)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iter):
        u = a / (K @ v + 1e-30)
        v = b / (K.t() @ u + 1e-30)

    P = torch.diag(u) @ K @ torch.diag(v)
    return (P * cost).sum()


def sliced_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 50,
) -> torch.Tensor:
    """Sliced 2-Wasserstein distance (cheap; no Sinkhorn needed)."""
    D = x.shape[-1]
    proj = torch.randn(D, n_projections, device=x.device, dtype=x.dtype)
    proj = proj / proj.norm(dim=0, keepdim=True)

    x_proj = x @ proj
    y_proj = y @ proj

    x_sorted, _ = torch.sort(x_proj, dim=0)
    y_sorted, _ = torch.sort(y_proj, dim=0)

    if x_sorted.shape[0] != y_sorted.shape[0]:
        n = min(x_sorted.shape[0], y_sorted.shape[0])
        x_sorted = x_sorted[:n]
        y_sorted = y_sorted[:n]

    return ((x_sorted - y_sorted) ** 2).mean()


_SYMBOLS = "!@#$%^&*+-="
_LETTERS = string.ascii_letters
_DIGITS = string.digits
_FULL_CHARSET = list(_SYMBOLS + _LETTERS + _DIGITS)


def _random_noise_text(length: int, rng: random.Random) -> str:
    """Single random noise string (5 chars default, no repeats, mixed types)."""
    for _ in range(1000):
        sample = rng.sample(_FULL_CHARSET, length)
        if (any(c in _SYMBOLS for c in sample)
                and any(c in _LETTERS for c in sample)
                and any(c in _DIGITS for c in sample)):
            return "".join(sample)
    forced = [
        rng.choice(list(_SYMBOLS)),
        rng.choice(list(_LETTERS)),
        rng.choice(list(_DIGITS)),
    ]
    remaining = [c for c in _FULL_CHARSET if c not in forced]
    forced += rng.sample(remaining, length - len(forced))
    rng.shuffle(forced)
    return "".join(forced)


@dataclass
class OTNoiseConfig:
    n_candidates: int = 500
    n_select: int = 100
    max_len: int = 5
    n_projections: int = 50
    seed: int = 42
    metric: str = "sliced"
    sinkhorn_eps: float = 0.1
    sinkhorn_iter: int = 30


class OTNoiseLearner:
    """Learn an optimal noise-prompt vocabulary via OT in CLIP embedding space.

    Algorithm (candidate-search variant):
      1. Generate `n_candidates` random noise strings (paper spec)
      2. Encode each via frozen CLIP -> noise embeddings
      3. Encode all explicit prompts -> explicit embeddings
      4. Score each candidate by Wasserstein distance from explicit distribution
      5. Select top-`n_select` candidates with HIGHEST distance
      6. Return result + save to disk
    """

    def __init__(
        self,
        frozen_encoder,
        tokenizer,
        device,
        config: Optional[OTNoiseConfig] = None,
        max_token_length: int = 77,
    ):
        self.frozen_encoder = frozen_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or OTNoiseConfig()
        self.max_token_length = max_token_length

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_token_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.frozen_encoder(tokens.input_ids).last_hidden_state
        return out.mean(dim=1)  # (B, D)

    def _wasserstein(self, x: torch.Tensor, y: torch.Tensor) -> float:
        if self.config.metric == "sliced":
            return float(sliced_wasserstein(x, y, self.config.n_projections).item())
        if self.config.metric == "sinkhorn":
            return float(
                sinkhorn_divergence(
                    x, y,
                    epsilon=self.config.sinkhorn_eps,
                    n_iter=self.config.sinkhorn_iter,
                ).item()
            )
        raise ValueError(f"Unknown metric: {self.config.metric!r}")

    def learn(self, explicit_prompts: List[str]) -> "OTNoiseResult":
        rng = random.Random(self.config.seed)

        candidates = [
            _random_noise_text(self.config.max_len, rng)
            for _ in range(self.config.n_candidates)
        ]

        z_candidates = self._encode(candidates)
        z_explicit = self._encode(explicit_prompts)

        scores: List[float] = []
        for i in range(z_candidates.shape[0]):
            zc = z_candidates[i : i + 1].expand(z_explicit.shape[0], -1)
            scores.append(self._wasserstein(zc, z_explicit))

        sorted_idx = sorted(range(len(scores)), key=lambda i: -scores[i])
        selected = sorted_idx[: self.config.n_select]
        selected_prompts = [candidates[i] for i in selected]
        selected_scores = [scores[i] for i in selected]

        z_selected = z_candidates[selected]
        final_w = self._wasserstein(z_selected, z_explicit)

        return OTNoiseResult(
            noise_prompts=selected_prompts,
            explicit_prompts=list(explicit_prompts),
            wasserstein_distance=float(final_w),
            candidate_scores=selected_scores,
            config=self.config,
        )


@dataclass
class OTNoiseResult:
    """Output of OTNoiseLearner.learn(...). Save/load JSON format."""
    noise_prompts: List[str]
    explicit_prompts: List[str]
    wasserstein_distance: float
    candidate_scores: List[float]
    config: OTNoiseConfig

    def to_dict(self) -> dict:
        return {
            "noise_prompts": self.noise_prompts,
            "explicit_prompts": self.explicit_prompts,
            "wasserstein_distance": self.wasserstein_distance,
            "candidate_scores": self.candidate_scores,
            "config": self.config.__dict__,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "OTNoiseResult":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cfg = OTNoiseConfig(**d["config"])
        return cls(
            noise_prompts=d["noise_prompts"],
            explicit_prompts=d["explicit_prompts"],
            wasserstein_distance=d["wasserstein_distance"],
            candidate_scores=d.get("candidate_scores", []),
            config=cfg,
        )
