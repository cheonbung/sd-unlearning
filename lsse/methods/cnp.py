"""N7 — Concept Null-Space Projection (CNP).

FCF와의 차별점:
  FCF는 noise 프롬프트를 목표로 MSE 손실 사용.
  CNP는 noise 프롬프트 불필요 — explicit 임베딩의 주성분 방향(c_dir)을 SVD로
  1회 추출한 후, 학습 인코더가 해당 방향 성분을 0으로 수렴하도록 유도.
  목표가 기하학적으로 정의되므로 arbitrary noise 선택 문제가 사라짐.

Callers:
  - methods/lsse_trainer.py (LSSETrainer.__init__, train_step)
  - tests/test_lsse.py

Data formats: PyTorch tensors only, no file I/O.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_concept_direction(embeddings: torch.Tensor) -> torch.Tensor:
    """Explicit 프롬프트 임베딩에서 SVD로 개념 방향 추출.

    Args:
        embeddings: (N, L, D) — frozen 인코더의 explicit 프롬프트 임베딩 배치.
                    N >= 2 이어야 의미 있는 방향이 추출됨.
    Returns:
        c_dir: (L, D) unit-normalized 개념 방향 벡터 (첫 번째 우-특이벡터).
               SVD 특성상 이미 단위 벡터로 반환됨.
    """
    N, L, D = embeddings.shape
    if N < 2:
        raise ValueError(f"compute_concept_direction requires N >= 2, got N={N}")

    flat = embeddings.reshape(N, L * D)
    flat_centered = flat - flat.mean(dim=0, keepdim=True)

    _, _, Vt = torch.linalg.svd(flat_centered, full_matrices=False)
    return Vt[0].reshape(L, D)


@torch.no_grad()
def compute_concept_direction_macd(embeddings: torch.Tensor) -> torch.Tensor:
    """N10 MACD: 구면 정규화 후 SVD로 개념 방향 추출.

    CLIP 임베딩은 좁은 cone에 분포하므로 유클리드 PCA보다 구면 PCA가 더 정확한 방향을 준다.
    각 샘플을 단위 초구면에 사영 후 접선 공간에서 주성분 추출.

    Args:
        embeddings: (N, L, D) — frozen 인코더의 explicit 프롬프트 임베딩 배치.
    Returns:
        c_dir: (L, D) 개념 방향 벡터 (구면 PCA의 첫 번째 주성분).
    """
    N, L, D = embeddings.shape
    if N < 2:
        raise ValueError(f"compute_concept_direction_macd requires N >= 2, got N={N}")

    flat = embeddings.reshape(N, L * D)
    flat_unit = F.normalize(flat, p=2, dim=1)
    mean_unit = F.normalize(flat_unit.mean(dim=0, keepdim=True), p=2, dim=1)
    flat_centered = flat_unit - mean_unit

    _, _, Vt = torch.linalg.svd(flat_centered, full_matrices=False)
    return Vt[0].reshape(L, D)


def cnp_loss(embeddings: torch.Tensor, concept_dir: torch.Tensor) -> torch.Tensor:
    """Concept 방향 성분의 크기를 최소화하는 손실 (학습 중 gradient 흐름).

    개념 방향으로의 사영 스칼라의 제곱합을 최소화.
    noise 프롬프트 없이 기하학적으로 망각 방향을 정의.

    Args:
        embeddings: (B, L, D) — 현재 인코더의 forget/implicit 프롬프트 임베딩.
        concept_dir: (L, D) — unit-normalized 개념 방향. 학습 전 1회 계산, 고정.
                     requires_grad=False 이어야 gradient가 올바르게 흐름.
    Returns:
        scalar loss: 배치 평균 squared projection.
    """
    B, L, D = embeddings.shape
    c_flat = concept_dir.detach().reshape(-1)       # (L*D,) — 고정 방향
    z_flat = embeddings.reshape(B, L * D)           # (B, L*D)
    proj = z_flat @ c_flat                          # (B,) — 개념 방향 사영 스칼라
    return (proj ** 2).mean()


def cnp_loss_ddf(
    z_current: torch.Tensor,
    z_frozen: torch.Tensor,
    concept_dir: torch.Tensor,
) -> torch.Tensor:
    """N12 DDF: frozen 임베딩의 null-space 사영을 명시적 MSE 목표로 학습.

    기존 cnp_loss는 사영 스칼라 → 0 만 유도.
    DDF는 목표점을 명시적으로 정의:
      target = z_frozen - proj(z_frozen, c_dir)   (개념 방향 성분 제거된 원본)
    z_current 가 target 으로 수렴하도록 MSE 학습.
    더 명확한 기하학적 목표 → 더 빠른 수렴 + 잔여 방향 유지.

    Args:
        z_current: (B, L, D) — 현재 인코더 출력 (gradient 흐름).
        z_frozen:  (B, L, D) — 초기 frozen 인코더 출력 (no_grad).
        concept_dir: (L, D) — unit-normalized 개념 방향. 고정.
    Returns:
        scalar MSE loss.
    """
    c_flat = concept_dir.detach().reshape(-1)                        # (L*D,)
    z_f_flat = z_frozen.detach().reshape(z_frozen.shape[0], -1)      # (B, L*D)
    proj_scalar = z_f_flat @ c_flat                                   # (B,)
    target = z_f_flat - proj_scalar.unsqueeze(1) * c_flat.unsqueeze(0)  # (B, L*D)

    z_c_flat = z_current.reshape(z_current.shape[0], -1)             # (B, L*D)
    return F.mse_loss(z_c_flat, target)


@torch.no_grad()
def compute_concept_directions(embeddings: torch.Tensor, top_k: int = 3) -> torch.Tensor:
    """N17 Multi-Direction CNP: top-K SVD 방향 추출.

    단일 방향(c_dir)만 사용하면 개념이 여러 방향에 분산된 경우 망각 불완전.
    Top-K 우-특이벡터를 모두 추출하여 개념의 다차원 표현을 포괄.

    Args:
        embeddings: (N, L, D) — frozen 인코더의 explicit 프롬프트 임베딩.
        top_k:      추출할 개념 방향 수.
    Returns:
        concept_dirs: (K, L, D) — unit-normalized K개 개념 방향.
    """
    N, L, D = embeddings.shape
    if N < 2:
        raise ValueError(f"compute_concept_directions requires N >= 2, got N={N}")

    flat = embeddings.reshape(N, L * D)
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(flat_centered, full_matrices=False)
    actual_k = min(top_k, Vt.shape[0])
    return Vt[:actual_k].reshape(actual_k, L, D)


def cnp_loss_multi(embeddings: torch.Tensor, concept_dirs: torch.Tensor) -> torch.Tensor:
    """N17: K개 개념 방향 모두에 대한 사영 크기 최소화.

    각 방향의 squared projection 평균을 구하고 K개 방향에 대해 평균.
    단일 방향 CNP보다 개념의 다차원 표현을 더 완전히 제거.

    Args:
        embeddings:   (B, L, D) — 현재 인코더의 forget/implicit 프롬프트 임베딩.
        concept_dirs: (K, L, D) — K개 unit-normalized 개념 방향. requires_grad=False.
    Returns:
        scalar loss: K개 방향 평균 squared projection.
    """
    B, L, D = embeddings.shape
    z_flat = embeddings.reshape(B, L * D)
    K = concept_dirs.shape[0]
    loss = torch.tensor(0.0, device=embeddings.device)
    for k in range(K):
        c_flat = concept_dirs[k].detach().reshape(-1)
        proj = z_flat @ c_flat
        loss = loss + (proj ** 2).mean()
    return loss / K
