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


def cnp_loss_multi(embeddings: torch.Tensor, concept_dirs: torch.Tensor,
                   weights=None) -> torch.Tensor:
    """N17 / multi-CONCEPT: minimize squared projection onto each concept direction.

    weights: optional per-concept weights (len K). Up-weighting a hard concept (e.g. nudity)
    counters the dilution that makes naive equal-weight multi-concept erasure under-erase it
    (v1 finding: nudity barely moved while style over-erased). None -> equal weights (orig /K).

    Args:
        embeddings:   (B, L, D) — current encoder forget/implicit embeddings.
        concept_dirs: (K, L, D) — K unit-normalized concept directions. requires_grad=False.
        weights:      optional list[float] length K (per-concept weight).
    Returns:
        scalar loss: weighted mean over K directions of squared projection.
    """
    B, L, D = embeddings.shape
    z_flat = embeddings.reshape(B, L * D)
    K = concept_dirs.shape[0]
    if weights is None:
        weights = [1.0] * K
    loss = torch.tensor(0.0, device=embeddings.device)
    wsum = 0.0
    for k in range(K):
        c_flat = concept_dirs[k].detach().reshape(-1)
        proj = z_flat @ c_flat
        loss = loss + float(weights[k]) * (proj ** 2).mean()
        wsum += float(weights[k])
    return loss / max(wsum, 1e-8)


@torch.no_grad()
def compute_concept_direction_token_selective(
    embeddings: torch.Tensor,
    top_frac: float = 0.3,
) -> torch.Tensor:
    """W1: 토큰 위치별 분산으로 content 토큰만 선택 후 SVD.

    문제: (N, L*D) 전체 flatten SVD는 77개 토큰을 동등 취급 → padding(EOS) 토큰이
    대다수라 2~4개 의미 토큰의 개념 신호가 희석됨.
    해법: padding 토큰은 프롬프트 간 분산이 거의 0(항상 같은 EOS 임베딩).
    위치별 분산 상위 top_frac 만 남기고(=content 토큰) 나머지는 0으로 마스킹 후 SVD.
    → 임베딩만으로 마스크 불필요하게 content 위치를 자동 식별.

    Args:
        embeddings: (N, L, D) — frozen 인코더의 explicit 프롬프트 임베딩. N>=2.
        top_frac:   분산 상위로 유지할 토큰 위치 비율 (0<frac<=1).
    Returns:
        c_dir: (L, D) — content 위치에서 추출된 개념 방향 (padding 위치는 0).
    """
    N, L, D = embeddings.shape
    if N < 2:
        raise ValueError(f"compute_concept_direction_token_selective requires N>=2, got N={N}")

    # 위치별 분산: 각 위치 l에서 N개 D-벡터의 분산을 D에 대해 평균 → (L,)
    pos_var = embeddings.var(dim=0, unbiased=False).mean(dim=1)  # (L,)
    k_keep = max(1, int(round(L * top_frac)))
    keep_idx = torch.topk(pos_var, k_keep).indices                # (k_keep,)
    mask = torch.zeros(L, device=embeddings.device, dtype=embeddings.dtype)
    mask[keep_idx] = 1.0

    masked = embeddings * mask.view(1, L, 1)                       # padding 위치 0
    flat = masked.reshape(N, L * D)
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(flat_centered, full_matrices=False)
    return Vt[0].reshape(L, D)


def cnp_loss_margin(
    z_current: torch.Tensor,
    z_frozen: torch.Tensor,
    concept_dir: torch.Tensor,
    ortho_weight: float = 0.1,
) -> torch.Tensor:
    """W2: 사영 제거 + 직교 보완 공간 약한 앵커링 (재라우팅 억제).

    문제: base cnp_loss는 사영 스칼라만 0으로 → 개념이 c_dir 직교 방향으로
    재라우팅되어도 손실이 0(미지정 목표).
    DDF(ortho_weight=1.0 등가)는 과제약 → 개념 대부분 잔존(나쁜 결과).
    W2는 그 사이: 사영은 0으로 강하게, 직교 성분은 frozen에 *약하게*(λ작게) 앵커.
    → c_dir 성분 제거는 유지하되 다른 방향으로 새 개념 구조가 생기는 것 억제.

    L = mean(proj^2) + ortho_weight * mean(||z_orth - z_frozen_orth||^2)

    Args:
        z_current:    (B, L, D) — 현재 인코더 출력 (gradient).
        z_frozen:     (B, L, D) — frozen 인코더 출력 (no_grad).
        concept_dir:  (L, D) — unit-normalized 개념 방향. 고정.
        ortho_weight: 직교 앵커 가중 λ (0=base CNP, 1≈DDF). 권장 0.05~0.2.
    Returns:
        scalar loss.
    """
    B = z_current.shape[0]
    c = concept_dir.detach().reshape(-1)
    c = c / (c.norm() + 1e-12)

    zc = z_current.reshape(B, -1)                       # (B, L*D)
    zf = z_frozen.detach().reshape(B, -1)               # (B, L*D)

    proj_c = zc @ c                                      # (B,) — 현재 사영
    loss_proj = (proj_c ** 2).mean()

    # 직교 성분 차이 (사영 성분 제거 후 비교)
    zc_orth = zc - proj_c.unsqueeze(1) * c.unsqueeze(0)
    proj_f = zf @ c
    zf_orth = zf - proj_f.unsqueeze(1) * c.unsqueeze(0)
    loss_orth = ((zc_orth - zf_orth) ** 2).mean()

    return loss_proj + ortho_weight * loss_orth
