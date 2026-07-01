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


def cnp_loss_slerp(
    z_current: torch.Tensor,
    z_frozen: torch.Tensor,
    concept_dir: torch.Tensor,
) -> torch.Tensor:
    """Ring-A-Bell OOD-collapse fix — Phase 1: norm-preserving (manifold) erasure.

    Diagnosis: `margin` mode drives proj²→0 but leaves the read-out *magnitude* free, so on OOD
    high-norm gibberish tokens the conditioning rotates into a degenerate low/zero-energy direction
    and the UNet decodes garbage (no human). sph_ot (SLERP, on-sphere) never collapses → the clue.
    Fix: target = frozen read-out with the concept component removed, then RENORMALIZED back to the
    frozen per-sample read-out norm. Erases the concept DIRECTION while pinning the read-out energy
    ‖R‖²=cᵀMc to its frozen value (stay on the same sphere radius) → no magnitude collapse.

    Args:
        z_current: (B, L, D) current encoder read-out embeddings (caller already applied M^½).
        z_frozen:  (B, L, D) frozen encoder read-out embeddings (no_grad).
        concept_dir: (L, D) unit-normalized read-out concept direction. Fixed.
    Returns:
        scalar MSE loss to the norm-preserved concept-removed target.
    """
    B = z_current.shape[0]
    c = concept_dir.detach().reshape(-1)
    c = c / (c.norm() + 1e-12)
    zf = z_frozen.detach().reshape(B, -1)                 # (B, L*D)
    proj_f = zf @ c                                        # (B,)
    tgt = zf - proj_f.unsqueeze(1) * c.unsqueeze(0)        # concept removed
    tgt = tgt * (zf.norm(dim=1, keepdim=True) / (tgt.norm(dim=1, keepdim=True) + 1e-12))
    zc = z_current.reshape(B, -1)
    return F.mse_loss(zc, tgt)


def cnp_loss_redirect(
    z_current: torch.Tensor,
    z_frozen: torch.Tensor,
    concept_dir: torch.Tensor,
    anchor_coord: float,
    strength: float = 1.0,
) -> torch.Tensor:
    """Ring-A-Bell OOD-collapse fix — Phase 2/3: redirect concept axis to a benign anchor.

    Diagnosis: erasing "to 0" leaves the post-erasure direction underdetermined; on OOD tokens the
    model picks a degenerate (non-human) direction. Fix: instead of nulling the concept projection,
    SHIFT the concept-axis coordinate from its current value to the benign anchor's coordinate
    (mean read-out projection of safe/clothed-person prompts), leaving every off-concept component
    at its frozen value:  target = z_frozen + (anchor_coord − proj_f)·ĉ. This gives a concrete
    benign attractor, so nudity (and OOD gibberish, when fed through this loss) maps to a coherent
    off-concept point rather than collapsing. Parameter-free (FCF-E empirical-redirect analog in
    read-out space, toward a benign concept instead of noise).

    Args:
        z_current: (B, L, D) current read-out embeddings (caller applied M^½).
        z_frozen:  (B, L, D) frozen read-out embeddings (no_grad).
        concept_dir: (L, D) unit-normalized read-out concept direction. Fixed.
        anchor_coord: benign target coordinate along ĉ (precomputed scalar). Fixed.
    Returns:
        scalar MSE loss to the redirected target.
    """
    B = z_current.shape[0]
    c = concept_dir.detach().reshape(-1)
    c = c / (c.norm() + 1e-12)
    zf = z_frozen.detach().reshape(B, -1)                 # (B, L*D)
    proj_f = zf @ c                                        # (B,)
    # strength>1 = OVERSHOOT past the benign anchor toward the anti-concept side (FCF-P-style strong
    # projection): land at proj_f + strength*(anchor-proj_f). strength=1 -> benign mean (under-erases).
    target = zf + strength * (anchor_coord - proj_f).unsqueeze(1) * c.unsqueeze(0)
    zc = z_current.reshape(B, -1)
    return F.mse_loss(zc, target)


# --- Parameter-free concept-discriminative directions (S1/S2/S3) ----------------------------------
# These make erasure concept-SPECIFIC (forget-vs-retain) instead of max-variance, expanding the
# Pareto frontier (lower ASR at same utility) WITHOUT any tunable λ/β knob. Caller passes embeddings
# already in the desired space (CAP-CNP transforms by M^½ first, so directions live in read-out space).

@torch.no_grad()
def compute_concept_direction_contrastive(
    explicit: torch.Tensor, retain: torch.Tensor
) -> torch.Tensor:
    """S1: 대조적 개념 방향 = normalize(mean(explicit) − mean(retain)).

    max-variance(top-SVD) 대신 forget↔retain 평균 이동 축. 정의상 retain 공통 변동과 거의 직교
    → 개념만 제거하고 일반 콘텐츠 보존. 파라미터 없음(평균 차이 한 벡터).

    Args:
        explicit: (Ne, L, D) frozen explicit 임베딩 (또는 M^½ 변환된 것).
        retain:   (Nr, L, D) frozen retain 임베딩 (같은 공간).
    Returns:
        c_dir: (L, D) unit-normalized.
    """
    L, D = explicit.shape[1], explicit.shape[2]
    delta = explicit.mean(dim=0) - retain.mean(dim=0)        # (L, D)
    flat = delta.reshape(-1)
    return (flat / (flat.norm() + 1e-12)).reshape(L, D)


@torch.no_grad()
def orthogonalize_direction(c_dir: torch.Tensor, retain: torch.Tensor) -> torch.Tensor:
    """S2: c_dir에서 retain span 성분 제거 (Gram-Schmidt) 후 정규화.

    erasure 방향이 retain 부분공간과 0 성분이 되도록 보장 → 유틸리티 구조적 보존.
    retain 전체(절단 rank 노브 없음)의 centered SVD 기저에 대해 직교화.

    Args:
        c_dir:  (L, D) 개념 방향 (contrastive 또는 svd).
        retain: (Nr, L, D) retain 임베딩 (같은 공간).
    Returns:
        c_dir_orth: (L, D) unit-normalized, span(retain)에 직교.
    """
    L, D = c_dir.shape
    c = c_dir.reshape(-1)                                     # (L*D,)
    R = retain.reshape(retain.shape[0], -1)                   # (Nr, L*D)
    R = R - R.mean(dim=0, keepdim=True)
    # orthonormal basis of retain row-space (rows of Vt with non-trivial singular value)
    _, S, Vt = torch.linalg.svd(R, full_matrices=False)       # Vt: (k, L*D)
    if S.numel() > 0:
        keep = S > (S.max() * 1e-5)
        B = Vt[keep]                                          # (m, L*D) orthonormal
        if B.shape[0] > 0:
            c = c - B.t() @ (B @ c)                           # remove span(B) component
    return (c / (c.norm() + 1e-12)).reshape(L, D)


@torch.no_grad()
def compute_concept_direction_whitened(
    explicit: torch.Tensor, retain: torch.Tensor
) -> torch.Tensor:
    """S3: 화이트닝된 대조 방향 (Fisher 풍) = Σ_retain^{-1} (μ_explicit − μ_retain).

    retain 분산이 큰 방향(일반 콘텐츠)을 down-weight → UNet이 읽으면서 retain엔 안 쓰이는
    개념-특이 방향만 강조. Σ_retain은 D×D 풀드 공분산, pinv로 파라미터 없이 역행렬(ridge 불요).

    Args:
        explicit: (Ne, L, D), retain: (Nr, L, D) (같은 공간).
    Returns:
        c_dir: (L, D) unit-normalized.
    """
    L, D = explicit.shape[1], explicit.shape[2]
    delta = explicit.mean(dim=0) - retain.mean(dim=0)        # (L, D)
    Rr = retain.reshape(-1, D)                                # (Nr*L, D)
    Rr = Rr - Rr.mean(dim=0, keepdim=True)
    Sigma = (Rr.t() @ Rr) / max(Rr.shape[0] - 1, 1)          # (D, D) pooled retain cov
    Sigma = 0.5 * (Sigma + Sigma.t())
    Sinv = torch.linalg.pinv(Sigma)                          # parameter-free inverse
    cdir = delta @ Sinv                                       # (L,D)@(D,D); Sinv symmetric
    flat = cdir.reshape(-1)
    return (flat / (flat.norm() + 1e-12)).reshape(L, D)


@torch.no_grad()
def _geodesic_away(z_flat: torch.Tensor, c_unit: torch.Tensor, eta: float,
                   eps: float = 1e-6) -> torch.Tensor:
    """Rotate each row of z_flat AWAY from the concept point c_unit along the sphere geodesic.

    Manifold-preserving erasure (the sph_ot ingredient the linear redirect/overshoot lacked):
    treat each embedding as a point on the sphere of its own radius, log-map toward the concept
    point, step −eta along that tangent (away from concept), exp-map back, rescale to original norm.
    Stays ON the manifold (no off-cone collapse) while reducing concept alignment. Batched.

    Args:
        z_flat: (B, N) embeddings (flattened L*D), any radius.
        c_unit: (N,) UNIT concept point on the sphere.
        eta:    geodesic step size (fraction of the log-map magnitude; >1 rotates further away).
    Returns:
        (B, N) rotated embeddings at the original per-row norm.
    """
    znorm = z_flat.norm(dim=1, keepdim=True)                       # (B,1)
    u = z_flat / (znorm + eps)                                     # (B,N) unit
    cos = (u @ c_unit).clamp(-1.0 + eps, 1.0 - eps)               # (B,)
    theta = torch.acos(cos)                                        # (B,)
    sin = torch.sin(theta)
    log_uc = (c_unit.unsqueeze(0) - cos.unsqueeze(1) * u) * (theta / (sin + eps)).unsqueeze(1)
    step = -eta * log_uc                                           # away from concept
    sn = step.norm(dim=1, keepdim=True)                            # (B,1)
    new_u = torch.cos(sn) * u + torch.sin(sn) * (step / (sn + eps))
    out = new_u * znorm
    degenerate = (sin.unsqueeze(1) < eps) | (sn < eps)            # leave untouched if ill-defined
    return torch.where(degenerate, z_flat, out)


def cnp_loss_geodesic(
    z_current: torch.Tensor,
    z_frozen: torch.Tensor,
    concept_dir: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    """Manifold-preserving (geodesic) erasure loss — the sph_ot mechanism inside LSSE.

    Target = frozen embedding rotated AWAY from the concept direction along the sphere geodesic by
    step eta (norm preserved). MSE the current embedding toward it. Unlike linear redirect/overshoot
    (which leaves the cone and collapses generation), the geodesic stays on-manifold → erase without
    collapse. Caller may pass read-out (M^½) or raw embeddings; concept_dir lives in the same space.

    Args:
        z_current: (B, L, D) current embeddings (grad).
        z_frozen:  (B, L, D) frozen embeddings (no_grad).
        concept_dir: (L, D) unit concept direction (same space). Used as the sphere point to rotate from.
        eta: geodesic step size.
    Returns:
        scalar MSE to the geodesic-cleaned target.
    """
    B = z_current.shape[0]
    c = concept_dir.detach().reshape(-1)
    c = c / (c.norm() + 1e-12)
    zf = z_frozen.detach().reshape(B, -1)
    target = _geodesic_away(zf, c, eta)
    return F.mse_loss(z_current.reshape(B, -1), target)


def cnp_loss_geodesic_multi(
    z_current: torch.Tensor,
    z_frozen: torch.Tensor,
    concept_points: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    """Multi-direction geodesic erasure: sequentially rotate the frozen embedding AWAY from each of
    K concept points on the sphere, then MSE the current embedding toward the result.

    The concept lives in a >1-D subspace; rotating away from K axes (not just the mean) removes more
    of it per unit coherence loss, aiming to push the geodesic Pareto curve PAST sph_ot. Each step is
    on-manifold (norm preserved), so the composition stays coherent (no collapse).

    Args:
        z_current: (B, L, D) current embeddings (grad).
        z_frozen:  (B, L, D) frozen embeddings (no_grad).
        concept_points: (K, L, D) concept directions/means (same space).
        eta: per-axis geodesic step.
    Returns:
        scalar MSE to the K-times geodesic-cleaned target.
    """
    B = z_current.shape[0]
    with torch.no_grad():
        target = z_frozen.detach().reshape(B, -1)
        for k in range(concept_points.shape[0]):
            c = concept_points[k].detach().reshape(-1)
            c = c / (c.norm() + 1e-12)
            target = _geodesic_away(target, c, eta)
    return F.mse_loss(z_current.reshape(B, -1), target)


@torch.no_grad()
def compute_concept_directions_contrastive_ortho(
    explicit: torch.Tensor, retain: torch.Tensor, top_k: int
) -> torch.Tensor:
    """Top-K concept-specific directions in the given (read-out) space — multi-direction erasure.

    Single-direction read-out erasure cannot remove nudity while keeping generation coherent (the
    concept lives in a >1-D subspace; nulling one axis leaves the rest, so coherent nudity returns).
    This returns K mutually-orthonormal, retain-orthogonal concept axes:
      dir 0     = contrastive mean (mean_explicit − mean_retain), orthogonalized to retain span
                  (the dominant forget axis; retain projects low, explicit high).
      dir 1..K-1 = top right-singular vectors of the centered explicit residual AFTER removing the
                  retain span and the already-chosen dirs (extra concept-variance axes). Each unit
                  norm and Gram-Schmidt-orthogonalized against the prior dirs. Parameter-free besides K.

    Args:
        explicit: (Ne, L, D) frozen explicit read-out embeddings (already M^½-transformed by caller).
        retain:   (Nr, L, D) frozen retain read-out embeddings (same space).
        top_k:    number of concept directions to return (>=1).
    Returns:
        dirs: (K, L, D) with K = min(top_k, available); orthonormal in the flattened L*D space.
    """
    L, D = explicit.shape[1], explicit.shape[2]
    d0 = orthogonalize_direction(
        compute_concept_direction_contrastive(explicit, retain), retain)   # (L, D)
    dirs = [d0.reshape(-1)]
    if top_k > 1:
        E = explicit.reshape(explicit.shape[0], -1)
        E = E - E.mean(dim=0, keepdim=True)
        R = retain.reshape(retain.shape[0], -1)
        R = R - R.mean(dim=0, keepdim=True)
        _, S, Vt = torch.linalg.svd(R, full_matrices=False)
        if S.numel() > 0:
            B = Vt[S > (S.max() * 1e-5)]                                    # retain orthonormal basis
            if B.shape[0] > 0:
                E = E - (E @ B.t()) @ B                                     # remove retain span
        for d in dirs:                                                      # remove dir0 component
            E = E - (E @ d).unsqueeze(1) * d.unsqueeze(0)
        _, _, Vt2 = torch.linalg.svd(E, full_matrices=False)
        for i in range(min(top_k - 1, Vt2.shape[0])):
            v = Vt2[i]
            for d in dirs:                                                  # Gram-Schmidt vs prior
                v = v - (v @ d) * d
            n = v.norm()
            if n < 1e-8:
                continue
            dirs.append(v / n)
    return torch.stack([d.reshape(L, D) for d in dirs], dim=0)             # (K, L, D)
