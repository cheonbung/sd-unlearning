"""diagnostics.py — LSSE 학습/평가용 진단 지표 (Phase 0 계측).

목적: "왜" 망각이 성공/실패하는지 정량화. ASR만으로는 보이지 않는
구조적 신호(개념 재라우팅, retain drift, 손실 충돌, 레이어 이동)를 측정.

모든 함수는 순수 함수(부수효과 없음). 학습 루프에서 주기적으로 호출.

지표 정의:
  projection_energy      — z의 c_dir 사영 제곱 평균 (= cnp_loss 값, forget는 ↓ 목표)
  residual_concept_var   — c_dir 제거 후 잔여 구조 분산 비율 (재라우팅 탐지, ↓ 목표)
  retain_drift           — retain 임베딩이 frozen 대비 이동한 정도 (↓ 목표=보존)
  gradient_cosine        — 두 손실 그래디언트 코사인 (음수=충돌)
  per_layer_drift        — 레이어별 hidden state 변화량 (개념 처리 이동 추적)
  separability_auc       — |사영|이 forget/retain을 가르는 AUC (c_dir 품질, ↑ 좋음)

Callers:
  - methods/lsse_trainer.py (LSSETrainer.compute_diagnostics)
  - experiments/aggregate.py (post-hoc 분석)
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F


def projection_energy(embeddings: torch.Tensor, concept_dir: torch.Tensor) -> float:
    """z를 c_dir에 사영한 스칼라의 제곱 평균.

    cnp_loss와 동일한 양이지만 grad 없이 지표로만 사용.

    Args:
        embeddings:  (B, L, D) 인코더 출력.
        concept_dir: (L, D) 개념 방향 (정규화 가정).
    Returns:
        평균 사영 에너지 (>= 0).
    """
    B = embeddings.shape[0]
    c_flat = concept_dir.detach().reshape(-1)
    z_flat = embeddings.detach().reshape(B, -1)
    proj = z_flat @ c_flat  # (B,)
    return (proj ** 2).mean().item()


def residual_concept_variance(
    forget_emb: torch.Tensor,
    concept_dir: torch.Tensor,
) -> float:
    """c_dir 성분을 제거한 뒤 남은 구조적 분산의 비율.

    망각이 표면적이면(개념이 c_dir 외 방향으로 재라우팅) 잔여 분산의
    최대 주성분이 여전히 큼. 1에 가까울수록 개념 구조가 그대로 남음.

    Args:
        forget_emb:  (N, L, D) forget 프롬프트 임베딩.
        concept_dir: (L, D) 개념 방향.
    Returns:
        top_singular(residual)^2 / total_variance  (0~1).
    """
    N = forget_emb.shape[0]
    z = forget_emb.detach().reshape(N, -1)
    z = z - z.mean(dim=0, keepdim=True)
    c = concept_dir.detach().reshape(-1)
    c = c / (c.norm() + 1e-12)
    # c_dir 성분 제거
    proj_coeff = z @ c  # (N,)
    z_res = z - proj_coeff.unsqueeze(1) * c.unsqueeze(0)
    total_var = (z ** 2).sum().item() + 1e-12
    if N < 2:
        return (z_res ** 2).sum().item() / total_var
    # 잔여의 최대 특이값 (top 구조 분산)
    try:
        s = torch.linalg.svdvals(z_res)
        top_var = (s[0] ** 2).item()
    except Exception:
        top_var = (z_res ** 2).sum().item()
    return top_var / total_var


def retain_drift(z_current: torch.Tensor, z_frozen: torch.Tensor) -> float:
    """retain 임베딩의 frozen 대비 평균 제곱 이동량 (토큰·차원 평균).

    낮을수록 retain 개념이 잘 보존됨. CSR이 의도대로 작동하는지 측정.
    """
    diff = (z_current.detach() - z_frozen.detach()) ** 2
    return diff.mean().item()


def gradient_cosine(
    params: Sequence[torch.Tensor],
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
) -> float:
    """두 손실의 그래디언트 코사인 유사도 (음수면 충돌).

    W6(손실 충돌) 진단의 핵심. forget 손실과 retain 손실이 같은
    파라미터에서 반대 방향을 당기면 음수.

    Args:
        params:  공통 trainable 파라미터 리스트.
        loss_a:  스칼라 손실 (예: L_cnp).
        loss_b:  스칼라 손실 (예: L_csr).
    Returns:
        코사인 유사도 [-1, 1]. 계산 불가 시 0.0.
    """
    plist = [p for p in params if p.requires_grad]
    if not plist:
        return 0.0
    ga = torch.autograd.grad(loss_a, plist, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, plist, retain_graph=True, allow_unused=True)
    fa_parts, fb_parts = [], []
    for a, b in zip(ga, gb):
        if a is None or b is None:
            continue
        fa_parts.append(a.flatten())
        fb_parts.append(b.flatten())
    if not fa_parts:
        return 0.0
    fa = torch.cat(fa_parts)
    fb = torch.cat(fb_parts)
    if fa.norm() < 1e-12 or fb.norm() < 1e-12:
        return 0.0
    return F.cosine_similarity(fa, fb, dim=0).item()


def per_layer_drift(
    hs_current: Sequence[torch.Tensor],
    hs_frozen: Sequence[torch.Tensor],
) -> List[float]:
    """레이어별 hidden state 평균 제곱 변화량.

    output_hidden_states=True로 얻은 (L+1) 튜플. 어느 레이어가 실제로
    움직였는지 → 개념 처리가 frozen 레이어로 이동했는지(W5) 추적.

    Returns:
        길이 (L+1) 리스트. 각 원소는 해당 레이어 drift.
    """
    drifts: List[float] = []
    for hc, hf in zip(hs_current, hs_frozen):
        drifts.append(((hc.detach() - hf.detach()) ** 2).mean().item())
    return drifts


def separability_auc(
    forget_emb: torch.Tensor,
    retain_emb: torch.Tensor,
    concept_dir: torch.Tensor,
) -> float:
    """|c_dir 사영|이 forget vs retain을 가르는 AUC.

    c_dir 품질 측정(W1). forget의 사영 크기가 retain보다 크면 c_dir이
    개념을 잘 포착한 것. 0.5=무작위, 1.0=완벽 분리.

    Args:
        forget_emb:  (Nf, L, D).
        retain_emb:  (Nr, L, D).
        concept_dir: (L, D).
    Returns:
        AUC [0, 1].
    """
    c = concept_dir.detach().reshape(-1)
    pf = (forget_emb.detach().reshape(forget_emb.shape[0], -1) @ c).abs()
    pr = (retain_emb.detach().reshape(retain_emb.shape[0], -1) @ c).abs()
    # AUC = P(pf > pr) over all pairs (Mann-Whitney U / (Nf*Nr))
    nf, nr = pf.numel(), pr.numel()
    if nf == 0 or nr == 0:
        return 0.5
    diff = pf.unsqueeze(1) - pr.unsqueeze(0)  # (Nf, Nr)
    wins = (diff > 0).float().sum() + 0.5 * (diff == 0).float().sum()
    return (wins / (nf * nr)).item()
