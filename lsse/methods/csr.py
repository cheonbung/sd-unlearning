"""N8 — Contrastive Semantic Retention (CSR).

FCF와의 차별점:
  FCF는 retain 손실로 MSE(z_r, z_ori_r) 사용 — 점 대 점 회귀.
  CSR은 SimCLR 스타일 InfoNCE를 사용 — retain 프롬프트의 현재 임베딩이
  frozen 임베딩과 가까워지도록 하되, forget 임베딩과는 멀어지도록 유도.
  이는 "retain 공간"과 "forget 공간"의 분리를 명시적으로 강제함.

Callers:
  - methods/lsse_trainer.py (LSSETrainer.train_step)
  - tests/test_lsse.py

Data formats: PyTorch tensors only, no file I/O.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def csr_loss(
    z_retain_current: torch.Tensor,
    z_retain_frozen: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """SimCLR 스타일 InfoNCE Retention 손실.

    각 retain 프롬프트의 현재 인코딩(query)이 frozen 인코딩(key)에 가까워지도록.
    같은 배치 내 다른 retain 프롬프트들이 in-batch negatives 역할.

    Args:
        z_retain_current: (B, L, D) — 현재 인코더의 retain 임베딩.
        z_retain_frozen:  (B, L, D) — frozen 인코더의 retain 임베딩.
        temperature:      InfoNCE 온도 τ. 낮을수록 harder contrast.
    Returns:
        scalar loss: 대각선(i→i) 매칭을 최대화하는 cross-entropy.
    """
    B = z_retain_current.shape[0]

    q = F.normalize(z_retain_current.mean(dim=1), dim=-1)  # (B, D)
    k = F.normalize(z_retain_frozen.mean(dim=1), dim=-1)   # (B, D)

    logits = q @ k.T / temperature  # (B, B) — 행: query, 열: key
    labels = torch.arange(B, device=q.device)

    return F.cross_entropy(logits, labels)


def csr_loss_with_negatives(
    z_retain_current: torch.Tensor,
    z_retain_frozen: torch.Tensor,
    z_forget_current: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Forget 임베딩을 명시적 negative로 포함한 확장 CSR 손실.

    retain 현재 인코딩이 forget 임베딩과도 멀어지도록 추가 pressure.
    배치 크기가 작을 때 in-batch negatives 보강에 유용.

    Args:
        z_retain_current: (B, L, D) — 현재 인코더의 retain 임베딩.
        z_retain_frozen:  (B, L, D) — frozen 인코더의 retain 임베딩.
        z_forget_current: (F, L, D) — 현재 인코더의 forget 임베딩 (negatives).
        temperature:      InfoNCE 온도 τ.
    Returns:
        scalar loss.
    """
    q = F.normalize(z_retain_current.mean(dim=1), dim=-1)    # (B, D)
    k_pos = F.normalize(z_retain_frozen.mean(dim=1), dim=-1) # (B, D)
    k_neg = F.normalize(z_forget_current.mean(dim=1), dim=-1) # (F, D)

    k_all = torch.cat([k_pos, k_neg], dim=0)                 # (B+F, D)
    logits = q @ k_all.T / temperature                        # (B, B+F)
    labels = torch.arange(q.shape[0], device=q.device)        # positives = index 0..B-1

    return F.cross_entropy(logits, labels)
