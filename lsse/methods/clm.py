"""N9 — CAP-Guided Layer Masking (CLM).

FCF와의 차별점:
  FCF는 전체 CLIP 인코더를 학습. CLM은 N1 CAP 분석 결과를 활용해
  개념 인코딩에 인과적으로 가장 중요한 top-K 레이어만 학습 가능하게 설정.
  나머지 레이어는 frozen → retain 품질 보존 + 수렴 속도 향상.

  CAP 분석 결과 (results.md): Layer 0 (0.4952) > 1 (0.4110) > 2 (0.3721).
  기본값 top_k=3은 이 결과를 반영.

Callers:
  - methods/lsse_trainer.py (LSSETrainer.__init__ — setup_layer_masking)
  - tests/test_lsse.py

Data formats:
  CAP JSON schema (read-only):
    {
      "n_layers":      12,
      "layer_scores":  [0.4952, 0.4110, 0.3721, ...],
      "max_layer_idx": 0,
      "config":        {"pool": "mean", "score_norm": "l2"}
    }
  No date fields.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Set

import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_encoder_layers(text_encoder: nn.Module) -> List[nn.Module]:
    """HuggingFace CLIPTextModel에서 transformer block 리스트 추출."""
    for path in [("text_model", "encoder", "layers"), ("encoder", "layers")]:
        obj = text_encoder
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, (list, nn.ModuleList)):
            return list(obj)
    raise AttributeError(
        "CLIPTextModel에서 encoder.layers를 찾을 수 없음. "
        ".text_model.encoder.layers 또는 .encoder.layers 경로를 확인하세요."
    )


def load_cap_scores(cap_json_path: str) -> List[float]:
    """CAP 분석 JSON에서 레이어별 인과 점수 로드.

    Args:
        cap_json_path: CAPResult.save()가 생성한 JSON 파일 경로.
    Returns:
        layer_scores: 레이어 인덱스 순서의 float 리스트.
    """
    with open(cap_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["layer_scores"]


def apply_layer_mask(
    text_encoder: nn.Module,
    cap_scores: List[float],
    top_k: int = 3,
) -> Set[int]:
    """CAP 점수 기반으로 top-K 레이어만 학습 가능하게 설정.

    나머지 레이어는 requires_grad=False로 동결.
    이 함수는 optimizer 생성 전에 호출해야 optimizer가
    올바른 파라미터 집합만 추적함.

    Args:
        text_encoder: 학습 대상 CLIPTextModel.
        cap_scores:   레이어별 인과 점수 (load_cap_scores 결과).
        top_k:        학습 가능하게 열어 둘 레이어 수.
    Returns:
        top_indices: 학습 가능한 레이어 인덱스 집합 (로깅/디버그용).
    """
    layers = _get_encoder_layers(text_encoder)
    n = len(layers)
    if top_k >= n:
        logger.warning(f"top_k={top_k} >= n_layers={n}; 전체 레이어 학습.")
        return set(range(n))

    sorted_indices = sorted(range(n), key=lambda i: -cap_scores[i])
    top_indices = set(sorted_indices[:top_k])

    for i, layer in enumerate(layers):
        layer.requires_grad_(i in top_indices)

    logger.info(
        f"[CLM] top-{top_k} 학습 레이어: {sorted(top_indices)} "
        f"(scores: {[f'{cap_scores[i]:.4f}' for i in sorted(top_indices)]})"
    )
    return top_indices


def apply_uniform_mask(
    text_encoder: nn.Module,
    top_k: int = 3,
) -> Set[int]:
    """CAP JSON 없이 초기 K개 레이어만 열어두는 fallback.

    CAP 점수 파일이 없을 때 사용. CLIP에서 초기 레이어가 개념을
    인코딩한다는 일반 경향을 반영.

    Args:
        text_encoder: 학습 대상 CLIPTextModel.
        top_k:        학습 가능하게 열어 둘 레이어 수 (0부터 top_k-1).
    Returns:
        top_indices: {0, 1, ..., top_k-1}.
    """
    layers = _get_encoder_layers(text_encoder)
    top_indices = set(range(min(top_k, len(layers))))
    for i, layer in enumerate(layers):
        layer.requires_grad_(i in top_indices)
    logger.info(f"[CLM] uniform fallback — 학습 레이어: {sorted(top_indices)}")
    return top_indices


def get_trainable_param_count(text_encoder: nn.Module) -> int:
    """학습 가능한 파라미터 수 반환 (로깅용)."""
    return sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
