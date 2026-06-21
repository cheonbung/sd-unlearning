"""Cross-Attention Pullback metric — CAP-CNP의 수학적 핵심.

문제(진단): LSSE의 CNP는 raw CLIP 좌표계에서 개념 에너지 ‖C·c_dir‖²를 벌점한다.
그러나 UNet은 cross-attention을 통해 C를 W_k, W_v로만 읽는다:
    Attn(Q=W_q h, K=W_k C, V=W_v C).
따라서 UNet이 실제로 보는 개념 에너지는 Σ_ℓ ‖W_k^ℓ C‖² + ‖W_v^ℓ C‖² 이고,
이는 C 한 줄(토큰) c에 대해
    Σ_ℓ cᵀ(W_k^ℓᵀW_k^ℓ + W_v^ℓᵀW_v^ℓ)c = cᵀ M c,   M = mean_ℓ(W_kᵀW_k + W_vᵀW_v) ∈ ℝ^{D×D}
로 쓰여진다. M은 동결 UNet의 상수(SD v1.4 고정)다.

해법: 임베딩을 읽기-공간으로 사상한 R = C·M^{1/2} 위에서 erasure를 수행하면,
‖R‖² = cᵀ M c 가 정확히 UNet 읽기 에너지가 된다. 즉 기존 CNP/W2 손실을 R 위에서
그대로 계산하면 "UNet이 보는 개념"만 지우게 되어 TE-only 과소결정 갭(G1)을 닫는다.
M^{1/2}는 동결 상수이므로 gradient는 텍스트 인코더로만 흐른다 — LSSE 순수성 유지.

이 모듈은 동결 UNet에서 M^{1/2}만 1회 추출(이후 UNet 폐기)하고 캐시한다.

Callers:
  - methods/lsse_trainer.py (LSSETrainer.__init__, use_cap_cnp=True 일 때)

Data formats:
  load_xattn_metric_sqrt() -> torch.Tensor (D, D) 대칭 PSD. 캐시는 torch.save .pt.
  No date fields.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _symmetric_sqrt(M: torch.Tensor) -> torch.Tensor:
    """대칭 PSD 행렬의 대칭 제곱근 M^{1/2} (eigh 기반)."""
    M = 0.5 * (M + M.t())                      # 수치 대칭화
    evals, evecs = torch.linalg.eigh(M)        # M = V diag(λ) Vᵀ, λ 오름차순
    evals = evals.clamp_min(0.0)               # PSD 보장 (수치 음수 절단)
    return (evecs * evals.sqrt().unsqueeze(0)) @ evecs.t()


def _norm_sqrt(M: torch.Tensor) -> torch.Tensor:
    """평균 대각 1로 정규화(임베딩 스케일 보존) 후 대칭 제곱근 → float32."""
    diag_mean = M.diagonal().mean().clamp_min(1e-12)
    return _symmetric_sqrt(M / diag_mean).to(torch.float32)


@torch.no_grad()
def load_xattn_metric_sqrt(
    unet_id: str = "CompVis/stable-diffusion-v1-4",
    device: Optional[torch.device] = None,
    expected_dim: int = 768,
    cache_path: Optional[str] = None,
    value_only: bool = False,
    per_layer: bool = False,
):
    """동결 SD-UNet의 cross-attn 사영에서 읽기-공간 메트릭 제곱근 M^{1/2}를 계산.

    기본: M = mean_ℓ ( W_k^ℓᵀ W_k^ℓ + W_v^ℓᵀ W_v^ℓ ), ℓ = 모든 cross-attn(attn2) 레이어.
    평균 대각이 1이 되도록 정규화(임베딩 스케일 보존 → 기존 lr/손실 크기 그대로 전이).

    Modes (파라미터형 노브 아님, 카테고리형 구조 선택):
      value_only=True  → S4: W_v 만 사용 (K=어디로/V=무엇을 중 콘텐츠 운반자 V만 타깃).
      per_layer=True   → S5: 평균하지 않고 레이어별 M_ℓ^{1/2} 리스트 반환 (손실은 레이어별 합).

    Args:
        unet_id:      diffusers UNet 식별자 (subfolder="unet").
        device:       반환 텐서 device. None이면 cpu.
        expected_dim: cross-attention context 차원 D (SD v1.4 = 768).
        cache_path:   .pt 캐시 경로(모드별로 다른 파일 권장). 존재하면 로드.
        value_only:   W_v 만 누적.
        per_layer:    레이어별 리스트 반환.
    Returns:
        per_layer=False: M_sqrt (D, D) 대칭 PSD.
        per_layer=True:  list[(D, D)] 레이어별 M_sqrt.
    """
    device = device or torch.device("cpu")
    if cache_path and Path(cache_path).exists():
        logger.info(f"[CAP-CNP] M^1/2 캐시 로드 -> {cache_path}")
        obj = torch.load(cache_path, map_location=device)
        return [t.to(device) for t in obj] if isinstance(obj, list) else obj.to(device)

    logger.info(f"[CAP-CNP] 동결 UNet에서 cross-attn 메트릭 추출 중: {unet_id} "
                f"(value_only={value_only}, per_layer={per_layer})")
    from diffusers import UNet2DConditionModel  # lazy import

    unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet")
    unet.eval()

    proj_names = ("to_v",) if value_only else ("to_k", "to_v")
    per_layer_M = []                                  # one (D,D) per cross-attn module
    for name, mod in unet.named_modules():
        if not name.endswith("attn2"):               # cross-attn only (attn1 = self-attn)
            continue
        layer_acc = torch.zeros(expected_dim, expected_dim, dtype=torch.float64)
        found = False
        for proj_name in proj_names:
            proj = getattr(mod, proj_name, None)
            if proj is None or not hasattr(proj, "weight"):
                continue
            W = proj.weight.data                      # (inner_dim, in_features)
            if W.shape[1] != expected_dim:            # in_features must be context dim
                continue
            Wf = W.to(torch.float64)
            layer_acc += Wf.t() @ Wf                  # (D, D)
            found = True
        if found:
            per_layer_M.append(layer_acc)

    if not per_layer_M:
        raise RuntimeError(
            f"[CAP-CNP] cross-attn {proj_names} (in_features={expected_dim}) 미발견 "
            f"— unet_id/expected_dim 확인 필요."
        )

    if per_layer:
        out = [_norm_sqrt(M) for M in per_layer_M]
        logger.info(f"[CAP-CNP] per-layer M^1/2: {len(out)} layers, each {tuple(out[0].shape)}")
    else:
        M = torch.stack(per_layer_M, dim=0).mean(dim=0)   # mean over cross-attn layers
        out = _norm_sqrt(M)
        logger.info(
            f"[CAP-CNP] M^1/2 준비 완료: {tuple(out.shape)} from {len(per_layer_M)} cross-attn layers, "
            f"trace(M^1/2)/D={out.diagonal().mean().item():.4f}"
        )

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(out, cache_path)
        logger.info(f"[CAP-CNP] M^1/2 캐시 저장 -> {cache_path}")

    del unet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [t.to(device) for t in out] if per_layer else out.to(device)
