"""LSSETrainer — Layer-Selective Semantic Erasure.

FCF와 완전히 독립적인 단일 루프 학습기.
FCFTrainer를 상속하지 않음 — 처음부터 독자적으로 설계.

통합 구성 요소:
  N7 CNP  — Concept Null-Space Projection (forget 손실)
  N8 CSR  — Contrastive Semantic Retention (retain 손실)
  N9 CLM  — CAP-Guided Layer Masking (선택적 레이어 고정)

학습 패러다임 차이 (FCF 대비):
  FCF: Stage 1 (explicit MSE-to-noise) → Stage 2 (implicit projection/empirical)
  LSSE: 단일 루프에서 explicit + implicit 동시 처리.
        noise 프롬프트 불필요. retain을 contrastive로 처리.

Callers:
  - methods/__init__.py (re-export)
  - train_lsse.py (main)
  - tests/test_lsse.py

Data formats:
  save(): HuggingFace save_pretrained 디렉토리 쓰기.
  No date fields.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from .cnp import (
    compute_concept_direction,
    compute_concept_direction_macd,
    compute_concept_direction_token_selective,
    compute_concept_directions,
    compute_concept_direction_contrastive,
    compute_concept_directions_contrastive_ortho,
    compute_concept_direction_whitened,
    orthogonalize_direction,
    cnp_loss,
    cnp_loss_ddf,
    cnp_loss_margin,
    cnp_loss_multi,
    cnp_loss_slerp,
    cnp_loss_redirect,
    cnp_loss_geodesic,
    cnp_loss_geodesic_multi,
)
from .csr import (
    csr_loss,
    csr_loss_with_negatives,
    csr_loss_tokenwise,
    csr_loss_membank,
    MemoryBank,
)
from .clm import (
    apply_layer_mask,
    apply_uniform_mask,
    load_cap_scores,
    get_trainable_param_count,
    recompute_layer_mask_dynamic,
)
from .xattn_metric import load_xattn_metric_sqrt
from . import diagnostics as diag

logger = logging.getLogger(__name__)


class LSSETrainer:
    """LSSE 단일 루프 학습기. FCFTrainer 상속 없이 독립 구현."""

    def __init__(
        self,
        text_encoder: nn.Module,
        tokenizer,
        device: torch.device,
        learning_rate: float = 2.5e-5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        temperature: float = 0.07,
        max_length: int = 77,
        batch_size: int = 8,
        cap_json_path: Optional[str] = None,
        clm_top_k: int = 3,
        use_extended_csr: bool = False,
        use_ddf: bool = False,
        use_macd: bool = False,
        use_plu: bool = False,
        plu_k1_frac: float = 1/3,
        plu_k2_frac: float = 2/3,
        use_multi_cnp: bool = False,
        num_concept_dirs: int = 3,
        use_ldlr: bool = False,
        save_every: int = 0,
        enable_diagnostics: bool = True,
        use_tokensel_dir: bool = False,
        tokensel_frac: float = 0.3,
        use_margin_cnp: bool = False,
        margin_ortho_weight: float = 0.1,
        use_tokenwise_csr: bool = False,
        use_membank: bool = False,
        membank_size: int = 512,
        use_dynamic_clm: bool = False,
        dynamic_clm_every: int = 15,
        use_adaptive_weights: bool = False,
        use_cap_cnp: bool = False,
        cap_unet_id: str = "CompVis/stable-diffusion-v1-4",
        cap_ortho_weight: float = 0.1,
        cap_cache_path: Optional[str] = None,
        cap_dir_mode: str = "svd",
        cap_metric_mode: str = "kv",
        cap_loss_mode: str = "margin",
        cap_retain_anchor: bool = False,
        cap_topk: int = 1,
        cap_redirect_strength: float = 1.0,
    ):
        """
        Args:
            text_encoder:    학습 대상 CLIPTextModel.
            tokenizer:       CLIPTokenizer.
            device:          torch.device.
            learning_rate:   Adam 학습률.
            alpha:           CNP forget 손실 가중치.
            beta:            CSR retain 손실 가중치.
            gamma:           implicit CNP 손실 가중치.
            temperature:     CSR InfoNCE 온도 τ.
            max_length:      토큰 최대 길이.
            batch_size:      미니배치 크기.
            cap_json_path:   N9 CLM용 CAP JSON 경로. None이면 uniform fallback.
            clm_top_k:       CLM에서 열어둘 레이어 수.
            use_extended_csr: forget 임베딩을 CSR negative로 추가 여부.
        """
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_extended_csr = use_extended_csr
        self.use_ddf = use_ddf
        self.use_macd = use_macd
        self.use_plu = use_plu
        self.plu_k1_frac = plu_k1_frac
        self.plu_k2_frac = plu_k2_frac
        self.use_multi_cnp = use_multi_cnp
        self.num_concept_dirs = num_concept_dirs
        self.use_ldlr = use_ldlr
        self.save_every = save_every
        self.enable_diagnostics = enable_diagnostics
        self.use_tokensel_dir = use_tokensel_dir
        self.tokensel_frac = tokensel_frac
        self.use_margin_cnp = use_margin_cnp
        self.margin_ortho_weight = margin_ortho_weight
        self.use_tokenwise_csr = use_tokenwise_csr
        self.use_membank = use_membank
        self.use_dynamic_clm = use_dynamic_clm
        self.dynamic_clm_every = dynamic_clm_every
        self.use_adaptive_weights = use_adaptive_weights
        self.use_cap_cnp = use_cap_cnp
        self.cap_ortho_weight = cap_ortho_weight
        self.cap_dir_mode = cap_dir_mode        # svd | contrastive | contrastive_ortho | whitened
        self.cap_metric_mode = cap_metric_mode  # kv | v_only | perlayer | perlayer_causal | perlayer_topk
        self.cap_loss_mode = cap_loss_mode      # margin (proj²+ortho anchor) | project (exact projection target, no overshoot)
        self.cap_retain_anchor = cap_retain_anchor  # A: pin retain embeddings in the SAME read-out metric
        self.cap_layer_weights: Optional[List[float]] = None  # B: per-layer concept-causality weights (parameter-free)
        # OOD-collapse fix (cap_loss_mode="redirect"): benign anchor coordinate(s) along c_dir,
        # precomputed from retain prompts. Scalar for single-metric, list[float] per-layer.
        self.redirect_coord: Optional[float] = None
        self.redirect_coords: Optional[List[float]] = None
        # Multi-direction (top-K) read-out erasure: per-layer (K,L,D) dirs + per-layer K anchor coords.
        # Active only when cap_topk>1 with perlayer metric (built in precompute_concept_direction).
        self.cap_topk = max(1, int(cap_topk))
        self.concept_dirs_list: Optional[List[torch.Tensor]] = None
        self.redirect_coords_list: Optional[List[List[float]]] = None
        # geodesic mode: per-layer explicit concept MEAN (read-out) = the sphere point to rotate away from.
        self.concept_mean_list: Optional[List[torch.Tensor]] = None
        # raw-space projection mechanism: cap_metric_mode="raw" sets M^1/2=I (erase in raw
        # last_hidden_state, FCF-P space); cap_redirect_strength>1 OVERSHOOTS past the benign anchor.
        self.cap_redirect_strength = float(cap_redirect_strength)
        self._clm_top_k = clm_top_k          # W5 동적 재랭킹용 보존
        # CAP-CNP: 동결 UNet cross-attn 읽기-공간 메트릭 M^{1/2} (1회 로드 후 UNet 폐기).
        # 개념 방향/erasure를 R = C·M^{1/2} 위에서 계산 → UNet이 보는 개념만 제거.
        # metric_mode: kv(기본) / v_only(W_v만) / perlayer(레이어별 리스트). 모드별 캐시 분리.
        self.M_sqrt: Optional[torch.Tensor] = None
        self.M_sqrt_list: Optional[List[torch.Tensor]] = None
        self.concept_dir_list: Optional[List[torch.Tensor]] = None
        is_perlayer = cap_metric_mode in ("perlayer", "perlayer_causal", "perlayer_topk")
        if use_cap_cnp and cap_metric_mode == "raw":
            # raw-space projection: M^1/2 = I → erase directly in raw last_hidden_state (FCF-P space),
            # not the UNet read-out space. No UNet load needed.
            D = text_encoder.config.hidden_size
            self.M_sqrt = torch.eye(D, device=device, dtype=torch.float32)
            logger.info(f"[CAP-CNP] raw-space mode: M^1/2 = I_{D} (erase in raw last_hidden_state)")
        elif use_cap_cnp:
            # perlayer 계열은 동일한 레이어별 M^{1/2} 추출을 공유(가중만 다름) → "perlayer" 캐시 통일.
            cache_p = self._mode_cache_path(
                cap_cache_path, "perlayer" if is_perlayer else cap_metric_mode
            )
            loaded = load_xattn_metric_sqrt(
                unet_id=cap_unet_id, device=device, cache_path=cache_p,
                value_only=(cap_metric_mode == "v_only"),
                per_layer=is_perlayer,
            )
            if is_perlayer:
                self.M_sqrt_list = loaded
            else:
                self.M_sqrt = loaded
        self.memory_bank = MemoryBank(membank_size) if use_membank else None
        # W6 uncertainty weighting: 손실별 log-variance (cnp, csr, implicit)
        self.log_vars = (
            nn.Parameter(torch.zeros(3, device=device)) if use_adaptive_weights else None
        )
        self.concept_dirs: Optional[torch.Tensor] = None  # (K, L, D) for multi-dir

        self.frozen_encoder = copy.deepcopy(text_encoder).to(device)
        self.frozen_encoder.requires_grad_(False)
        self.frozen_encoder.eval()

        self.concept_dir: Optional[torch.Tensor] = None
        self.active_layers: Optional[Set[int]] = None

        self._setup_layer_masking(cap_json_path, clm_top_k)
        self._reset_optimizer()

        logger.info(f"[LSSE] 학습 가능 파라미터: {get_trainable_param_count(self.text_encoder):,}")

    @staticmethod
    def _mode_cache_path(path: Optional[str], mode: str) -> Optional[str]:
        """모드별 M^{1/2} 캐시 파일 분리 (kv는 원본 경로, 그 외는 .mode 접미사)."""
        if not path or mode == "kv":
            return path
        base, ext = os.path.splitext(path)
        return f"{base}.{mode}{ext}"

    @torch.no_grad()
    def _cap_direction(self, expl_R: torch.Tensor, ret_R: Optional[torch.Tensor]) -> torch.Tensor:
        """cap_dir_mode에 따라 읽기-공간(R) 개념 방향 계산. ret_R은 contrastive/ortho/whitened용."""
        mode = self.cap_dir_mode
        if mode == "svd":
            return compute_concept_direction(expl_R)
        if ret_R is None:
            raise ValueError(f"cap_dir_mode='{mode}'는 retain 임베딩이 필요합니다.")
        if mode == "contrastive":
            return compute_concept_direction_contrastive(expl_R, ret_R)
        if mode == "contrastive_ortho":
            base = compute_concept_direction_contrastive(expl_R, ret_R)
            return orthogonalize_direction(base, ret_R)
        if mode == "whitened":
            return compute_concept_direction_whitened(expl_R, ret_R)
        raise ValueError(f"알 수 없는 cap_dir_mode: {mode}")

    def _cap_loss_one(self, z_cur_R: torch.Tensor, z_frozen_R: torch.Tensor,
                      cdir: torch.Tensor, anchor_coord: Optional[float] = None) -> torch.Tensor:
        """단일 읽기-공간 erasure 손실.

        cap_loss_mode:
          margin   — proj²→0 + 약한 직교 앵커 (W2). 경계 너머로 계속 미는 오버슈팅 가능.
          project  — 정확 사영 목표 P=I−ĉĉᵀ (cnp_loss_ddf). 직교 보완을 frozen에 고정,
                     과회전 불가 → 일반 콘텐츠 보존(D3 해결).
          slerp    — Ring-A-Bell OOD-collapse 수정 P1: 개념 제거 후 frozen 노름으로 재정규화
                     (읽기-공간 에너지 보존 → OOD에서 방향 퇴화/garbage 방지).
          redirect — OOD-collapse 수정 P2/3: 개념 축 좌표를 benign 앵커 좌표로 이동
                     (0으로 소거 대신 구체적 benign 끌개 부여). anchor_coord 필요.
        """
        if self.cap_loss_mode == "project":
            return cnp_loss_ddf(z_cur_R, z_frozen_R, cdir)
        if self.cap_loss_mode == "slerp":
            return cnp_loss_slerp(z_cur_R, z_frozen_R, cdir)
        if self.cap_loss_mode == "geodesic":   # manifold-preserving sphere rotation away from concept
            return cnp_loss_geodesic(z_cur_R, z_frozen_R, cdir, self.cap_redirect_strength)
        if self.cap_loss_mode == "redirect":
            if anchor_coord is None:
                raise ValueError("cap_loss_mode='redirect' requires a precomputed anchor_coord")
            return cnp_loss_redirect(z_cur_R, z_frozen_R, cdir, anchor_coord,
                                     self.cap_redirect_strength)
        return cnp_loss_margin(z_cur_R, z_frozen_R, cdir, self.cap_ortho_weight)

    def _cap_margin_loss(self, z_cur: torch.Tensor, z_frozen: torch.Tensor) -> torch.Tensor:
        """CAP-CNP erasure 손실 (읽기-공간).

        perlayer 계열: 레이어별 손실의 (인과)가중 합.
          - 가중 None(plain perlayer) → 균일 평균 (R2 기존 동작 보존).
          - cap_layer_weights 존재(causal/topk) → Σ_ℓ w_ℓ·loss_ℓ, Σw=1 (정규화 불필요).
            w_ℓ≈0 레이어는 건너뜀 → 비인과 레이어 왜곡 제거(D2 해결).
        그 외: 단일 M^{1/2}.
        """
        if self.M_sqrt_list is not None:
            w = self.cap_layer_weights
            terms = []
            for i, Mh in enumerate(self.M_sqrt_list):
                wi = 1.0 if w is None else float(w[i])
                if wi <= 0.0:
                    continue
                z_cR, z_fR = z_cur @ Mh, z_frozen @ Mh
                if self.concept_dirs_list is not None:      # top-K: erasure over K dirs/layer
                    dirs = self.concept_dirs_list[i]
                    if self.cap_loss_mode == "geodesic":    # sequential geodesic rotation away from K axes
                        terms.append(wi * cnp_loss_geodesic_multi(
                            z_cR, z_fR, dirs, self.cap_redirect_strength))
                    else:
                        coords = (self.redirect_coords_list[i]
                                  if self.redirect_coords_list is not None
                                  else [None] * dirs.shape[0])
                        sub = sum(self._cap_loss_one(z_cR, z_fR, dirs[j], coords[j])
                                  for j in range(dirs.shape[0]))
                        terms.append(wi * sub)
                elif self.cap_loss_mode == "geodesic":     # rotate away from per-layer concept MEAN
                    terms.append(wi * self._cap_loss_one(z_cR, z_fR, self.concept_mean_list[i]))
                else:
                    cdir = self.concept_dir_list[i]
                    coord = None if self.redirect_coords is None else self.redirect_coords[i]
                    terms.append(wi * self._cap_loss_one(z_cR, z_fR, cdir, coord))
            if not terms:
                return torch.tensor(0.0, device=z_cur.device)
            denom = float(len(terms)) if w is None else 1.0  # causal/topk weights already sum to 1
            return sum(terms) / denom
        return self._cap_loss_one(z_cur @ self.M_sqrt, z_frozen @ self.M_sqrt,
                                  self.concept_dir, self.redirect_coord)

    def _cap_retain_anchor_loss(self, z_ret_cur: torch.Tensor,
                                z_ret_frozen: torch.Tensor) -> torch.Tensor:
        """A: retain 임베딩을 erasure와 같은 읽기-공간 메트릭에 고정 (UNet이 보는 일반 콘텐츠 보존).

        L = mean_ℓ ‖(z_cur − z_frozen) @ M_ℓ^{1/2}‖²  (perlayer면 전 레이어 균일 평균).
        개념 가중(w_ℓ)을 쓰지 않음 — 일반 콘텐츠는 개념이 약한 레이어에서도 보존돼야 하므로.
        소거 연산자가 retain 부분공간에서 항등이 되도록 강제 → COCO-CLIP이 측정하는 공간을 직접 보호(D1 해결).
        """
        if self.M_sqrt_list is not None:
            terms = [((z_ret_cur @ Mh - z_ret_frozen @ Mh) ** 2).mean()
                     for Mh in self.M_sqrt_list]
            return sum(terms) / max(len(terms), 1)
        d = z_ret_cur @ self.M_sqrt - z_ret_frozen @ self.M_sqrt
        return (d ** 2).mean()

    @torch.no_grad()
    def _compute_layer_weights(self, expl_raw: torch.Tensor,
                               ret_raw: Optional[torch.Tensor]) -> Optional[List[float]]:
        """B: 레이어별 개념-인과 가중 w_ℓ = ‖(μ_explicit − μ_retain) @ M_ℓ^{1/2}‖² (파라미터-free).

        perlayer(plain) → None(균일). perlayer_causal → Σw=1 정규화 연속 가중.
        perlayer_topk → 누적 개념 에너지 90%를 담는 최소 레이어 집합만 유지(나머지 0), 그 안에서 정규화.
        근거: 최소 왜곡 하 개념-에너지 제거 — 개념이 약한 레이어 소거는 순수 손실이므로 가중 0.
        """
        if self.cap_metric_mode not in ("perlayer_causal", "perlayer_topk"):
            return None
        if ret_raw is None or self.M_sqrt_list is None:
            logger.warning("  [CAP-CNP] causal 가중에 retain 필요 — 균일 fallback")
            return None
        delta = expl_raw.mean(dim=0) - ret_raw.mean(dim=0)                 # (L, D)
        energies = torch.stack([((delta @ Mh) ** 2).sum() for Mh in self.M_sqrt_list])
        if self.cap_metric_mode == "perlayer_topk":
            total_e = energies.sum().clamp_min(1e-12)
            sorted_e, idx = torch.sort(energies, descending=True)
            cum = torch.cumsum(sorted_e, 0) / total_e
            k = int((cum < 0.9).sum().item()) + 1     # 누적 에너지 ≥90% 최소 집합
            mask = torch.zeros_like(energies)
            mask[idx[:k]] = energies[idx[:k]]
            energies = mask
        w = (energies / energies.sum().clamp_min(1e-12)).tolist()
        nz = sum(1 for x in w if x > 0)
        logger.info(f"  [CAP-CNP] layer weights ({self.cap_metric_mode}): "
                    f"{nz}/{len(w)} active, top3={sorted(w, reverse=True)[:3]}")
        return w

    @torch.no_grad()
    def _compute_redirect_coords(self, ret_raw: Optional[torch.Tensor]) -> None:
        """OOD-collapse fix P2/3: benign anchor coordinate(s) along c_dir from retain prompts.

        anchor_coord = mean over retain prompts of ⟨retain read-out, ĉ⟩. The redirect loss shifts
        each forget/implicit/OOD read-out's concept-axis coordinate to this benign value instead of
        nulling it → a concrete coherent attractor (prevents OOD direction degeneration). Sets
        self.redirect_coords (perlayer list) or self.redirect_coord (single).
        """
        if ret_raw is None:
            raise ValueError("redirect mode needs retain prompts to set the benign anchor coord")

        def _coord(R: torch.Tensor, cdir: torch.Tensor) -> float:
            a = R.reshape(R.shape[0], -1)
            cf = cdir.reshape(-1)
            cf = cf / (cf.norm() + 1e-12)
            return float((a @ cf).mean())

        if self.concept_dirs_list is not None:           # top-K multi-direction per layer
            self.redirect_coords_list = [
                [_coord(ret_raw @ Mh, dirs[j]) for j in range(dirs.shape[0])]
                for Mh, dirs in zip(self.M_sqrt_list, self.concept_dirs_list)
            ]
            logger.info(f"  [CAP-CNP] top-K redirect coords: {self.cap_topk}/layer")
            return

        if self.M_sqrt_list is not None:
            self.redirect_coords = [_coord(ret_raw @ Mh, cdir)
                                    for Mh, cdir in zip(self.M_sqrt_list, self.concept_dir_list)]
            mean_c = sum(self.redirect_coords) / max(len(self.redirect_coords), 1)
            logger.info(f"  [CAP-CNP] redirect anchor coords (perlayer): mean={mean_c:.4f}")
        else:
            self.redirect_coord = _coord(ret_raw @ self.M_sqrt, self.concept_dir)
            logger.info(f"  [CAP-CNP] redirect anchor coord={self.redirect_coord:.4f}")

    def _setup_layer_masking(self, cap_json_path: Optional[str], top_k: int):
        if cap_json_path and Path(cap_json_path).exists():
            scores = load_cap_scores(cap_json_path)
            self.active_layers = apply_layer_mask(self.text_encoder, scores, top_k)
        else:
            if cap_json_path:
                logger.warning(f"CAP JSON 없음: {cap_json_path} — uniform fallback.")
            self.active_layers = apply_uniform_mask(self.text_encoder, top_k)

    def _update_layer_mask(self, top_k: int):
        """PLU: 학습 중 레이어 마스크 동적 업데이트 후 optimizer 재생성."""
        for p in self.text_encoder.parameters():
            p.requires_grad_(True)
        self.active_layers = apply_uniform_mask(self.text_encoder, top_k)
        self._reset_optimizer()
        logger.info(f"  [PLU] top_k={top_k} → layers={sorted(self.active_layers)}, params={get_trainable_param_count(self.text_encoder):,}")

    def _reset_optimizer(self):
        if self.use_ldlr:
            from .clm import _get_encoder_layers
            try:
                layers = _get_encoder_layers(self.text_encoder)
                param_groups, seen = [], set()
                for i, layer in enumerate(layers):
                    lp = [p for p in layer.parameters() if p.requires_grad]
                    if lp:
                        lr = self.learning_rate * (0.1 if i >= 3 else 1.0)
                        param_groups.append({"params": lp, "lr": lr})
                        seen.update(id(p) for p in lp)
                other = [p for p in self.text_encoder.parameters() if p.requires_grad and id(p) not in seen]
                if other:
                    param_groups.append({"params": other, "lr": self.learning_rate * 0.1})
                self.optimizer = Adam(param_groups)
                logger.info(f"  [LDLR] layers 0-2: lr={self.learning_rate}, layers 3+: lr={self.learning_rate*0.1}")
                return
            except AttributeError:
                pass
        trainable = [p for p in self.text_encoder.parameters() if p.requires_grad]
        groups = [{"params": trainable, "lr": self.learning_rate}]
        if self.log_vars is not None:  # W6 적응가중 파라미터 포함
            groups.append({"params": [self.log_vars], "lr": self.learning_rate})
        self.optimizer = Adam(groups)

    def _tokenize(self, texts: List[str]) -> object:
        return self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _encode(self, texts: List[str]) -> torch.Tensor:
        tokens = self._tokenize(texts)
        return self.text_encoder(tokens.input_ids).last_hidden_state

    @torch.no_grad()
    def _encode_frozen(self, texts: List[str]) -> torch.Tensor:
        tokens = self._tokenize(texts)
        return self.frozen_encoder(tokens.input_ids).last_hidden_state

    @torch.no_grad()
    def precompute_concept_direction(self, explicit_prompts: List[str],
                                     retain_prompts: Optional[List[str]] = None):
        """학습 전 1회 개념 방향 계산 (N7 CNP 핵심).

        frozen 인코더 기준 explicit 임베딩의 첫 번째 주성분 추출.
        학습 루프 내 재계산 금지. retain_prompts는 CAP-CNP의 contrastive/ortho/whitened용.
        """
        logger.info("개념 방향(c_dir) 계산 중 ...")
        all_z = []
        for i in range(0, len(explicit_prompts), self.batch_size):
            batch = explicit_prompts[i : i + self.batch_size]
            all_z.append(self._encode_frozen(batch))
        all_z_cat = torch.cat(all_z, dim=0)  # (N, L, D)
        if self.use_cap_cnp:
            # 읽기-공간 R = C·M^{1/2} 에서 개념 방향 추출 (UNet이 보는 개념 축).
            ret_raw = None
            if self.cap_dir_mode != "svd" or self.cap_loss_mode == "redirect":
                if not retain_prompts:
                    raise ValueError("CAP-CNP dir_mode!=svd / redirect 에는 retain_prompts 필요")
                rz = [self._encode_frozen(retain_prompts[i:i + self.batch_size])
                      for i in range(0, len(retain_prompts), self.batch_size)]
                ret_raw = torch.cat(rz, dim=0)
            if self.M_sqrt_list is not None:
                self.concept_dir_list = [
                    self._cap_direction(all_z_cat @ Mh,
                                        (ret_raw @ Mh) if ret_raw is not None else None)
                    for Mh in self.M_sqrt_list
                ]
                self.concept_dir = self.concept_dir_list[0]
                # B: 인과 레이어 가중 (perlayer_causal/topk일 때만; 그 외 None=균일)
                self.cap_layer_weights = self._compute_layer_weights(all_z_cat, ret_raw)
                logger.info(f"  [CAP-CNP] per-layer 개념 방향 {len(self.concept_dir_list)}개 "
                            f"(dir_mode={self.cap_dir_mode}, metric={self.cap_metric_mode}, "
                            f"layer_w={'uniform' if self.cap_layer_weights is None else 'causal'})")
                if self.cap_topk > 1:                       # multi-direction (top-K) read-out erasure
                    if ret_raw is None:
                        raise ValueError("cap_topk>1 (multi-direction) requires retain prompts")
                    self.concept_dirs_list = [
                        compute_concept_directions_contrastive_ortho(
                            all_z_cat @ Mh, ret_raw @ Mh, self.cap_topk)
                        for Mh in self.M_sqrt_list
                    ]
                    logger.info(f"  [CAP-CNP] top-K multi-dir: K={self.concept_dirs_list[0].shape[0]} "
                                f"per layer x {len(self.concept_dirs_list)} layers")
                if self.cap_loss_mode == "geodesic":        # per-layer explicit MEAN as geodesic anchor
                    self.concept_mean_list = [(all_z_cat @ Mh).mean(dim=0) for Mh in self.M_sqrt_list]
                    logger.info(f"  [CAP-CNP] geodesic mode: per-layer concept means x "
                                f"{len(self.concept_mean_list)} (eta={self.cap_redirect_strength})")
            else:
                ret_R = (ret_raw @ self.M_sqrt) if ret_raw is not None else None
                self.concept_dir = self._cap_direction(all_z_cat @ self.M_sqrt, ret_R)
                logger.info(f"  [CAP-CNP] R=C·M^1/2 개념 방향 "
                            f"(dir_mode={self.cap_dir_mode}, metric_mode={self.cap_metric_mode})")
            if self.cap_loss_mode == "redirect":
                self._compute_redirect_coords(ret_raw)
        elif self.use_multi_cnp:
            self.concept_dirs = compute_concept_directions(all_z_cat, self.num_concept_dirs)
            self.concept_dir = self.concept_dirs[0].clone()
            logger.info(f"  [Multi-CNP] {self.num_concept_dirs}개 개념 방향 추출")
        elif self.use_macd:
            self.concept_dir = compute_concept_direction_macd(all_z_cat)
            logger.info("  [MACD] 구면 PCA 기반 개념 방향 사용")
        elif self.use_tokensel_dir:
            self.concept_dir = compute_concept_direction_token_selective(
                all_z_cat, self.tokensel_frac
            )
            logger.info(f"  [W1] 토큰 선택 SVD 개념 방향 (top_frac={self.tokensel_frac})")
        else:
            self.concept_dir = compute_concept_direction(all_z_cat)
        logger.info(
            f"  c_dir shape={tuple(self.concept_dir.shape)}, "
            f"norm={self.concept_dir.norm().item():.4f}"
        )

    @torch.no_grad()
    def precompute_multiconcept_directions(self, concept_groups: "Dict[str, List[str]]", weights=None):
        """Multi-CONCEPT erasure: one CNP direction per concept (nudity/violence/style/...).

        concept_groups: {name -> explicit prompts}. Each group's frozen-encoder embeddings yield
        one SVD concept direction; stacked into self.concept_dirs (K, L, D) and consumed by
        cnp_loss_multi (use_multi_cnp forced True). This differs from the existing use_multi_cnp,
        which extracts top-K directions of a SINGLE concept. Sets self.concept_dir so train() skips
        its own single-concept precompute.
        """
        dirs = []
        for name, prompts in concept_groups.items():
            zs = [self._encode_frozen(prompts[i:i + self.batch_size])
                  for i in range(0, len(prompts), self.batch_size)]
            cdir = compute_concept_direction(torch.cat(zs, dim=0))
            dirs.append(cdir)
            logger.info(f"  [Multi-CONCEPT] '{name}': c_dir norm={cdir.norm().item():.4f} "
                        f"(n={len(prompts)})")
        self.concept_dirs = torch.stack(dirs, dim=0)          # (K, L, D)
        self.concept_dir = self.concept_dirs[0].clone()
        self.concept_weights = weights
        self.use_multi_cnp = True
        logger.info(f"  [Multi-CONCEPT] {len(dirs)} dirs ready -> cnp_loss_multi (weights={weights})")

    def _train_step(
        self,
        batch_forget: List[str],
        batch_retain: List[str],
        batch_implicit: Optional[List[str]],
    ) -> Dict[str, float]:
        self.optimizer.zero_grad()

        z_forget = self._encode(batch_forget)
        z_retain = self._encode(batch_retain)
        with torch.no_grad():
            z_retain_frozen = self._encode_frozen(batch_retain)
            need_forget_frozen = self.use_ddf or self.use_margin_cnp or self.use_cap_cnp
            z_forget_frozen = (
                self._encode_frozen(batch_forget) if need_forget_frozen else None
            )

        # --- CNP explicit (forget) ---
        if self.use_cap_cnp:                                            # CAP-CNP
            # 읽기-공간 R = C·M^{1/2} 위에서 W2 margin erasure (UNet이 보는 개념만 제거).
            L_cnp_explicit = self._cap_margin_loss(z_forget, z_forget_frozen)
        elif self.use_multi_cnp and self.concept_dirs is not None:
            L_cnp_explicit = cnp_loss_multi(z_forget, self.concept_dirs,
                                            getattr(self, "concept_weights", None))
        elif self.use_margin_cnp:                                       # W2
            L_cnp_explicit = cnp_loss_margin(
                z_forget, z_forget_frozen, self.concept_dir, self.margin_ortho_weight
            )
        elif self.use_ddf:
            L_cnp_explicit = cnp_loss_ddf(z_forget, z_forget_frozen, self.concept_dir)
        else:
            L_cnp_explicit = cnp_loss(z_forget, self.concept_dir)

        # --- CSR (retain) ---
        if self.use_membank and self.memory_bank is not None:           # W4
            L_csr = csr_loss_membank(
                z_retain, z_retain_frozen, self.memory_bank, self.temperature
            )
        elif self.use_tokenwise_csr:                                    # W3
            L_csr = csr_loss_tokenwise(z_retain, z_retain_frozen, self.temperature)
        elif self.use_extended_csr:
            L_csr = csr_loss_with_negatives(z_retain, z_retain_frozen, z_forget, self.temperature)
        else:
            L_csr = csr_loss(z_retain, z_retain_frozen, self.temperature)

        # --- A: 읽기-공간 retain 앵커 (CAP-CNP 전용; 소거가 일어나는 바로 그 공간에서 retain 고정) ---
        L_retain_anchor = torch.tensor(0.0, device=self.device)
        if self.use_cap_cnp and self.cap_retain_anchor:
            L_retain_anchor = self._cap_retain_anchor_loss(z_retain, z_retain_frozen)

        # --- CNP implicit ---
        L_cnp_implicit = torch.tensor(0.0, device=self.device)
        if batch_implicit:
            z_implicit = self._encode(batch_implicit)
            if self.use_cap_cnp:                                        # CAP-CNP
                with torch.no_grad():
                    z_implicit_frozen = self._encode_frozen(batch_implicit)
                L_cnp_implicit = self._cap_margin_loss(z_implicit, z_implicit_frozen)
            elif self.use_multi_cnp and self.concept_dirs is not None:
                L_cnp_implicit = cnp_loss_multi(z_implicit, self.concept_dirs,
                                                getattr(self, "concept_weights", None))
            elif self.use_margin_cnp:                                   # W2
                with torch.no_grad():
                    z_implicit_frozen = self._encode_frozen(batch_implicit)
                L_cnp_implicit = cnp_loss_margin(
                    z_implicit, z_implicit_frozen, self.concept_dir, self.margin_ortho_weight
                )
            elif self.use_ddf:
                with torch.no_grad():
                    z_implicit_frozen = self._encode_frozen(batch_implicit)
                L_cnp_implicit = cnp_loss_ddf(z_implicit, z_implicit_frozen, self.concept_dir)
            else:
                L_cnp_implicit = cnp_loss(z_implicit, self.concept_dir)

        # --- 총 손실 ---
        if self.use_adaptive_weights and self.log_vars is not None:     # W6
            # Kendall uncertainty weighting: Σ exp(-s_i)·L_i + s_i
            losses = [L_cnp_explicit, L_csr, L_cnp_implicit]
            L_total = sum(
                torch.exp(-self.log_vars[i]) * losses[i] + self.log_vars[i]
                for i in range(3)
            )
            # retain 앵커는 균형된 가중 없이 단위 추가(소거와 동급 보존 제약)
            L_total = L_total + L_retain_anchor
        else:
            L_total = (
                self.alpha * L_cnp_explicit
                + self.beta * L_csr
                + self.gamma * L_cnp_implicit
                + self.beta * L_retain_anchor   # CSR와 동일 retain 가중 재사용 → 새 하이퍼파라미터 없음
            )
        L_total.backward()
        self.optimizer.step()

        return {
            "L_cnp_explicit": L_cnp_explicit.item(),
            "L_csr": L_csr.item(),
            "L_cnp_implicit": L_cnp_implicit.item(),
            "L_retain_anchor": L_retain_anchor.item(),
            "L_total": L_total.item(),
        }

    def train(
        self,
        dataset,
        num_epochs: int = 60,
        log_every: int = 10,
        ckpt_dir: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """단일 루프 LSSE 학습.

        Args:
            dataset:    LSSEDataset 인스턴스.
            num_epochs: 총 epoch 수.
            log_every:  로그 출력 epoch 간격.
        Returns:
            history: epoch별 평균 손실 딕셔너리 리스트.
        """
        if self.concept_dir is None:
            self.precompute_concept_direction(dataset.explicit_prompts, dataset.retain_prompts)

        self.text_encoder.train()
        explicit_p = dataset.explicit_prompts
        retain_p = dataset.retain_prompts
        implicit_c = dataset.implicit_concepts
        n = len(explicit_p)
        n_batches = max(1, (n + self.batch_size - 1) // self.batch_size)

        # PLU 초기 마스크: k=1부터 시작
        if self.use_plu:
            self._update_layer_mask(1)

        history = []
        for epoch in range(1, num_epochs + 1):
            # PLU: epoch 진입 시 레이어 수 점진 증가 (1/3, 2/3 지점)
            if self.use_plu:
                unlock_mid = int(num_epochs * self.plu_k1_frac) + 1
                unlock_full = int(num_epochs * self.plu_k2_frac) + 1
                if epoch == unlock_mid:
                    self._update_layer_mask(3)
                elif epoch == unlock_full:
                    self._update_layer_mask(6)

            # W5: 동적 CLM — drift 기반 top-K 재랭킹 후 optimizer 재생성
            if self.use_dynamic_clm and epoch > 1 and (epoch - 1) % self.dynamic_clm_every == 0:
                try:
                    fb = explicit_p[: self.batch_size]
                    for p in self.text_encoder.parameters():
                        p.requires_grad_(True)
                    _, hsc = self._encode_hidden(fb)
                    _, hsf = self._encode_frozen_hidden(fb)
                    drifts = diag.per_layer_drift(hsc, hsf)
                    self.active_layers = recompute_layer_mask_dynamic(
                        self.text_encoder, drifts, self._clm_top_k
                    )
                    self._reset_optimizer()
                except Exception as exc:
                    logger.warning(f"  [W5] 동적 CLM 재랭킹 실패: {exc}")

            epoch_losses: Dict[str, float] = {
                "L_cnp_explicit": 0.0,
                "L_csr": 0.0,
                "L_cnp_implicit": 0.0,
                "L_retain_anchor": 0.0,
                "L_total": 0.0,
            }

            pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}/{num_epochs}", leave=False)
            for batch_idx in pbar:
                s = batch_idx * self.batch_size
                e = min(s + self.batch_size, n)
                batch_forget = explicit_p[s:e]

                r_start = s % len(retain_p)
                r_end = r_start + (e - s)
                if r_end <= len(retain_p):
                    batch_retain = retain_p[r_start:r_end]
                else:
                    batch_retain = (retain_p * 2)[r_start:r_end]

                step = self._train_step(
                    batch_forget=batch_forget,
                    batch_retain=batch_retain,
                    batch_implicit=implicit_c if implicit_c else None,
                )
                for k in epoch_losses:
                    epoch_losses[k] += step[k]
                pbar.set_postfix({
                    "L_cnp": f"{step['L_cnp_explicit']:.4f}",
                    "L_csr": f"{step['L_csr']:.4f}",
                })

            for k in epoch_losses:
                epoch_losses[k] /= n_batches

            # 진단 지표 (log 시점마다, 고정 배치 기준)
            if self.enable_diagnostics and (epoch % log_every == 0 or epoch == num_epochs):
                epoch_losses.update(self.compute_diagnostics(dataset))

            epoch_losses["epoch"] = epoch
            history.append(epoch_losses)

            if epoch % log_every == 0:
                logger.info(
                    f"  Epoch {epoch}/{num_epochs} | "
                    f"L_cnp={epoch_losses['L_cnp_explicit']:.4f}  "
                    f"L_csr={epoch_losses['L_csr']:.4f}  "
                    f"L_impl={epoch_losses['L_cnp_implicit']:.4f}  "
                    f"L_total={epoch_losses['L_total']:.4f}"
                    + (f"  resid={epoch_losses.get('diag_residual_var', float('nan')):.3f}"
                       f"  gcos={epoch_losses.get('diag_grad_cosine', float('nan')):.3f}"
                       if self.enable_diagnostics else "")
                )

            # 체크포인트 궤적 (W6/PLU 붕괴 규명용)
            if ckpt_dir and self.save_every > 0 and epoch % self.save_every == 0 and epoch < num_epochs:
                ep_dir = os.path.join(ckpt_dir, f"epoch_{epoch}")
                self.text_encoder.eval()
                self.save(ep_dir)
                self.text_encoder.train()
                logger.info(f"  [ckpt] epoch {epoch} -> {ep_dir}")

        return history

    @torch.no_grad()
    def _encode_hidden(self, texts: List[str]):
        """진단용: last_hidden_state + 전체 레이어 hidden_states 반환."""
        tokens = self._tokenize(texts)
        out = self.text_encoder(tokens.input_ids, output_hidden_states=True)
        return out.last_hidden_state, out.hidden_states

    @torch.no_grad()
    def _encode_frozen_hidden(self, texts: List[str]):
        tokens = self._tokenize(texts)
        out = self.frozen_encoder(tokens.input_ids, output_hidden_states=True)
        return out.last_hidden_state, out.hidden_states

    def compute_diagnostics(self, dataset, max_n: int = 64) -> Dict[str, float]:
        """학습 상태 진단 지표 묶음 (diag_* 접두사).

        no_grad 지표(사영/잔여분산/drift/AUC/레이어 drift) + grad 지표(손실 코사인).
        고정 부분집합 사용 — 학습 데이터 순서 영향 최소화.
        """
        if self.concept_dir is None:
            return {}
        was_training = self.text_encoder.training
        fp = dataset.explicit_prompts[:max_n]
        rp = dataset.retain_prompts[: self.batch_size]

        out: Dict[str, float] = {}
        # --- no_grad 지표 ---
        with torch.no_grad():
            z_forget = torch.cat(
                [self._encode(fp[i:i + self.batch_size]) for i in range(0, len(fp), self.batch_size)],
                dim=0,
            )
            z_retain = self._encode(rp)
            z_retain_frozen = self._encode_frozen(rp)
            out["diag_proj_energy"] = diag.projection_energy(z_forget, self.concept_dir)
            out["diag_residual_var"] = diag.residual_concept_variance(z_forget, self.concept_dir)
            out["diag_retain_drift"] = diag.retain_drift(z_retain, z_retain_frozen)
            out["diag_separability_auc"] = diag.separability_auc(
                z_forget[: self.batch_size], z_retain, self.concept_dir
            )
            # 레이어별 drift (forget 한 배치) — output_hidden_states 미지원 모델은 skip
            try:
                _, hsc = self._encode_hidden(fp[: self.batch_size])
                _, hsf = self._encode_frozen_hidden(fp[: self.batch_size])
                drifts = diag.per_layer_drift(hsc, hsf)
                out["diag_layer_drift_max"] = max(drifts) if drifts else 0.0
                out["diag_layer_drift_argmax"] = (
                    float(int(torch.tensor(drifts).argmax())) if drifts else -1.0
                )
            except TypeError:
                pass  # mock/단순 모델: hidden_states 미지원

        # --- grad 지표: L_cnp vs L_csr 충돌 ---
        try:
            self.optimizer.zero_grad()
            zf = self._encode(fp[: self.batch_size])
            zr = self._encode(rp)
            with torch.no_grad():
                zrf = self._encode_frozen(rp)
            l_cnp = cnp_loss(zf, self.concept_dir)
            l_csr = csr_loss(zr, zrf, self.temperature)
            params = [p for p in self.text_encoder.parameters() if p.requires_grad]
            out["diag_grad_cosine"] = diag.gradient_cosine(params, l_cnp, l_csr)
            self.optimizer.zero_grad()
        except Exception as exc:  # 진단 실패가 학습을 막지 않도록
            logger.warning(f"  [diag] grad_cosine 계산 실패: {exc}")
            out["diag_grad_cosine"] = float("nan")

        if was_training:
            self.text_encoder.train()
        return out

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.text_encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"인코더 저장 완료 -> {save_dir}")

    def save_pt(self, save_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(self.text_encoder.state_dict(), save_path)
        logger.info(f"state_dict 저장 완료 -> {save_path}")
