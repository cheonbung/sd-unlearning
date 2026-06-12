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
    cnp_loss,
    cnp_loss_ddf,
    cnp_loss_margin,
    cnp_loss_multi,
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
        self._clm_top_k = clm_top_k          # W5 동적 재랭킹용 보존
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
    def precompute_concept_direction(self, explicit_prompts: List[str]):
        """학습 전 1회 개념 방향 계산 (N7 CNP 핵심).

        frozen 인코더 기준 explicit 임베딩의 첫 번째 주성분 추출.
        학습 루프 내 재계산 금지.
        """
        logger.info("개념 방향(c_dir) 계산 중 ...")
        all_z = []
        for i in range(0, len(explicit_prompts), self.batch_size):
            batch = explicit_prompts[i : i + self.batch_size]
            all_z.append(self._encode_frozen(batch))
        all_z_cat = torch.cat(all_z, dim=0)  # (N, L, D)
        if self.use_multi_cnp:
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
            need_forget_frozen = self.use_ddf or self.use_margin_cnp
            z_forget_frozen = (
                self._encode_frozen(batch_forget) if need_forget_frozen else None
            )

        # --- CNP explicit (forget) ---
        if self.use_multi_cnp and self.concept_dirs is not None:
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

        # --- CNP implicit ---
        L_cnp_implicit = torch.tensor(0.0, device=self.device)
        if batch_implicit:
            z_implicit = self._encode(batch_implicit)
            if self.use_multi_cnp and self.concept_dirs is not None:
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
        else:
            L_total = (
                self.alpha * L_cnp_explicit
                + self.beta * L_csr
                + self.gamma * L_cnp_implicit
            )
        L_total.backward()
        self.optimizer.step()

        return {
            "L_cnp_explicit": L_cnp_explicit.item(),
            "L_csr": L_csr.item(),
            "L_cnp_implicit": L_cnp_implicit.item(),
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
            self.precompute_concept_direction(dataset.explicit_prompts)

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
