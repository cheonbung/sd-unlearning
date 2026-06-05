"""DACETrainer -- Dynamic Adversarial Concept Erasure (concept-axis, corrected).

Independent single loop (no FCFTrainer/LSSETrainer inheritance). Corrected after the
P0/P0b gates: the erased axis is the CONCEPT axis (explicit vs concept-stripped neutral),
NOT forget-vs-retain (P0 showed that is lexically saturated and anti-correlated w/ ASR).

  min_theta  alpha*L_forget + gamma*L_ortho + beta*L_retain
  U = top-k SVD of live concept-shift vectors d_i = pool(z_explicit_i) - pool(z_neutral_i),
      recomputed every `adv_every` steps -> pursues the concept as it reroutes.

use_plu: progressive layer unlocking (k=1 -> 3 -> 6 over training). The earlier ablation
showed PLU (layer dynamics) -- not the loss -- is LSSE's dominant lever; DACE+PLU tests
whether the concept-axis loss adds anything ON TOP of PLU.

save(): HF save_pretrained (eval-compatible with lsse/evaluate.py --encoder_dir).
"""
from __future__ import annotations

import copy
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from methods.adversary import concept_subspace, pool, SubspaceTracker
from methods.erasure import dace_concept_losses
from methods import diagnostics as diag
from core.dataset import neutralize

logger = logging.getLogger(__name__)


def _count_trainable(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class DACETrainer:
    def __init__(
        self, text_encoder, tokenizer, device,
        learning_rate: float = 2.5e-5, alpha: float = 1.0, gamma: float = 0.5,
        beta: float = 1.0, subspace_k: int = 4, adv_every: int = 10,
        adv_sample: int = 64, ema_decay: float = 0.0, pool_mode: str = "mean",
        max_length: int = 77, batch_size: int = 8, enable_diagnostics: bool = True,
        adv_ridge: float = 1e-2, use_plu: bool = False,
        plu_k1_frac: float = 1 / 3, plu_k2_frac: float = 2 / 3,
    ):
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.alpha, self.gamma, self.beta = alpha, gamma, beta
        self.subspace_k = subspace_k
        self.adv_every = max(1, adv_every)
        self.adv_sample = adv_sample
        self.pool_mode = pool_mode
        self.max_length = max_length
        self.batch_size = batch_size
        self.enable_diagnostics = enable_diagnostics
        self.adv_ridge = adv_ridge
        self.use_plu = use_plu
        self.plu_k1_frac, self.plu_k2_frac = plu_k1_frac, plu_k2_frac

        self.frozen_encoder = copy.deepcopy(text_encoder).to(device)
        self.frozen_encoder.requires_grad_(False)
        self.frozen_encoder.eval()

        self.tracker = SubspaceTracker(ema_decay)
        self.U: Optional[torch.Tensor] = None
        self._reset_optimizer()

    def _reset_optimizer(self):
        self.optimizer = Adam(
            [p for p in self.text_encoder.parameters() if p.requires_grad], lr=self.learning_rate)

    def _encoder_layers(self):
        enc = getattr(self.text_encoder, "text_model", self.text_encoder)
        return enc.encoder.layers

    def _set_trainable_layers(self, k: int):
        """PLU: train only the first k transformer layers (CAP: front layers most causal)."""
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for i, layer in enumerate(self._encoder_layers()):
            if i < k:
                for p in layer.parameters():
                    p.requires_grad_(True)
        self._reset_optimizer()
        logger.info(f"  [PLU] trainable first {k} layers, params={_count_trainable(self.text_encoder):,}")

    def _tokenize(self, texts):
        return self.tokenizer(texts, padding="max_length", max_length=self.max_length,
                              truncation=True, return_tensors="pt").to(self.device)

    def _encode(self, texts):
        return self.text_encoder(self._tokenize(texts).input_ids).last_hidden_state

    @torch.no_grad()
    def _encode_frozen(self, texts):
        return self.frozen_encoder(self._tokenize(texts).input_ids).last_hidden_state

    @torch.no_grad()
    def _pooled(self, prompts, frozen=False):
        enc = self._encode_frozen if frozen else self._encode
        out = []
        for i in range(0, len(prompts), self.batch_size):
            out.append(pool(enc(prompts[i:i + self.batch_size]), self.pool_mode))
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def refresh_subspace(self, explicit_sample, neutral_sample):
        pe = self._pooled(explicit_sample)
        pn = self._pooled(neutral_sample)
        d = (pe - pn).float()
        U = concept_subspace(d, self.subspace_k)
        self.U = self.tracker.update(U).to(self.text_encoder.dtype)

    def _train_step(self, b_exp, b_neu, b_ret) -> Dict[str, float]:
        self.optimizer.zero_grad()
        ze = self._encode(b_exp); zn = self._encode(b_neu); zr = self._encode(b_ret)
        with torch.no_grad():
            ze_fz = self._encode_frozen(b_exp)
            zn_fz = self._encode_frozen(b_neu)
            zr_fz = self._encode_frozen(b_ret)
        Lf, Lo, Lr = dace_concept_losses(ze, ze_fz, zn, zn_fz, zr, zr_fz, self.U,
                                         pool_mode=self.pool_mode)
        L = self.alpha * Lf + self.gamma * Lo + self.beta * Lr
        L.backward(); self.optimizer.step()
        return {"L_forget": Lf.item(), "L_ortho": Lo.item(),
                "L_retain": Lr.item(), "L_total": L.item()}

    def train(self, dataset, num_epochs: int = 30, log_every: int = 5,
              ckpt_dir: Optional[str] = None, save_every: int = 0):
        exp = dataset.forget_prompts
        neu = [neutralize(p) for p in exp]
        ret = dataset.retain_prompts
        n = len(exp)
        nb = max(1, (n + self.batch_size - 1) // self.batch_size)
        es, ns = exp[: self.adv_sample], neu[: self.adv_sample]

        self.text_encoder.train()
        if self.use_plu:
            self._set_trainable_layers(1)
        self.refresh_subspace(es, ns)

        history, gstep = [], 0
        for epoch in range(1, num_epochs + 1):
            if self.use_plu:
                if epoch == int(num_epochs * self.plu_k1_frac) + 1:
                    self._set_trainable_layers(3)
                elif epoch == int(num_epochs * self.plu_k2_frac) + 1:
                    self._set_trainable_layers(6)
            ep = {"L_forget": 0.0, "L_ortho": 0.0, "L_retain": 0.0, "L_total": 0.0}
            pbar = tqdm(range(nb), desc=f"Epoch {epoch}/{num_epochs}", leave=False)
            for bi in pbar:
                s = bi * self.batch_size
                e = min(s + self.batch_size, n)
                be, bn = exp[s:e], neu[s:e]
                r0 = s % len(ret)
                br = (ret * 2)[r0:r0 + (e - s)]
                if gstep % self.adv_every == 0:
                    self.refresh_subspace(es, ns)
                step = self._train_step(be, bn, br)
                gstep += 1
                for k in ep:
                    ep[k] += step[k]
                pbar.set_postfix({"Lf": f"{step['L_forget']:.3f}", "Lr": f"{step['L_retain']:.3f}"})
            for k in ep:
                ep[k] /= nb
            if self.enable_diagnostics and (epoch % log_every == 0 or epoch == num_epochs):
                ep.update(self.compute_diagnostics(dataset))
            ep["epoch"] = epoch
            history.append(ep)
            if epoch % log_every == 0:
                logger.info(
                    f"  Epoch {epoch}/{num_epochs} | Lf={ep['L_forget']:.4f} "
                    f"Lo={ep['L_ortho']:.4f} Lr={ep['L_retain']:.4f} "
                    f"| shift={ep.get('diag_concept_shift', float('nan')):.3f}")
            if ckpt_dir and save_every > 0 and epoch % save_every == 0 and epoch < num_epochs:
                d = os.path.join(ckpt_dir, f"epoch_{epoch}")
                self.text_encoder.eval(); self.save(d); self.text_encoder.train()
        return history

    @torch.no_grad()
    def compute_diagnostics(self, dataset, max_n: int = 64) -> Dict[str, float]:
        was = self.text_encoder.training
        self.text_encoder.eval()
        exp = dataset.forget_prompts[:max_n]
        neu = [neutralize(p) for p in exp]
        ret = dataset.retain_prompts[:max_n]
        pe = self._pooled(exp); pn = self._pooled(neu)
        zr = self._pooled(ret); zr_fz = self._pooled(ret, frozen=True)
        out = {
            "diag_concept_shift": (pe - pn).mean(dim=0).norm().item(),
            "diag_sep_auc": diag.linear_separability_auc(pe, zr, self.adv_ridge),
            "diag_retain_drift": diag.retain_drift(zr, zr_fz),
        }
        if was:
            self.text_encoder.train()
        return out

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.text_encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"encoder saved -> {save_dir}")
