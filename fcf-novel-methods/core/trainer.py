"""FCF Trainer (local independent copy).

Implements the three core algorithms from the paper:
  Algorithm 1 - Explicit Concept Forgetting        (train_explicit)
  Algorithm 2 - Projection Feature Forgetting      (train_projection_implicit, FCF-P)
  Algorithm 3 - Empirical Feature Forgetting       (train_empirical_implicit,  FCF-E)

Aligned with the official FCF repository:
  https://github.com/f-c-forgetting/FCF

Key design decisions matching the original:
  - Epoch-based training (60 epochs, full-dataset iteration for Stage 1)
  - FCF-P: Frobenius-norm flat-vector projection + (target - eta * proj) formula
  - FCF-E: experience = mean(T_ori(forget) - T_ori(noise)) per CSV row (Eq. 7)
  - Stage 2 target is computed ONCE before training, not per step
  - Implicit concepts are processed as a batch per epoch (not random-sampled)
  - Sequential two-group passes for implicit forgetting

This file is a verbatim port of fcf/trainer.py — kept local so this sub-project
has no runtime dependency on the parent thesis package.

Callers:
  - core/__init__.py (re-export)
  - methods/trainer.py (subclasses FCFTrainer as NovelFCFTrainer)

Data formats handled:
  - .pt files: torch.save(state_dict, path); torch.load(path, weights_only=True).
  - HuggingFace pretrained dirs: save_pretrained / from_pretrained.
  - No date fields.

Reference:
    "Fortified Concept Forgetting for text-to-image generative models
     by machine unlearning on CLIP"
    Fan et al., Computer Standards & Interfaces 97 (2026) 104142
"""

import copy
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from .dataset import FCFDataset

logger = logging.getLogger(__name__)


class FCFTrainer:
    """Wraps a CLIPTextModel and implements the two-stage FCF training.

    Stage 1 — Explicit Concept Forgetting   (Algorithm 1)
    Stage 2 — Implicit Concept Forgetting   (Algorithm 2 = FCF-P  or  Algorithm 3 = FCF-E)
    """

    def __init__(
        self,
        text_encoder,
        tokenizer,
        device: torch.device,
        learning_rate: float = 2.5e-5,
        eta: float = 0.25,
        mu_p: float = 0.7,
        mu_e: float = 1.0,
        max_length: int = 77,
        batch_size: int = 4,
    ):
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.eta = eta
        self.mu_p = mu_p
        self.mu_e = mu_e
        self.max_length = max_length
        self.batch_size = batch_size

        # Frozen reference encoder epsilon* — never updated.
        self.frozen_encoder = copy.deepcopy(text_encoder).to(device)
        self.frozen_encoder.requires_grad_(False)
        self.frozen_encoder.eval()

        self.criterion = nn.MSELoss()
        self._reset_optimizer()

    def _reset_optimizer(self):
        self.optimizer = Adam(self.text_encoder.parameters(), lr=self.learning_rate)

    def _tokenize(self, texts: List[str]) -> dict:
        return self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _encode(self, texts: List[str], encoder=None) -> torch.Tensor:
        tokens = self._tokenize(texts)
        if encoder is None:
            return self.text_encoder(tokens.input_ids).last_hidden_state
        with torch.no_grad():
            return encoder(tokens.input_ids).last_hidden_state

    @torch.no_grad()
    def _compute_cleaned_projection_target(
        self,
        implicit_concepts: List[str],
        concept_texts: List[str],
        eta_clean: float,
    ) -> torch.Tensor:
        """FCF-P projection target for a group of implicit concepts (paper Eq. 6)."""
        concept_feats = self._encode(concept_texts, encoder=self.frozen_encoder)
        concept_mean = concept_feats.mean(dim=0)
        concept_norm = concept_mean / concept_mean.norm()

        target_feats = self._encode(implicit_concepts, encoder=self.frozen_encoder)
        target_mean = target_feats.mean(dim=0)

        proj_length = (target_mean * concept_norm).sum()
        projection = proj_length * concept_norm

        return target_mean - eta_clean * projection

    @torch.no_grad()
    def compute_experience(self, dataset: "FCFDataset") -> torch.Tensor:
        """Compute the experience vector for FCF-E (paper Eq. 7)."""
        logger.info("Computing experience vector for FCF-E ...")
        diffs = []
        explicit_p = dataset.explicit_prompts
        noise_p = dataset.noise_prompts
        n = len(explicit_p)
        for s in range(0, n, self.batch_size):
            batch_f = explicit_p[s : s + self.batch_size]
            batch_n = noise_p[s : s + self.batch_size]
            z_f = self._encode(batch_f, encoder=self.frozen_encoder)
            z_n = self._encode(batch_n, encoder=self.frozen_encoder)
            diffs.append(z_f - z_n)

        experience = torch.cat(diffs, dim=0).mean(dim=0)
        logger.info(
            f"Experience vector: shape={tuple(experience.shape)}, "
            f"norm={experience.norm().item():.4f}"
        )
        return experience

    def train_explicit(
        self,
        dataset: "FCFDataset",
        num_epochs: int = 60,
        log_every: int = 1,
    ) -> List[Dict[str, float]]:
        """Stage 1: Explicit Concept Forgetting (Algorithm 1)."""
        logger.info(
            f"[Stage 1] Explicit forgetting — {num_epochs} epochs, "
            f"{len(dataset)} samples/epoch, batch_size={self.batch_size}"
        )
        self._reset_optimizer()
        self.text_encoder.train()

        history = []
        explicit_p = dataset.explicit_prompts
        noise_p = dataset.noise_prompts
        retain_p = dataset.retain_prompts
        n_samples = len(dataset)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        for epoch in range(1, num_epochs + 1):
            epoch_losses = {"L_maintain": 0.0, "L_forget": 0.0, "L_total": 0.0}
            pbar = tqdm(range(n_batches), total=n_batches,
                        desc=f"Epoch {epoch}/{num_epochs}", leave=False)

            for batch_idx in pbar:
                s = batch_idx * self.batch_size
                e = min(s + self.batch_size, n_samples)

                batch_f = explicit_p[s:e]
                batch_n = noise_p[s:e]
                batch_r = retain_p[s:e]

                self.optimizer.zero_grad()
                with torch.no_grad():
                    z_ori_r = self.frozen_encoder(
                        self._tokenize(batch_r).input_ids
                    ).last_hidden_state
                    z_ori_n = self.frozen_encoder(
                        self._tokenize(batch_n).input_ids
                    ).last_hidden_state

                z_tar_f = self.text_encoder(
                    self._tokenize(batch_f).input_ids
                ).last_hidden_state
                z_tar_r = self.text_encoder(
                    self._tokenize(batch_r).input_ids
                ).last_hidden_state

                L_forget = self.criterion(z_tar_f, z_ori_n)
                L_retain = self.criterion(z_tar_r, z_ori_r)
                L_total = L_retain + self.eta * L_forget

                L_total.backward()
                self.optimizer.step()

                epoch_losses["L_maintain"] += L_retain.item()
                epoch_losses["L_forget"] += L_forget.item()
                epoch_losses["L_total"] += L_total.item()
                pbar.set_postfix({"L_fgt": f"{L_forget.item():.4f}",
                                  "L_ret": f"{L_retain.item():.4f}"})

            for k in epoch_losses:
                epoch_losses[k] /= n_batches
            history.append(epoch_losses)

            if epoch % log_every == 0:
                logger.info(
                    f"  Epoch {epoch}/{num_epochs} | "
                    f"L_retain={epoch_losses['L_maintain']:.4f}  "
                    f"L_forget={epoch_losses['L_forget']:.4f}  "
                    f"L_total={epoch_losses['L_total']:.4f}"
                )

        return history

    def train_projection_implicit(
        self,
        dataset: "FCFDataset",
        num_epochs: int = 60,
        log_every: int = 1,
        retain_text: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """Stage 2: Projection Feature Forgetting (Algorithm 2, FCF-P)."""
        encoder_state = copy.deepcopy(self.text_encoder.state_dict())
        if retain_text is None:
            retain_text = (
                dataset.retain_prompts[0] if dataset.retain_prompts
                else "a person wearing clothes"
            )

        history = []
        for group_idx, group_concepts in enumerate(dataset.implicit_groups):
            logger.info(
                f"[Stage 2 FCF-P] Group {group_idx}: {group_concepts} — {num_epochs} epochs"
            )

            self.text_encoder.load_state_dict(encoder_state)
            self._reset_optimizer()
            self.text_encoder.train()

            cleaned_features = self._compute_cleaned_projection_target(
                implicit_concepts=group_concepts,
                concept_texts=dataset.explicit_concepts,
                eta_clean=self.mu_p,
            )

            pbar = tqdm(range(1, num_epochs + 1),
                        desc=f"FCF-P group {group_idx}", leave=False)

            for epoch in pbar:
                self.optimizer.zero_grad()
                with torch.no_grad():
                    z_ori_r = self.frozen_encoder(
                        self._tokenize([retain_text]).input_ids
                    ).last_hidden_state

                z_tar_f = self.text_encoder(
                    self._tokenize(group_concepts).input_ids
                ).last_hidden_state
                z_tar_r = self.text_encoder(
                    self._tokenize([retain_text]).input_ids
                ).last_hidden_state

                L_forget = self.criterion(
                    z_tar_f, cleaned_features.unsqueeze(0).expand_as(z_tar_f)
                )
                L_retain = self.criterion(z_tar_r, z_ori_r)
                L_total = L_retain + self.eta * L_forget

                L_total.backward()
                self.optimizer.step()

                history.append({
                    "group": group_idx,
                    "L_maintain": L_retain.item(),
                    "L_forget": L_forget.item(),
                    "L_total": L_total.item(),
                })

                if epoch % log_every == 0:
                    pbar.set_postfix({"L_fgt": f"{L_forget.item():.4f}",
                                      "L_ret": f"{L_retain.item():.4f}"})

            encoder_state = copy.deepcopy(self.text_encoder.state_dict())
            logger.info(f"  Group {group_idx} done.")

        return history

    def train_empirical_implicit(
        self,
        dataset: "FCFDataset",
        experience: Optional[torch.Tensor] = None,
        num_epochs: int = 60,
        log_every: int = 1,
        retain_text: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """Stage 2: Empirical Feature Forgetting (Algorithm 3, FCF-E)."""
        if experience is None:
            experience = self.compute_experience(dataset)

        encoder_state = copy.deepcopy(self.text_encoder.state_dict())
        if retain_text is None:
            retain_text = (
                dataset.retain_prompts[0] if dataset.retain_prompts
                else "a person wearing clothes"
            )

        history = []
        for group_idx, group_concepts in enumerate(dataset.implicit_groups):
            logger.info(
                f"[Stage 2 FCF-E] Group {group_idx}: {group_concepts} — {num_epochs} epochs"
            )

            self.text_encoder.load_state_dict(encoder_state)
            self._reset_optimizer()
            self.text_encoder.train()

            with torch.no_grad():
                z_group = self.frozen_encoder(
                    self._tokenize(group_concepts).input_ids
                ).last_hidden_state
                target_feats = z_group - self.mu_e * experience.unsqueeze(0)

            pbar = tqdm(range(1, num_epochs + 1),
                        desc=f"FCF-E group {group_idx}", leave=False)

            for epoch in pbar:
                self.optimizer.zero_grad()
                with torch.no_grad():
                    z_ori_r = self.frozen_encoder(
                        self._tokenize([retain_text]).input_ids
                    ).last_hidden_state

                z_tar_f = self.text_encoder(
                    self._tokenize(group_concepts).input_ids
                ).last_hidden_state
                z_tar_r = self.text_encoder(
                    self._tokenize([retain_text]).input_ids
                ).last_hidden_state

                L_forget = self.criterion(z_tar_f, target_feats.detach())
                L_retain = self.criterion(z_tar_r, z_ori_r)
                L_total = L_retain + self.eta * L_forget

                L_total.backward()
                self.optimizer.step()

                history.append({
                    "group": group_idx,
                    "L_maintain": L_retain.item(),
                    "L_forget": L_forget.item(),
                    "L_total": L_total.item(),
                })

                if epoch % log_every == 0:
                    pbar.set_postfix({"L_fgt": f"{L_forget.item():.4f}",
                                      "L_ret": f"{L_retain.item():.4f}"})

            encoder_state = copy.deepcopy(self.text_encoder.state_dict())
            logger.info(f"  Group {group_idx} done.")

        return history

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.text_encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved fine-tuned encoder -> {save_dir}")

    def save_pt(self, save_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(self.text_encoder.state_dict(), save_path)
        logger.info(f"Saved state_dict -> {save_path}")

    def load(self, load_dir: str):
        from transformers import CLIPTextModel
        self.text_encoder = CLIPTextModel.from_pretrained(load_dir).to(self.device)
        self._reset_optimizer()
        logger.info(f"Loaded encoder from {load_dir}")

    def load_pt(self, pt_path: str):
        try:
            state_dict = torch.load(pt_path, map_location=self.device, weights_only=True)
        except (TypeError, RuntimeError) as e:
            logger.warning(
                f"weights_only=True failed ({e}); falling back to legacy load."
            )
            state_dict = torch.load(pt_path, map_location=self.device)
        self.text_encoder.load_state_dict(state_dict)
        self._reset_optimizer()
        logger.info(f"Loaded state_dict from {pt_path}")
