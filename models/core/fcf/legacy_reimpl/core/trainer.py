"""
FCF Trainer: implements the three core algorithms from the paper.

  Algorithm 1 - Explicit Concept Forgetting
  Algorithm 2 - Projection Feature Forgetting  (FCF-P)
  Algorithm 3 - Empirical Feature Forgetting   (FCF-E)

Aligned with the official FCF repository:
  https://github.com/f-c-forgetting/FCF

Key design decisions matching the original:
  - Epoch-based training (60 epochs, full-dataset iteration for Stage 1)
  - FCF-P: Frobenius-norm flat-vector projection + (target - η·proj) formula
  - FCF-E: experience = mean(T_ori(forget) - T_ori(retain)) per CSV row
  - Stage 2 target is computed ONCE before training, not per step
  - Implicit concepts are processed as a batch per epoch (not random-sampled)
  - Sequential two-group passes for implicit forgetting

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
    """
    Wraps a CLIPTextModel and implements the two-stage FCF training.

    Stage 1  –  Explicit Concept Forgetting  (Algorithm 1)
    Stage 2  –  Implicit Concept Forgetting  (Algorithm 2 = FCF-P  or  Algorithm 3 = FCF-E)

    Matches the original FCF repository logic:
      - concept_forgetting_train.py  →  train_explicit()
      - features_forgetting_P.py     →  train_projection_implicit()
      - features_forgetting_E.py     →  train_empirical_implicit()

    Args:
        text_encoder:  HuggingFace CLIPTextModel (fine-tuned in place).
        tokenizer:     Matching CLIPTokenizer.
        device:        torch.device.
        learning_rate: Adam LR (paper default 2.5e-5).
        eta:           Forgetting weight η (paper default 0.25).
        mu_p:          Projection forgetting strength μ_p (paper 0.7).
        mu_e:          Empirical  forgetting strength μ_e (paper 1.0).
        max_length:    CLIP max token length (77 for SD).
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
        self.tokenizer    = tokenizer
        self.device       = device
        self.learning_rate = learning_rate
        self.eta   = eta
        self.mu_p  = mu_p
        self.mu_e  = mu_e
        self.max_length = max_length
        self.batch_size = batch_size

        # Frozen reference encoder ε* — never updated.
        # Deep-copied at init so it always holds the ORIGINAL pretrained weights,
        # even after Stage 1 modifies self.text_encoder.
        self.frozen_encoder = copy.deepcopy(text_encoder).to(device)
        self.frozen_encoder.requires_grad_(False)
        self.frozen_encoder.eval()

        self.criterion = nn.MSELoss()
        self._reset_optimizer()

    def _reset_optimizer(self):
        """Create a fresh Adam optimizer for self.text_encoder."""
        self.optimizer = Adam(self.text_encoder.parameters(), lr=self.learning_rate)

    # ------------------------------------------------------------------ #
    #  Low-level helpers                                                   #
    # ------------------------------------------------------------------ #

    def _tokenize(self, texts: List[str]) -> dict:
        return self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _encode(self, texts: List[str], encoder=None) -> torch.Tensor:
        """
        Encode texts → last_hidden_state, shape (B, L, D).
        encoder=None  → trainable self.text_encoder  (gradients tracked)
        encoder=given → frozen encoder, wrapped in torch.no_grad()
        """
        tokens = self._tokenize(texts)
        if encoder is None:
            return self.text_encoder(tokens.input_ids).last_hidden_state
        with torch.no_grad():
            return encoder(tokens.input_ids).last_hidden_state

    # ------------------------------------------------------------------ #
    #  FCF-P: Projection-based cleaned feature computation                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _compute_cleaned_projection_target(
        self,
        implicit_concepts: List[str],
        concept_texts: List[str],
        eta_clean: float,
    ) -> torch.Tensor:
        """
        Compute the FCF-P projection target for a group of implicit concepts.

        Matches features_forgetting_P.py :: TextFeatureFilter.remove_projection()

        Steps:
          1. concept_mean  = mean over concept_texts embeddings   (L, D)
          2. concept_norm  = concept_mean / ||concept_mean||_F    (Frobenius, scalar divisor)
          3. target_mean   = mean over implicit_concepts embeddings  (L, D)
          4. proj_length   = <target_mean, concept_norm>_F           (scalar flat dot-product)
          5. projection    = proj_length * concept_norm               (L, D)
          6. cleaned       = target_mean - eta_clean * projection

        Returns:
            cleaned: Tensor shape (L, D)
        """
        # Concept direction from original frozen encoder
        concept_feats = self._encode(concept_texts, encoder=self.frozen_encoder)   # (B, L, D)
        concept_mean  = concept_feats.mean(dim=0)                                  # (L, D)

        # Normalize by Frobenius norm (treats L×D tensor as flat vector)
        concept_norm = concept_mean / concept_mean.norm()                          # (L, D)

        # Target embeddings from original frozen encoder
        target_feats = self._encode(implicit_concepts, encoder=self.frozen_encoder)  # (B, L, D)
        target_mean  = target_feats.mean(dim=0)                                      # (L, D)

        # Flat dot-product projection
        proj_length = (target_mean * concept_norm).sum()                           # scalar
        projection  = proj_length * concept_norm                                   # (L, D)

        # Remove projection component (Paper Eq. 6: V'_implicit = V_implicit - μ_p · V_proj)
        cleaned = target_mean - eta_clean * projection                            # (L, D)
        return cleaned

    # ------------------------------------------------------------------ #
    #  FCF-E: Experience vector computation                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_experience(self, dataset: "FCFDataset") -> torch.Tensor:
        """
        Compute the experience vector for FCF-E.

        Paper Eq. 7:
            V_emp = mean( T_ε*(P_explicit,i) - T_ε*(P_noise,i) )

        Uses (prompt_f, prompt_n) pairs — forget prompt minus NOISE prompt —
        from the original frozen encoder. This captures the concept direction
        as the difference between concept embeddings and their noise counterparts.

        Returns:
            experience: Tensor shape (L, D)
        """
        logger.info("Computing experience vector for FCF-E ...")
        diffs = []
        explicit_p = dataset.explicit_prompts
        noise_p    = dataset.noise_prompts
        n = len(explicit_p)
        for s in range(0, n, self.batch_size):
            batch_f = explicit_p[s:s + self.batch_size]
            batch_n = noise_p[s:s + self.batch_size]
            z_f = self._encode(batch_f, encoder=self.frozen_encoder)  # (B, L, D)
            z_n = self._encode(batch_n, encoder=self.frozen_encoder)  # (B, L, D)
            diffs.append(z_f - z_n)                                    # (B, L, D)

        experience = torch.cat(diffs, dim=0).mean(dim=0)    # (L, D)
        logger.info(f"Experience vector computed from {len(diffs)} pairs, "
                    f"shape={tuple(experience.shape)}, "
                    f"norm={experience.norm().item():.4f}")
        return experience

    # ------------------------------------------------------------------ #
    #  Algorithm 1 – Explicit Concept Forgetting (epoch-based)            #
    # ------------------------------------------------------------------ #

    def train_explicit(
        self,
        dataset: "FCFDataset",
        num_epochs: int = 60,
        log_every: int = 1,
    ) -> List[Dict[str, float]]:
        """
        Stage 1: Explicit Concept Forgetting (Algorithm 1).

        Matches concept_forgetting_train.py :: train().

        For each epoch, iterates over all (prompt_f, prompt_n, prompt_r) triplets:
            L_forget  = MSE( T_ε(prompt_f),  T_ε*(prompt_n) )
            L_retain  = MSE( T_ε(prompt_r),  T_ε*(prompt_r) )
            L_total   = L_retain + η × L_forget

        Returns list of per-epoch mean-loss dicts.
        """
        logger.info(f"[Stage 1] Explicit forgetting — {num_epochs} epochs, "
                    f"{len(dataset)} samples/epoch, batch_size={self.batch_size}")
        self._reset_optimizer()
        self.text_encoder.train()

        history      = []
        explicit_p   = dataset.explicit_prompts
        noise_p      = dataset.noise_prompts
        retain_p     = dataset.retain_prompts
        n_samples    = len(dataset)
        n_batches    = (n_samples + self.batch_size - 1) // self.batch_size

        for epoch in range(1, num_epochs + 1):
            epoch_losses = {"L_maintain": 0.0, "L_forget": 0.0, "L_total": 0.0}

            pbar = tqdm(
                range(n_batches),
                total=n_batches,
                desc=f"Epoch {epoch}/{num_epochs}",
                leave=False,
            )

            for batch_idx in pbar:
                s = batch_idx * self.batch_size
                e = min(s + self.batch_size, n_samples)

                batch_f = explicit_p[s:e]
                batch_n = noise_p[s:e]
                batch_r = retain_p[s:e]

                self.optimizer.zero_grad()

                # Frozen targets (no grad)
                with torch.no_grad():
                    z_ori_r = self.frozen_encoder(
                        self._tokenize(batch_r).input_ids
                    ).last_hidden_state                        # (B, L, D)
                    z_ori_n = self.frozen_encoder(
                        self._tokenize(batch_n).input_ids
                    ).last_hidden_state                        # (B, L, D)

                # Trainable outputs
                z_tar_f = self.text_encoder(
                    self._tokenize(batch_f).input_ids
                ).last_hidden_state                            # (B, L, D)
                z_tar_r = self.text_encoder(
                    self._tokenize(batch_r).input_ids
                ).last_hidden_state                            # (B, L, D)

                L_forget  = self.criterion(z_tar_f, z_ori_n)
                L_retain  = self.criterion(z_tar_r, z_ori_r)
                L_total   = L_retain + self.eta * L_forget

                L_total.backward()
                self.optimizer.step()

                epoch_losses["L_maintain"] += L_retain.item()
                epoch_losses["L_forget"]   += L_forget.item()
                epoch_losses["L_total"]    += L_total.item()

                pbar.set_postfix({"L_fgt": f"{L_forget.item():.4f}",
                                  "L_ret": f"{L_retain.item():.4f}"})

            # Mean over all batches
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

    # ------------------------------------------------------------------ #
    #  Algorithm 2 – Projection Feature Forgetting (FCF-P)               #
    # ------------------------------------------------------------------ #

    def train_projection_implicit(
        self,
        dataset: "FCFDataset",
        num_epochs: int = 60,
        log_every: int = 1,
        retain_text: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """
        Stage 2: Projection Feature Forgetting (Algorithm 2, FCF-P).

        Matches features_forgetting_P.py :: main().

        For each implicit group in dataset.implicit_groups:
          1. Compute cleaned_features ONCE from the frozen original encoder.
          2. Load current encoder state (Stage 1 for first group; previous group's
             trained state for subsequent groups — sequential chaining).
          3. Train for num_epochs gradient steps:
               L_forget = MSE( T_ε(group_concepts), cleaned_features )
               L_retain = MSE( T_ε(retain_text),    T_ε*(retain_text) )
               L_total  = L_retain + η × L_forget
          4. Snapshot weights — next group continues from this state.

        Design notes (matches original FCF repo):
          - Sequential chaining across groups (NOT independent per-group runs)
          - Single retain anchor text in Stage 2 (vs. diverse retain prompts in Stage 1)

        Args:
            retain_text: Override the Stage-2 retain anchor. Defaults to
                         dataset.retain_prompts[0] (or fallback string).

        Returns combined per-step loss history across all groups.
        """
        # Snapshot the current encoder state — each subsequent group continues
        # from the previous group's trained state (sequential chaining).
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

            # Load the chained encoder state (Stage-1 for group 0, prev group otherwise)
            self.text_encoder.load_state_dict(encoder_state)
            self._reset_optimizer()
            self.text_encoder.train()

            # ── Compute cleaned target ONCE from frozen original encoder ──────
            cleaned_features = self._compute_cleaned_projection_target(
                implicit_concepts=group_concepts,
                concept_texts=dataset.explicit_concepts,
                eta_clean=self.mu_p,
            )                                                       # (L, D)

            pbar = tqdm(range(1, num_epochs + 1),
                        desc=f"FCF-P group {group_idx}", leave=False)

            for epoch in pbar:
                self.optimizer.zero_grad()

                with torch.no_grad():
                    z_ori_r = self.frozen_encoder(
                        self._tokenize([retain_text]).input_ids
                    ).last_hidden_state                             # (1, L, D)

                # Batch: all group concepts together
                z_tar_f = self.text_encoder(
                    self._tokenize(group_concepts).input_ids
                ).last_hidden_state                                 # (B, L, D)
                z_tar_r = self.text_encoder(
                    self._tokenize([retain_text]).input_ids
                ).last_hidden_state                                 # (1, L, D)

                # cleaned_features (L, D) broadcasts to (B, L, D)
                L_forget = self.criterion(z_tar_f, cleaned_features.unsqueeze(0).expand_as(z_tar_f))
                L_retain = self.criterion(z_tar_r, z_ori_r)
                L_total  = L_retain + self.eta * L_forget

                L_total.backward()
                self.optimizer.step()

                losses = {
                    "group": group_idx,
                    "L_maintain": L_retain.item(),
                    "L_forget":   L_forget.item(),
                    "L_total":    L_total.item(),
                }
                history.append(losses)

                if epoch % log_every == 0:
                    pbar.set_postfix({"L_fgt": f"{L_forget.item():.4f}",
                                      "L_ret": f"{L_retain.item():.4f}"})

            # Snapshot trained state for next group (sequential chaining)
            encoder_state = copy.deepcopy(self.text_encoder.state_dict())
            logger.info(f"  Group {group_idx} done.")

        return history

    # ------------------------------------------------------------------ #
    #  Algorithm 3 – Empirical Feature Forgetting (FCF-E)                 #
    # ------------------------------------------------------------------ #

    def train_empirical_implicit(
        self,
        dataset: "FCFDataset",
        experience: Optional[torch.Tensor] = None,
        num_epochs: int = 60,
        log_every: int = 1,
        retain_text: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """
        Stage 2: Empirical Feature Forgetting (Algorithm 3, FCF-E).

        Matches features_forgetting_E.py :: main().

        experience can be precomputed via compute_experience(dataset) or passed in.
        If None, it is computed automatically.

        For each implicit group:
          1. Compute target = T_ori(group_concepts) - mu_e * experience.
          2. Train for num_epochs:
               L_forget = MSE( T_ε(group), target )
               L_retain = MSE( T_ε(retain), T_ε*(retain) )
          3. Snapshot weights; next group continues from this state (sequential chaining).

        Design notes (matches original FCF repo):
          - Sequential chaining across groups (NOT independent per-group runs)
          - Single retain anchor text in Stage 2 (vs. diverse retain prompts in Stage 1)

        Args:
            retain_text: Override the Stage-2 retain anchor. Defaults to
                         dataset.retain_prompts[0] (or fallback string).
        """
        if experience is None:
            experience = self.compute_experience(dataset)

        # Snapshot current encoder state for sequential group chaining
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

            # ── Compute target ONCE: T_ori(group) - mu_e * experience ────────
            with torch.no_grad():
                z_group = self.frozen_encoder(
                    self._tokenize(group_concepts).input_ids
                ).last_hidden_state                                 # (B, L, D)
                # Broadcast experience (L, D) to (B, L, D)
                target_feats = z_group - self.mu_e * experience.unsqueeze(0)  # (B, L, D)

            pbar = tqdm(range(1, num_epochs + 1),
                        desc=f"FCF-E group {group_idx}", leave=False)

            for epoch in pbar:
                self.optimizer.zero_grad()

                with torch.no_grad():
                    z_ori_r = self.frozen_encoder(
                        self._tokenize([retain_text]).input_ids
                    ).last_hidden_state                             # (1, L, D)

                z_tar_f = self.text_encoder(
                    self._tokenize(group_concepts).input_ids
                ).last_hidden_state                                 # (B, L, D)
                z_tar_r = self.text_encoder(
                    self._tokenize([retain_text]).input_ids
                ).last_hidden_state                                 # (1, L, D)

                L_forget = self.criterion(z_tar_f, target_feats.detach())
                L_retain = self.criterion(z_tar_r, z_ori_r)
                L_total  = L_retain + self.eta * L_forget

                L_total.backward()
                self.optimizer.step()

                losses = {
                    "group": group_idx,
                    "L_maintain": L_retain.item(),
                    "L_forget":   L_forget.item(),
                    "L_total":    L_total.item(),
                }
                history.append(losses)

                if epoch % log_every == 0:
                    pbar.set_postfix({"L_fgt": f"{L_forget.item():.4f}",
                                      "L_ret": f"{L_retain.item():.4f}"})

            encoder_state = copy.deepcopy(self.text_encoder.state_dict())
            logger.info(f"  Group {group_idx} done.")

        return history

    # ------------------------------------------------------------------ #
    #  Checkpoint helpers                                                  #
    # ------------------------------------------------------------------ #

    def save(self, save_dir: str):
        """Save as HuggingFace format (compatible with generate_images.py)."""
        os.makedirs(save_dir, exist_ok=True)
        self.text_encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved fine-tuned encoder → {save_dir}")

    def save_pt(self, save_path: str):
        """
        Save state_dict as a .pt file (matches original repo format).
        Compatible with: torch.load(save_path) → model.load_state_dict(...)
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(self.text_encoder.state_dict(), save_path)
        logger.info(f"Saved state_dict → {save_path}")

    def load(self, load_dir: str):
        """Load HuggingFace format checkpoint."""
        from transformers import CLIPTextModel
        self.text_encoder = CLIPTextModel.from_pretrained(load_dir).to(self.device)
        self._reset_optimizer()
        logger.info(f"Loaded encoder from {load_dir}")

    def load_pt(self, pt_path: str):
        """Load state_dict from a .pt file (original repo format).

        Uses weights_only=True for security (avoids arbitrary code execution
        from untrusted checkpoint files). Falls back to legacy mode if the
        checkpoint requires it.
        """
        try:
            state_dict = torch.load(pt_path, map_location=self.device, weights_only=True)
        except (TypeError, RuntimeError) as e:
            # weights_only added in torch 1.13; legacy fallback for older versions
            # or checkpoints containing non-tensor objects.
            logger.warning(
                f"weights_only=True failed ({e}); falling back to legacy load. "
                f"Only do this for checkpoints you trust."
            )
            state_dict = torch.load(pt_path, map_location=self.device)
        self.text_encoder.load_state_dict(state_dict)
        self._reset_optimizer()
        logger.info(f"Loaded state_dict from {pt_path}")
