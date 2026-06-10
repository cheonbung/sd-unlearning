"""LSSE 핵심 컴포넌트 단위 테스트.

테스트 항목:
  - N7 CNP: compute_concept_direction, cnp_loss
  - N8 CSR: csr_loss, csr_loss_with_negatives
  - N9 CLM: apply_uniform_mask, get_trainable_param_count
  - LSSEDataset: 직접 생성 / 예외 확인
  - LSSETrainer: init, precompute_concept_direction, 1-step train
"""

from __future__ import annotations

import pytest
import torch

from methods.cnp import compute_concept_direction, cnp_loss
from methods.csr import csr_loss, csr_loss_with_negatives
from methods.clm import apply_uniform_mask, get_trainable_param_count
from core.dataset import LSSEDataset
from methods.lsse_trainer import LSSETrainer


# --------------------------------------------------------------------------- #
#  N7 CNP                                                                      #
# --------------------------------------------------------------------------- #

class TestCNP:
    def test_concept_direction_shape(self, sample_embeddings):
        c_dir = compute_concept_direction(sample_embeddings)
        assert c_dir.shape == sample_embeddings.shape[1:]

    def test_concept_direction_unit_norm(self, sample_embeddings):
        c_dir = compute_concept_direction(sample_embeddings)
        assert abs(c_dir.norm().item() - 1.0) < 1e-5

    def test_concept_direction_requires_n_ge_2(self, device):
        with pytest.raises(ValueError, match="N >= 2"):
            compute_concept_direction(torch.randn(1, 4, 16, device=device))

    def test_cnp_loss_nonnegative(self, sample_embeddings, sample_batch, device):
        c_dir = compute_concept_direction(sample_embeddings)
        loss = cnp_loss(sample_batch, c_dir)
        assert loss.item() >= 0.0

    def test_cnp_loss_decreases_with_gradient(self, sample_embeddings, device):
        c_dir = compute_concept_direction(sample_embeddings)
        z = torch.randn(3, 4, 16, device=device, requires_grad=True)
        opt = torch.optim.SGD([z], lr=0.1)
        loss_before = cnp_loss(z, c_dir).item()
        for _ in range(10):
            opt.zero_grad()
            cnp_loss(z, c_dir).backward()
            opt.step()
        assert cnp_loss(z.detach(), c_dir).item() < loss_before


# --------------------------------------------------------------------------- #
#  N8 CSR                                                                      #
# --------------------------------------------------------------------------- #

class TestCSR:
    def test_csr_loss_nonnegative(self, sample_batch, device):
        z_f = torch.randn_like(sample_batch)
        loss = csr_loss(sample_batch, z_f)
        assert loss.item() >= 0.0

    def test_csr_loss_identity_lower_than_random(self, sample_batch, device):
        z_rand = torch.randn_like(sample_batch)
        assert csr_loss(sample_batch, sample_batch).item() < csr_loss(sample_batch, z_rand).item()

    def test_csr_loss_with_negatives_scalar(self, sample_batch, device):
        z_f = torch.randn(2, 4, 16, device=device)
        loss = csr_loss_with_negatives(sample_batch, sample_batch.clone(), z_f)
        assert loss.dim() == 0

    def test_csr_loss_finite_batch1(self, device):
        z = torch.randn(1, 4, 16, device=device)
        assert torch.isfinite(csr_loss(z, z))


# --------------------------------------------------------------------------- #
#  N9 CLM                                                                      #
# --------------------------------------------------------------------------- #

class TestCLM:
    def test_uniform_mask_freezes_correct_layers(self, mock_encoder):
        apply_uniform_mask(mock_encoder, top_k=2)
        layers = list(mock_encoder.text_model.encoder.layers)
        assert all(p.requires_grad for p in layers[0].parameters())
        assert all(p.requires_grad for p in layers[1].parameters())
        assert not any(p.requires_grad for p in layers[2].parameters())
        assert not any(p.requires_grad for p in layers[3].parameters())

    def test_trainable_param_count_reduced(self, mock_encoder):
        total = sum(p.numel() for p in mock_encoder.parameters())
        apply_uniform_mask(mock_encoder, top_k=2)
        assert get_trainable_param_count(mock_encoder) < total

    def test_top_k_exceeds_layers_opens_all(self, mock_encoder):
        result = apply_uniform_mask(mock_encoder, top_k=100)
        assert len(result) == 4


# --------------------------------------------------------------------------- #
#  LSSEDataset                                                                 #
# --------------------------------------------------------------------------- #

class TestLSSEDataset:
    def test_construction_and_len(self):
        ds = LSSEDataset(
            explicit_prompts=["a", "b"],
            retain_prompts=["c"],
            implicit_concepts=["x", "y"],
            target_concept="test",
        )
        assert len(ds) == 2

    def test_empty_explicit_raises(self):
        with pytest.raises(ValueError, match="explicit_prompts"):
            LSSEDataset([], ["r"], ["i"], "t")

    def test_empty_retain_raises(self):
        with pytest.raises(ValueError, match="retain_prompts"):
            LSSEDataset(["f"], [], ["i"], "t")

    def test_sample_retain(self):
        ds = LSSEDataset(["f"], ["r1", "r2"], ["i"], "t")
        assert ds.sample_retain() in ["r1", "r2"]

    def test_sample_implicit(self):
        ds = LSSEDataset(["f"], ["r"], ["x", "y"], "t")
        assert ds.sample_implicit() in ["x", "y"]


# --------------------------------------------------------------------------- #
#  LSSETrainer                                                                 #
# --------------------------------------------------------------------------- #

class TestLSSETrainer:
    @pytest.fixture
    def trainer(self, mock_encoder, mock_tokenizer, device):
        return LSSETrainer(
            text_encoder=mock_encoder,
            tokenizer=mock_tokenizer,
            device=device,
            learning_rate=1e-3,
            alpha=1.0,
            beta=1.0,
            gamma=0.5,
            temperature=0.1,
            max_length=16,
            batch_size=2,
            cap_json_path=None,
            clm_top_k=2,
        )

    def test_concept_direction_computed(self, trainer):
        trainer.precompute_concept_direction(["a", "b", "c"])
        assert trainer.concept_dir is not None

    def test_train_returns_history(self, trainer):
        ds = LSSEDataset(
            explicit_prompts=["a", "b", "c"],
            retain_prompts=["r1", "r2"],
            implicit_concepts=["x", "y"],
            target_concept="t",
        )
        history = trainer.train(ds, num_epochs=1, log_every=1)
        assert len(history) == 1
        assert "L_total" in history[0]

    def test_train_losses_finite(self, trainer):
        ds = LSSEDataset(
            explicit_prompts=["a", "b"],
            retain_prompts=["r1", "r2"],
            implicit_concepts=["x"],
            target_concept="t",
        )
        history = trainer.train(ds, num_epochs=2, log_every=1)
        for epoch_log in history:
            for k, v in epoch_log.items():
                assert torch.isfinite(torch.tensor(float(v))), f"{k}={v}"
