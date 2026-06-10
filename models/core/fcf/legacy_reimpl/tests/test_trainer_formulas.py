"""Verify the FCF-P projection and FCF-E experience formulas.

Tests use a mock CLIP encoder (see conftest.py) to keep them fast and
GPU-free while still exercising the actual `FCFTrainer` code path.

FCF-P (Algorithm 2, Eq. 6):
    concept_norm = concept_mean / ||concept_mean||_F
    proj_length  = <target_mean, concept_norm>_F
    cleaned      = target_mean - mu_p * (proj_length * concept_norm)

FCF-E (Algorithm 3, Eq. 7):
    V_emp = mean( T_eps*(P_explicit_i) - T_eps*(P_noise_i) )
"""

import torch

from core.dataset import FCFDataset
from core.trainer import FCFTrainer


def _build_trainer(mock_encoder, mock_tokenizer, cpu_device, **overrides) -> FCFTrainer:
    return FCFTrainer(
        text_encoder=mock_encoder,
        tokenizer=mock_tokenizer,
        device=cpu_device,
        learning_rate=overrides.get("lr", 2.5e-5),
        eta=overrides.get("eta", 0.25),
        mu_p=overrides.get("mu_p", 0.7),
        mu_e=overrides.get("mu_e", 1.0),
        max_length=mock_tokenizer.max_length,
        batch_size=overrides.get("batch_size", 2),
    )


# --------------------------------------------------------------------------- #
#  FCF-P projection formula                                                    #
# --------------------------------------------------------------------------- #


def test_projection_formula_matches_paper(mock_encoder, mock_tokenizer, cpu_device):
    trainer = _build_trainer(mock_encoder, mock_tokenizer, cpu_device, mu_p=0.7)

    implicit = ["male", "boy", "man"]
    concepts = ["nudity", "naked"]

    cleaned = trainer._compute_cleaned_projection_target(
        implicit_concepts=implicit,
        concept_texts=concepts,
        eta_clean=0.7,
    )

    with torch.no_grad():
        concept_feats = trainer._encode(concepts, encoder=trainer.frozen_encoder)
        concept_mean = concept_feats.mean(dim=0)
        concept_norm = concept_mean / concept_mean.norm()

        target_feats = trainer._encode(implicit, encoder=trainer.frozen_encoder)
        target_mean = target_feats.mean(dim=0)

        proj_length = (target_mean * concept_norm).sum()
        expected = target_mean - 0.7 * (proj_length * concept_norm)

    assert torch.allclose(cleaned, expected, atol=1e-6), (
        f"FCF-P formula mismatch (Eq. 6). Max diff: "
        f"{(cleaned - expected).abs().max().item():.6e}"
    )


def test_projection_zero_mu_returns_target(mock_encoder, mock_tokenizer, cpu_device):
    """With mu_p=0 the cleaned target should equal target_mean exactly."""
    trainer = _build_trainer(mock_encoder, mock_tokenizer, cpu_device)

    implicit = ["male", "boy"]
    concepts = ["nudity"]

    cleaned = trainer._compute_cleaned_projection_target(
        implicit_concepts=implicit,
        concept_texts=concepts,
        eta_clean=0.0,
    )

    with torch.no_grad():
        target_feats = trainer._encode(implicit, encoder=trainer.frozen_encoder)
        target_mean = target_feats.mean(dim=0)

    assert torch.allclose(cleaned, target_mean, atol=1e-6)


def test_projection_unit_mu_subtracts_full_projection(mock_encoder, mock_tokenizer, cpu_device):
    trainer = _build_trainer(mock_encoder, mock_tokenizer, cpu_device)

    cleaned = trainer._compute_cleaned_projection_target(
        implicit_concepts=["male"],
        concept_texts=["nudity"],
        eta_clean=1.0,
    )

    with torch.no_grad():
        concept_feats = trainer._encode(["nudity"], encoder=trainer.frozen_encoder)
        concept_mean = concept_feats.mean(dim=0)
        concept_norm = concept_mean / concept_mean.norm()
        target_feats = trainer._encode(["male"], encoder=trainer.frozen_encoder)
        target_mean = target_feats.mean(dim=0)

        residual_along_concept = (cleaned * concept_norm).sum()
        original_along_concept = (target_mean * concept_norm).sum()

    assert abs(residual_along_concept.item()) < 1e-4, (
        f"After full projection removal, residual along concept = "
        f"{residual_along_concept.item():.6e} (should be ~0). "
        f"Original = {original_along_concept.item():.6e}"
    )


def test_projection_target_shape(mock_encoder, mock_tokenizer, cpu_device):
    trainer = _build_trainer(mock_encoder, mock_tokenizer, cpu_device)
    cleaned = trainer._compute_cleaned_projection_target(
        implicit_concepts=["male", "boy"],
        concept_texts=["nudity"],
        eta_clean=0.7,
    )
    assert cleaned.shape == (mock_tokenizer.max_length, mock_encoder.hidden_dim)


# --------------------------------------------------------------------------- #
#  FCF-E experience vector                                                     #
# --------------------------------------------------------------------------- #


def _build_fcfe_dataset() -> FCFDataset:
    return FCFDataset(
        explicit_prompts=["a nude man", "a nude woman", "nude cat"],
        noise_prompts=["a X1#aB man", "a Y2$cD woman", "Z3@eF cat"],
        retain_prompts=["a man", "a woman", "a cat"],
        implicit_groups=[["male", "boy"], ["female", "girl"]],
        maintain_prompts=["safe1"],
        explicit_concepts=["nudity"],
        target_concept="nudity",
    )


def test_experience_formula_matches_paper(mock_encoder, mock_tokenizer, cpu_device):
    """V_emp = mean(T_eps*(P_explicit) - T_eps*(P_noise))  (Eq. 7)"""
    trainer = _build_trainer(mock_encoder, mock_tokenizer, cpu_device, batch_size=2)
    ds = _build_fcfe_dataset()

    experience = trainer.compute_experience(ds)

    with torch.no_grad():
        z_f = trainer._encode(ds.explicit_prompts, encoder=trainer.frozen_encoder)
        z_n = trainer._encode(ds.noise_prompts, encoder=trainer.frozen_encoder)
        expected = (z_f - z_n).mean(dim=0)

    assert torch.allclose(experience, expected, atol=1e-6), (
        f"FCF-E experience formula mismatch (Eq. 7). Max diff: "
        f"{(experience - expected).abs().max().item():.6e}"
    )


def test_experience_shape(mock_encoder, mock_tokenizer, cpu_device):
    trainer = _build_trainer(mock_encoder, mock_tokenizer, cpu_device, batch_size=2)
    ds = _build_fcfe_dataset()
    experience = trainer.compute_experience(ds)
    assert experience.shape == (mock_tokenizer.max_length, mock_encoder.hidden_dim)


def test_experience_zero_when_explicit_equals_noise(mock_encoder, mock_tokenizer, cpu_device):
    """If P_explicit == P_noise, the experience vector should be exactly zero."""
    trainer = _build_trainer(mock_encoder, mock_tokenizer, cpu_device, batch_size=2)

    same = ["a man", "a woman", "a cat"]
    ds = FCFDataset(
        explicit_prompts=same,
        noise_prompts=list(same),
        retain_prompts=same,
        implicit_groups=[["male"]],
        maintain_prompts=["x"],
        explicit_concepts=["c"],
        target_concept="c",
    )

    experience = trainer.compute_experience(ds)
    assert torch.allclose(experience, torch.zeros_like(experience), atol=1e-6)


# --------------------------------------------------------------------------- #
#  Frozen encoder isolation                                                    #
# --------------------------------------------------------------------------- #


def test_frozen_encoder_not_modified_by_training_state(mock_encoder, mock_tokenizer, cpu_device):
    trainer = _build_trainer(mock_encoder, mock_tokenizer, cpu_device)

    with torch.no_grad():
        before = trainer._encode(["hello"], encoder=trainer.frozen_encoder).clone()

    with torch.no_grad():
        for p in trainer.text_encoder.parameters():
            p.add_(1.0)

    with torch.no_grad():
        after = trainer._encode(["hello"], encoder=trainer.frozen_encoder)

    assert torch.allclose(before, after, atol=1e-6), (
        "Frozen encoder must be decoupled from trainable encoder (deep-copied)."
    )


def test_frozen_encoder_has_no_grad(mock_encoder, mock_tokenizer, cpu_device):
    trainer = _build_trainer(mock_encoder, mock_tokenizer, cpu_device)
    for p in trainer.frozen_encoder.parameters():
        assert not p.requires_grad, "Frozen encoder must have requires_grad=False"
