"""Unit tests for methods/cap_analyzer.py (N1)."""

from __future__ import annotations

import json

import pytest
import torch

from methods.cap_analyzer import (
    CAPConfig,
    CAPResult,
    CausalActivationPatcher,
    _get_encoder_layers,
)


def test_get_encoder_layers_finds_layers(mock_text_encoder):
    layers = _get_encoder_layers(mock_text_encoder)
    assert len(layers) == 4  # MockCLIPTextModel has 4 layers


def test_get_encoder_layers_raises_on_missing():
    class Empty:
        pass
    with pytest.raises(AttributeError):
        _get_encoder_layers(Empty())


def test_patcher_init(mock_text_encoder, mock_tokenizer, device):
    p = CausalActivationPatcher(
        text_encoder=mock_text_encoder,
        tokenizer=mock_tokenizer,
        device=device,
        max_length=8,
        config=CAPConfig(pool="mean", score_norm="l2"),
    )
    assert p.n_layers == 4


def test_patcher_analyze_returns_result(mock_text_encoder, mock_tokenizer, device):
    p = CausalActivationPatcher(
        text_encoder=mock_text_encoder,
        tokenizer=mock_tokenizer,
        device=device,
        max_length=8,
        config=CAPConfig(pool="mean", score_norm="l2"),
    )
    result = p.analyze(
        explicit_prompts=["nude man", "nude woman", "nude body"],
        noise_prompts=["a7$Bx", "Q1#mZ", "9w&Ey"],
    )
    assert isinstance(result, CAPResult)
    assert result.n_layers == 4
    assert len(result.layer_scores) == 4
    assert 0 <= result.max_layer_idx < 4


def test_patcher_assert_length_mismatch(mock_text_encoder, mock_tokenizer, device):
    p = CausalActivationPatcher(
        text_encoder=mock_text_encoder,
        tokenizer=mock_tokenizer,
        device=device,
        max_length=8,
    )
    with pytest.raises(AssertionError):
        p.analyze(
            explicit_prompts=["a", "b", "c"],
            noise_prompts=["x", "y"],
        )


def test_patcher_pool_modes(mock_text_encoder, mock_tokenizer, device):
    for pool in ["mean", "cls", "eos"]:
        p = CausalActivationPatcher(
            text_encoder=mock_text_encoder,
            tokenizer=mock_tokenizer,
            device=device,
            max_length=8,
            config=CAPConfig(pool=pool, score_norm="l2"),
        )
        result = p.analyze(
            explicit_prompts=["a", "b"],
            noise_prompts=["x", "y"],
        )
        assert len(result.layer_scores) == 4


def test_patcher_invalid_pool_raises(mock_text_encoder, mock_tokenizer, device):
    p = CausalActivationPatcher(
        text_encoder=mock_text_encoder,
        tokenizer=mock_tokenizer,
        device=device,
        max_length=8,
        config=CAPConfig(pool="bogus", score_norm="l2"),
    )
    with pytest.raises(ValueError, match="Unknown pool"):
        p.analyze(["a", "b"], ["x", "y"])


def test_patcher_invalid_score_norm_raises(mock_text_encoder, mock_tokenizer, device):
    p = CausalActivationPatcher(
        text_encoder=mock_text_encoder,
        tokenizer=mock_tokenizer,
        device=device,
        max_length=8,
        config=CAPConfig(pool="mean", score_norm="bogus"),
    )
    with pytest.raises(ValueError, match="Unknown score_norm"):
        p.analyze(["a", "b"], ["x", "y"])


def test_cap_result_save_and_top_k(tmp_path):
    cfg = CAPConfig(pool="mean", score_norm="l2")
    result = CAPResult(
        n_layers=4,
        layer_scores=[0.1, 0.9, 0.3, 0.5],
        max_layer_idx=1,
        config=cfg,
    )
    path = tmp_path / "cap.json"
    result.save(str(path))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["n_layers"] == 4
    assert data["max_layer_idx"] == 1
    assert data["config"]["pool"] == "mean"

    top2 = result.top_k_layers(2)
    assert top2[0] == 1  # highest score
    assert top2[1] == 3  # second-highest
