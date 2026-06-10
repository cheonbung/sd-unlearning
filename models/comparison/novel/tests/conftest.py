"""pytest fixtures shared across tests/test_*.py.

Provides:
  - sys.path insertion so `core` and `methods` are importable from inside tests
  - device fixture (cpu / cuda when available)
  - mock_text_encoder / mock_tokenizer fixtures for CAP analyzer tests (no
    network download required)
  - sample tensors for spherical / OT-noise tests
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn

# Make `core` and `methods` importable from inside the test suite.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_target() -> torch.Tensor:
    """A small (L=4, D=8) target tensor for spherical-projection tests."""
    torch.manual_seed(0)
    return torch.randn(4, 8)


@pytest.fixture
def small_concept() -> torch.Tensor:
    """A small (L=4, D=8) concept tensor for spherical-projection tests."""
    torch.manual_seed(1)
    return torch.randn(4, 8)


# --------------------------------------------------------------------------- #
#  Mock CLIPTextModel-like encoder (for cap_analyzer tests)                    #
# --------------------------------------------------------------------------- #


class _MockEncoderOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _MockTransformerLayer(nn.Module):
    """Single transformer-block stand-in for CAP analyzer tests."""

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 1.01)

    def forward(self, x):
        return (x * self.scale,)


class _MockEncoderLayers(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_MockTransformerLayer(hidden_dim) for _ in range(n_layers)]
        )


class _MockTextModel(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.encoder = _MockEncoderLayers(n_layers, hidden_dim)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim = hidden_dim


class MockCLIPTextModel(nn.Module):
    """Minimal stand-in for HuggingFace CLIPTextModel for CAP unit tests."""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 8, vocab_size: int = 50):
        super().__init__()
        self.text_model = _MockTextModel(n_layers, hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids):
        h = self.text_model.embed(input_ids)
        for layer in self.text_model.encoder.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return _MockEncoderOutput(h)


class MockTokenizer:
    """Minimal CLIP-style tokenizer for tests."""

    def __init__(self, vocab_size: int = 50, max_length: int = 8):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(
        self,
        texts,
        padding: str = "max_length",
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: str = "pt",
    ):
        if isinstance(texts, str):
            texts = [texts]
        L = max_length or self.max_length
        ids = []
        for t in texts:
            ints = [(ord(c) % (self.vocab_size - 1)) + 1 for c in t[:L]]
            while len(ints) < L:
                ints.append(0)
            ids.append(ints)
        ids_t = torch.tensor(ids, dtype=torch.long)

        class _TokOut:
            def __init__(self, input_ids):
                self.input_ids = input_ids

            def to(self, dev):
                return _TokOut(self.input_ids.to(dev))

        return _TokOut(ids_t)


@pytest.fixture
def mock_text_encoder(device) -> MockCLIPTextModel:
    torch.manual_seed(42)
    return MockCLIPTextModel(n_layers=4, hidden_dim=8, vocab_size=50).to(device).eval()


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer(vocab_size=50, max_length=8)
