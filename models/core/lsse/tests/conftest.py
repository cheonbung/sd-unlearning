"""pytest 공용 fixture.

lsse/ 서브 프로젝트 단위 테스트용. 네트워크 다운로드 없이 mock 인코더 사용.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MockEncoderOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _MockLayer(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 1.001)

    def forward(self, x):
        return (x * self.scale,)


class _MockEncoderLayers(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(hidden_dim) for _ in range(n_layers)])


class _MockTextModel(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.encoder = _MockEncoderLayers(n_layers, hidden_dim)
        self.embed = nn.Embedding(vocab_size, hidden_dim)


class MockCLIPTextModel(nn.Module):
    def __init__(self, n_layers: int = 4, hidden_dim: int = 16, vocab_size: int = 64):
        super().__init__()
        self.text_model = _MockTextModel(n_layers, hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids):
        h = self.text_model.embed(input_ids)
        for layer in self.text_model.encoder.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return _MockEncoderOutput(h)

    def save_pretrained(self, path: str):
        pass


class MockTokenizer:
    def __init__(self, vocab_size: int = 64, max_length: int = 16):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(self, texts, padding="max_length", max_length=None,
                 truncation=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        L = max_length or self.max_length
        ids = []
        for t in texts:
            ints = [(ord(c) % (self.vocab_size - 1)) + 1 for c in t[:L]]
            while len(ints) < L:
                ints.append(0)
            ids.append(ints)

        class _Out:
            def __init__(self, input_ids):
                self.input_ids = input_ids
            def to(self, dev):
                return _Out(self.input_ids.to(dev))

        return _Out(torch.tensor(ids, dtype=torch.long))

    def save_pretrained(self, path: str):
        pass


@pytest.fixture
def mock_encoder(device) -> MockCLIPTextModel:
    torch.manual_seed(42)
    return MockCLIPTextModel(n_layers=4, hidden_dim=16, vocab_size=64).to(device)


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer(vocab_size=64, max_length=16)


@pytest.fixture
def sample_embeddings(device) -> torch.Tensor:
    """(N=5, L=4, D=16) — compute_concept_direction 테스트용."""
    torch.manual_seed(7)
    return torch.randn(5, 4, 16, device=device)


@pytest.fixture
def sample_batch(device) -> torch.Tensor:
    """(B=3, L=4, D=16) — cnp_loss / csr_loss 테스트용."""
    torch.manual_seed(11)
    return torch.randn(3, 4, 16, device=device)
