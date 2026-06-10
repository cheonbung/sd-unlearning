"""Shared pytest fixtures for the FCF test suite.

These fixtures provide tiny synthetic encoders/datasets so trainer formulas
can be verified without loading the 350M-parameter CLIP text encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Mock CLIP-like text encoder (deterministic, GPU-free)                      #
# --------------------------------------------------------------------------- #


@dataclass
class _EncoderOutput:
    """Mimics HuggingFace CLIPTextModel return signature."""
    last_hidden_state: torch.Tensor


class MockTextEncoder(nn.Module):
    """A tiny embedding-table encoder.

    Returns a deterministic (B, L, D) tensor whose values depend only on the
    input token IDs — small enough to run unit tests in milliseconds on CPU.
    """

    def __init__(self, vocab_size: int = 1000, max_length: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        torch.manual_seed(0)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> _EncoderOutput:
        hidden = self.embedding(input_ids)
        return _EncoderOutput(last_hidden_state=hidden)


class MockTokenizer:
    """Deterministic char-level tokenizer matching HuggingFace call signature."""

    def __init__(self, max_length: int = 8, vocab_size: int = 1000):
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __call__(
        self,
        texts: List[str],
        padding: str = "max_length",
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: str = "pt",
    ):
        max_length = max_length or self.max_length
        ids = []
        for t in texts:
            row = [ord(c) % self.vocab_size for c in t[:max_length]]
            row += [0] * (max_length - len(row))
            ids.append(row)
        tensor = torch.tensor(ids, dtype=torch.long)

        class _BatchEncoding:
            def __init__(self, input_ids):
                self.input_ids = input_ids

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                return self

        return _BatchEncoding(tensor)


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_encoder() -> MockTextEncoder:
    return MockTextEncoder()


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def tiny_csv(tmp_path):
    """Write a 3-row synthetic FCF CSV. Returns its path.

    Schema (FCF training CSV format):
      prompt_r — retain prompt
      prompt_n — noise prompt (concept replaced)
      prompt_f — explicit (forget) prompt
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "prompt_r": ["a person walking", "a dog running", "a cat sitting"],
            "prompt_n": ["a Xy3#! walking", "a Ab1$! running", "a Cd2%! sitting"],
            "prompt_f": ["a nude person", "a nude dog", "a nude cat"],
        }
    )
    p = tmp_path / "train.csv"
    df.to_csv(p, index=False)
    return p
