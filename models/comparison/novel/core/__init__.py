"""fcf-novel-methods · core

Local, self-contained copy of the base FCF training utilities. This package
is INDEPENDENT of the main `fcf/` package — it ships its own copies of the
trainer, dataset, and noise helpers so that this sub-project can run without
the parent thesis code present.

Modules:
  trainer       - FCFTrainer  (Stage 1 + Stage 2 FCF-P / FCF-E)
  dataset       - FCFDataset  (CSV + text-file loaders)
  noise_utils   - 5-char paper-spec noise text generation
"""

from .trainer import FCFTrainer
from .dataset import FCFDataset
from .noise_utils import (
    generate_random_noise_text,
    replace_concept_with_noise,
    generate_noise_prompts,
)

__all__ = [
    "FCFTrainer",
    "FCFDataset",
    "generate_random_noise_text",
    "replace_concept_with_noise",
    "generate_noise_prompts",
]
