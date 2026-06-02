"""
Dataset utilities for FCF training.

Supports two input formats:
  1. CSV format  (original repo style):
       columns: prompt_r, prompt_n, prompt_f
  2. Plain text files (one entry per line)

Also holds implicit_groups: a list of concept-word groups for Stage 2.
The original repo uses two gender groups for nudity:
  [["male", "boy", "man"], ["female", "girl", "woman"]]
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

from .noise_utils import generate_noise_prompts


class FCFDataset:
    """
    Holds all prompt sets needed for FCF training.

    Attributes:
        explicit_prompts:  Prompts containing the target concept (prompt_f).
        noise_prompts:     explicit_prompts with concept replaced by random noise (prompt_n).
        retain_prompts:    Safe/neutral retain prompts paired with explicit_prompts (prompt_r).
        implicit_groups:   List of concept-word groups for Stage 2 implicit forgetting.
                           Each group is a list of strings processed as a batch.
        maintain_prompts:  Unpaired safe prompts (used as retain_text fallback).
        explicit_concepts: Concept name variants for FCF-P projection direction.
        target_concept:    The target concept string (e.g. "nudity").
    """

    def __init__(
        self,
        explicit_prompts: List[str],
        noise_prompts: List[str],
        retain_prompts: List[str],
        implicit_groups: List[List[str]],
        maintain_prompts: List[str],
        explicit_concepts: List[str],
        target_concept: str,
        seed: int = 42,
    ):
        assert len(explicit_prompts) == len(noise_prompts) == len(retain_prompts), (
            "explicit_prompts, noise_prompts, and retain_prompts must have equal length"
        )
        self.explicit_prompts  = explicit_prompts
        self.noise_prompts     = noise_prompts
        self.retain_prompts    = retain_prompts
        self.implicit_groups   = implicit_groups
        self.maintain_prompts  = maintain_prompts
        self.explicit_concepts = explicit_concepts
        self.target_concept    = target_concept
        self.seed              = seed

        random.seed(seed)

    def __len__(self) -> int:
        return len(self.explicit_prompts)

    # ------------------------------------------------------------------
    # Sampling helpers (for ablation/sweep scripts)
    # ------------------------------------------------------------------

    def sample_explicit_pair(self) -> Tuple[str, str]:
        """Return a random (explicit_prompt, noise_prompt) pair."""
        idx = random.randint(0, len(self.explicit_prompts) - 1)
        return self.explicit_prompts[idx], self.noise_prompts[idx]

    def sample_maintain_prompt(self) -> str:
        return random.choice(self.maintain_prompts)

    def sample_implicit_concept(self) -> str:
        """Return a random implicit concept from any group."""
        flat = [c for g in self.implicit_groups for c in g]
        return random.choice(flat)

    def sample_explicit_concept(self) -> str:
        return random.choice(self.explicit_concepts)

    # ------------------------------------------------------------------
    # Class methods: load from CSV (primary/original-repo format)
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        train_csv: str,
        implicit_groups: List[List[str]],
        explicit_concepts: List[str],
        target_concept: str,
        maintain_prompts: Optional[List[str]] = None,
        seed: int = 42,
    ) -> "FCFDataset":
        """
        Load from a CSV file with columns: prompt_r, prompt_n, prompt_f
        (matching the original FCF repository's data/train/ CSVs).

        Args:
            train_csv:         Path to CSV file.
            implicit_groups:   List of concept-word groups, e.g.
                               [["male","boy","man"], ["female","girl","woman"]]
            explicit_concepts: Concept name variants for FCF-P direction.
            target_concept:    Target concept string.
            maintain_prompts:  Optional extra safe prompts list.
            seed:              Random seed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV loading: pip install pandas")

        df = pd.read_csv(train_csv)
        required = {"prompt_r", "prompt_n", "prompt_f"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        explicit_prompts = [str(r) for r in df["prompt_f"].tolist()]
        noise_prompts    = [str(r) for r in df["prompt_n"].tolist()]
        retain_prompts   = [str(r) for r in df["prompt_r"].tolist()]

        if maintain_prompts is None:
            maintain_prompts = retain_prompts

        return cls(
            explicit_prompts  = explicit_prompts,
            noise_prompts     = noise_prompts,
            retain_prompts    = retain_prompts,
            implicit_groups   = implicit_groups,
            maintain_prompts  = maintain_prompts,
            explicit_concepts = explicit_concepts,
            target_concept    = target_concept,
            seed              = seed,
        )

    # ------------------------------------------------------------------
    # Class methods: load from plain text files (backward compatibility)
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        explicit_prompts_file: str,
        implicit_concepts_file: str,
        maintain_prompts_file: str,
        explicit_concepts_file: str,
        target_concept: str,
        noise_prompts_file: Optional[str] = None,
        retain_prompts_file: Optional[str] = None,
        implicit_groups_config: Optional[List[List[str]]] = None,
        seed: int = 42,
    ) -> "FCFDataset":
        """
        Load from plain text files (one entry per line).

        implicit_groups_config: optional list of lists for grouping implicit concepts.
            If None, all implicit concepts are treated as one group.
        """

        def load_lines(path: str) -> List[str]:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Prompt file not found: {path}")
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
            return [ln for ln in lines if ln and not ln.startswith("#")]

        explicit_prompts  = load_lines(explicit_prompts_file)
        implicit_concepts = load_lines(implicit_concepts_file)
        maintain_prompts  = load_lines(maintain_prompts_file)
        explicit_concepts = load_lines(explicit_concepts_file)

        noise_prompts = (
            load_lines(noise_prompts_file)
            if noise_prompts_file
            else generate_noise_prompts(explicit_prompts, target_concept)
        )

        retain_prompts = (
            load_lines(retain_prompts_file)
            if retain_prompts_file
            else maintain_prompts[:len(explicit_prompts)] + [maintain_prompts[0]] * max(
                0, len(explicit_prompts) - len(maintain_prompts)
            )
        )

        # Build implicit_groups
        if implicit_groups_config:
            implicit_groups = implicit_groups_config
        else:
            implicit_groups = [implicit_concepts]

        return cls(
            explicit_prompts  = explicit_prompts,
            noise_prompts     = noise_prompts,
            retain_prompts    = retain_prompts,
            implicit_groups   = implicit_groups,
            maintain_prompts  = maintain_prompts,
            explicit_concepts = explicit_concepts,
            target_concept    = target_concept,
            seed              = seed,
        )

    def __repr__(self) -> str:
        n_implicit = sum(len(g) for g in self.implicit_groups)
        return (
            f"FCFDataset(concept='{self.target_concept}', "
            f"explicit={len(self.explicit_prompts)}, "
            f"implicit_groups={len(self.implicit_groups)} ({n_implicit} concepts), "
            f"maintain={len(self.maintain_prompts)})"
        )
