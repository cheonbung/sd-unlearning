"""LSSEDataset — 단순화된 데이터셋 (noise 프롬프트 불필요).

FCFDataset과의 차이:
  FCF는 (explicit, noise, retain) 3-tuple 필요.
  LSSE는 CNP가 noise를 대체하므로 (explicit, retain, implicit) 만 필요.
  단순한 구조로 데이터 준비 부담 감소.

Callers:
  - train_lsse.py (build_dataset)

Data formats:
  Plain text files: UTF-8, 한 줄에 하나, '#'로 시작하는 줄은 주석.
  No date fields.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List


class LSSEDataset:
    """LSSE 학습용 프롬프트 집합.

    noise_prompts 없음 — N7 CNP가 기하학적으로 망각 방향을 정의.
    """

    def __init__(
        self,
        explicit_prompts: List[str],
        retain_prompts: List[str],
        implicit_concepts: List[str],
        target_concept: str,
        seed: int = 42,
    ):
        if not explicit_prompts:
            raise ValueError("explicit_prompts가 비어 있음.")
        if not retain_prompts:
            raise ValueError("retain_prompts가 비어 있음.")
        self.explicit_prompts = explicit_prompts
        self.retain_prompts = retain_prompts
        self.implicit_concepts = implicit_concepts
        self.target_concept = target_concept
        self.seed = seed
        random.seed(seed)

    def __len__(self) -> int:
        return len(self.explicit_prompts)

    def sample_retain(self) -> str:
        return random.choice(self.retain_prompts)

    def sample_implicit(self) -> str:
        if not self.implicit_concepts:
            raise ValueError("implicit_concepts가 비어 있음.")
        return random.choice(self.implicit_concepts)

    @classmethod
    def from_files(
        cls,
        explicit_file: str,
        retain_file: str,
        implicit_file: str,
        target_concept: str,
        seed: int = 42,
    ) -> "LSSEDataset":
        """Plain text 파일에서 로드."""

        def _load(path: str) -> List[str]:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"프롬프트 파일 없음: {path}")
            lines = p.read_text(encoding="utf-8").splitlines()
            return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]

        return cls(
            explicit_prompts=_load(explicit_file),
            retain_prompts=_load(retain_file),
            implicit_concepts=_load(implicit_file),
            target_concept=target_concept,
            seed=seed,
        )

    def __repr__(self) -> str:
        return (
            f"LSSEDataset(concept='{self.target_concept}', "
            f"explicit={len(self.explicit_prompts)}, "
            f"retain={len(self.retain_prompts)}, "
            f"implicit={len(self.implicit_concepts)})"
        )
