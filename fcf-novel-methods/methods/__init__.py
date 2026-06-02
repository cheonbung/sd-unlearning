"""fcf-novel-methods · methods

Novel research methods that extend the base FCF training (defined in core/):

  - spherical.py    - N5: Riemannian Geodesic Forgetting (manifold-aware projection)
  - ot_noise.py     - N6: Optimal-Transport-derived noise prompts
  - cap_analyzer.py - N1: Causal Activation Patching (interpretability analysis)
  - trainer.py      - NovelFCFTrainer (subclass of core.FCFTrainer wiring N5/N6)

All modules depend ONLY on `core/` and PyTorch / transformers — they do not
import from the parent thesis package `fcf/`.
"""

from .spherical import spherical_cleaned_target, manifold_cleaned_target
from .trainer import NovelFCFTrainer

__all__ = [
    "spherical_cleaned_target",
    "manifold_cleaned_target",
    "NovelFCFTrainer",
]
