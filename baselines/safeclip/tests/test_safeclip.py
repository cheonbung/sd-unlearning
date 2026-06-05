"""Safe-CLIP unit tests (CPU, fast). Run: pytest baselines/safeclip/tests -v

Asserts the public checkpoint id and the loader entry point. The actual weight swap (network +
SD load) is exercised in the integration eval, not here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safeclip_loader import SAFECLIP_ID, load_safeclip_text_encoder  # noqa: E402


def test_public_checkpoint_id():
    # Phase 0 verified this HF repo exists (ViT-L/14, text hidden_size 768, SD v1.x-compatible).
    assert SAFECLIP_ID == "aimagelab/safeclip_vit-l_14"


def test_loader_is_callable():
    assert callable(load_safeclip_text_encoder)
