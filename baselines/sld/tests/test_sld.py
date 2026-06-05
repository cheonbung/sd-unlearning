"""SLD unit tests (CPU, fast). Run: pytest baselines/sld/tests -v

Asserts the four safety presets match Schramowski et al. exactly and the safety concept covers
the paper's listed categories. Does not load SD (denoise loop is exercised in integration eval).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sld_pipeline import SLD_CONFIGS, SAFETY_CONCEPT, sld_generate  # noqa: E402


def test_presets_match_paper_table():
    # (warmup delta, guidance s_S, threshold lambda, momentum s_m, mom_beta beta_m)
    assert SLD_CONFIGS["weak"] == {"warmup": 15, "guidance_scale": 200, "threshold": 0.0,
                                   "momentum_scale": 0.0, "mom_beta": 0.0}
    assert SLD_CONFIGS["medium"] == {"warmup": 10, "guidance_scale": 1000, "threshold": 0.01,
                                     "momentum_scale": 0.3, "mom_beta": 0.4}
    assert SLD_CONFIGS["strong"] == {"warmup": 7, "guidance_scale": 2000, "threshold": 0.025,
                                     "momentum_scale": 0.5, "mom_beta": 0.7}
    assert SLD_CONFIGS["max"] == {"warmup": 0, "guidance_scale": 5000, "threshold": 1.0,
                                  "momentum_scale": 0.5, "mom_beta": 0.7}


def test_aggressiveness_monotonic():
    # Guidance scale increases and warmup decreases from weak -> max.
    order = ["weak", "medium", "strong", "max"]
    gs = [SLD_CONFIGS[k]["guidance_scale"] for k in order]
    wu = [SLD_CONFIGS[k]["warmup"] for k in order]
    assert gs == sorted(gs) and wu == sorted(wu, reverse=True)


def test_safety_concept_covers_categories():
    for term in ("nudity", "sexual", "violence", "blood", "weapons"):
        assert term in SAFETY_CONCEPT
    assert callable(sld_generate)
