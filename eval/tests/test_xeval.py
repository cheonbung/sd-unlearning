"""xmodel unit tests (CPU, fast). Run: pytest eval/tests -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xeval  # noqa: E402


def test_registry_has_core_models():
    for key in ("raw_v14", "odace_v3", "raw_v15", "sd21base", "safe_neg"):
        assert key in xeval.REGISTRY
        assert xeval.REGISTRY[key]["kind"] in {"sd1", "sd2", "odace"}
    assert xeval.REGISTRY["safe_neg"].get("neg_prompt")
    assert "Manojb/stable-diffusion-2-1-base" in xeval.REGISTRY["sd21base"]["model_id"]


def test_attacks_cover_five():
    keys = {a[0] for a in xeval.ATTACKS}
    assert keys == {"I2P", "Ring-A-Bell", "Ring-A-Bell(Re)", "P4D", "UnlearnDiffAtk"}


def test_no_discarded_custom_metrics():
    # CLIP_atk / CLIP_neu / people_clip / people_asr and their helpers were discarded by
    # user instruction; locality is measured only by the standard COCO protocol (eval_coco.py).
    for gone in ("neutralize", "SEMANTIC", "compute_clip_score"):
        assert not hasattr(xeval, gone), f"discarded symbol '{gone}' reappeared in xeval"


def test_find_sub_variants(tmp_path):
    (tmp_path / "ring_a_bellre").mkdir()
    assert xeval.find_sub(tmp_path, ["ring_a_bell_re", "ring_a_bellre"]).name == "ring_a_bellre"
    assert xeval.find_sub(tmp_path, ["nope"]) is None


def test_reproduced_baselines_registered():
    # Phase 1-4 reference reproductions, all on SD v1.4.
    assert xeval.REGISTRY["esd_u"]["kind"] == "esd"
    assert xeval.REGISTRY["esd_u"]["unet_dir"].startswith("models/esd/")
    for key in ("sld_medium", "sld_strong", "sld_max"):
        assert xeval.REGISTRY[key]["kind"] == "sld"
        assert xeval.REGISTRY[key]["config"] in xeval.SLD_CONFIGS
    assert xeval.REGISTRY["safeclip"]["kind"] == "safeclip"
    assert xeval.REGISTRY["safeclip"]["safeclip_id"] == "aimagelab/safeclip_vit-l_14"


def test_sld_configs_match_paper():
    # Schramowski et al. presets (delta warmup, s_S guidance, lambda threshold, s_m, beta_m).
    m = xeval.SLD_CONFIGS["medium"]
    assert (m["warmup"], m["guidance_scale"], m["threshold"]) == (10, 1000, 0.01)
    mx = xeval.SLD_CONFIGS["max"]
    assert (mx["warmup"], mx["guidance_scale"], mx["threshold"]) == (0, 5000, 1.0)


def test_new_kind_helpers_importable():
    # Generation/loader entry points wired into the harness must be importable.
    assert callable(xeval.sld_generate)
    assert callable(xeval.load_safeclip_text_encoder)
