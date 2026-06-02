"""Verify FCFDataset loading and sampling behavior."""

import pytest

from core.dataset import FCFDataset


# --------------------------------------------------------------------------- #
#  from_csv                                                                    #
# --------------------------------------------------------------------------- #


def test_from_csv_loads_three_columns(tiny_csv):
    ds = FCFDataset.from_csv(
        train_csv=str(tiny_csv),
        implicit_groups=[["male", "boy"], ["female", "girl"]],
        explicit_concepts=["nudity", "naked"],
        target_concept="nudity",
    )
    assert len(ds.explicit_prompts) == 3
    assert len(ds.noise_prompts) == 3
    assert len(ds.retain_prompts) == 3
    assert ds.target_concept == "nudity"


def test_from_csv_paired_lengths(tiny_csv):
    ds = FCFDataset.from_csv(
        train_csv=str(tiny_csv),
        implicit_groups=[["male"], ["female"]],
        explicit_concepts=["nudity"],
        target_concept="nudity",
    )
    assert len(ds.explicit_prompts) == len(ds.noise_prompts) == len(ds.retain_prompts)


def test_from_csv_preserves_implicit_groups(tiny_csv):
    groups = [["male", "boy", "man"], ["female", "girl", "woman"]]
    ds = FCFDataset.from_csv(
        train_csv=str(tiny_csv),
        implicit_groups=groups,
        explicit_concepts=["nudity"],
        target_concept="nudity",
    )
    assert ds.implicit_groups == groups


def test_from_csv_missing_column_raises(tmp_path):
    import pandas as pd

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"prompt_r": ["x"], "prompt_n": ["y"]}).to_csv(bad, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        FCFDataset.from_csv(
            train_csv=str(bad),
            implicit_groups=[["x"]],
            explicit_concepts=["nudity"],
            target_concept="nudity",
        )


# --------------------------------------------------------------------------- #
#  Length assertion in __init__                                                #
# --------------------------------------------------------------------------- #


def test_unequal_prompt_lengths_raise():
    with pytest.raises(AssertionError):
        FCFDataset(
            explicit_prompts=["a", "b"],
            noise_prompts=["x"],
            retain_prompts=["q", "w"],
            implicit_groups=[["m"]],
            maintain_prompts=["m"],
            explicit_concepts=["c"],
            target_concept="c",
        )


# --------------------------------------------------------------------------- #
#  Sampling helpers                                                            #
# --------------------------------------------------------------------------- #


def _make_dataset() -> FCFDataset:
    return FCFDataset(
        explicit_prompts=["a nude man", "a nude woman"],
        noise_prompts=["a X1#aB man", "a Y2$cD woman"],
        retain_prompts=["a man", "a woman"],
        implicit_groups=[["male", "boy"], ["female", "girl"]],
        maintain_prompts=["safe1", "safe2"],
        explicit_concepts=["nudity", "naked"],
        target_concept="nudity",
        seed=42,
    )


def test_sample_explicit_pair_returns_aligned_pair():
    ds = _make_dataset()
    forget, noise = ds.sample_explicit_pair()
    assert forget in ds.explicit_prompts
    idx = ds.explicit_prompts.index(forget)
    assert noise == ds.noise_prompts[idx]


def test_sample_implicit_concept_in_groups():
    ds = _make_dataset()
    c = ds.sample_implicit_concept()
    flat = [w for g in ds.implicit_groups for w in g]
    assert c in flat


def test_sample_explicit_concept_in_list():
    ds = _make_dataset()
    assert ds.sample_explicit_concept() in ds.explicit_concepts


def test_repr_includes_concept():
    ds = _make_dataset()
    assert "nudity" in repr(ds)


# --------------------------------------------------------------------------- #
#  from_files                                                                  #
# --------------------------------------------------------------------------- #


def test_from_files_basic(tmp_path):
    explicit = tmp_path / "explicit.txt"
    implicit = tmp_path / "implicit.txt"
    maintain = tmp_path / "maintain.txt"
    concepts = tmp_path / "concepts.txt"

    explicit.write_text("a nude man\na nude woman\n", encoding="utf-8")
    implicit.write_text("male\nfemale\n# comment line\n", encoding="utf-8")
    maintain.write_text("a person\na human\n", encoding="utf-8")
    concepts.write_text("nudity\nnaked\n", encoding="utf-8")

    ds = FCFDataset.from_files(
        explicit_prompts_file=str(explicit),
        implicit_concepts_file=str(implicit),
        maintain_prompts_file=str(maintain),
        explicit_concepts_file=str(concepts),
        target_concept="nudity",
    )
    assert len(ds.explicit_prompts) == 2
    assert len(ds.noise_prompts) == 2
    assert "# comment line" not in [c for g in ds.implicit_groups for c in g]


def test_from_files_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FCFDataset.from_files(
            explicit_prompts_file=str(tmp_path / "nonexistent.txt"),
            implicit_concepts_file=str(tmp_path / "implicit.txt"),
            maintain_prompts_file=str(tmp_path / "maintain.txt"),
            explicit_concepts_file=str(tmp_path / "concepts.txt"),
            target_concept="nudity",
        )
