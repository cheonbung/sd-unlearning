"""Verify FCF noise-text generation matches the paper specification.

Paper spec for P_noise:
  - Fixed length of 5 characters
  - No repeated characters
  - Must contain at least one symbol, one letter, and one digit
"""

import string

import pytest

from core.noise_utils import (
    generate_noise_prompts,
    generate_random_noise_text,
    replace_concept_with_noise,
)

SYMBOLS = "!@#$%^&*+-="


def _has_symbol(s: str) -> bool:
    return any(c in SYMBOLS for c in s)


def _has_letter(s: str) -> bool:
    return any(c in string.ascii_letters for c in s)


def _has_digit(s: str) -> bool:
    return any(c in string.digits for c in s)


# --------------------------------------------------------------------------- #
#  generate_random_noise_text                                                  #
# --------------------------------------------------------------------------- #


def test_noise_default_length_is_5():
    n = generate_random_noise_text(seed=0)
    assert len(n) == 5, f"paper specifies length=5, got {len(n)}"


@pytest.mark.parametrize("length", [3, 5, 8, 10])
def test_noise_arbitrary_length(length):
    n = generate_random_noise_text(length=length, seed=0)
    assert len(n) == length


def test_noise_no_repeated_characters():
    for seed in range(50):
        n = generate_random_noise_text(seed=seed)
        assert len(set(n)) == len(n), f"noise '{n}' (seed={seed}) has repeats"


def test_noise_has_symbol_letter_digit():
    for seed in range(50):
        n = generate_random_noise_text(seed=seed)
        assert _has_symbol(n), f"'{n}' lacks symbol"
        assert _has_letter(n), f"'{n}' lacks letter"
        assert _has_digit(n), f"'{n}' lacks digit"


def test_noise_reproducible_with_seed():
    a = generate_random_noise_text(seed=123)
    b = generate_random_noise_text(seed=123)
    assert a == b, "same seed must produce same noise"


def test_noise_different_seeds_likely_differ():
    samples = {generate_random_noise_text(seed=s) for s in range(20)}
    assert len(samples) >= 18


# --------------------------------------------------------------------------- #
#  replace_concept_with_noise                                                  #
# --------------------------------------------------------------------------- #


def test_replace_basic():
    out = replace_concept_with_noise("a nude person", "nude", noise="X1#aB")
    assert out == "a X1#aB person"


def test_replace_case_insensitive():
    out = replace_concept_with_noise("a NUDE Person", "nude", noise="X1#aB")
    assert out == "a X1#aB Person"


def test_replace_multiple_occurrences():
    out = replace_concept_with_noise("nude nude nude", "nude", noise="X1#aB")
    assert out == "X1#aB X1#aB X1#aB"


def test_replace_no_match_returns_original():
    out = replace_concept_with_noise("a person walking", "nude", noise="X1#aB")
    assert out == "a person walking"


def test_replace_handles_regex_metacharacters():
    """concept must be regex-escaped — '.' must not match underscore."""
    out = replace_concept_with_noise("file.ext name", "file.ext", noise="X1#aB")
    assert out == "X1#aB name"
    out2 = replace_concept_with_noise("file_ext name", "file.ext", noise="X1#aB")
    assert out2 == "file_ext name"


# --------------------------------------------------------------------------- #
#  generate_noise_prompts                                                      #
# --------------------------------------------------------------------------- #


def test_generate_noise_prompts_preserves_count():
    prompts = ["a nude man", "a nude woman", "a nude child"]
    out = generate_noise_prompts(prompts, concept="nude")
    assert len(out) == len(prompts)


def test_generate_noise_prompts_replaces_each():
    prompts = ["a nude man", "a nude woman"]
    out = generate_noise_prompts(prompts, concept="nude")
    for original, noisy in zip(prompts, out):
        assert "nude" not in noisy.lower(), f"concept leaked: {noisy}"
        assert noisy != original


def test_generate_noise_prompts_fixed_noise():
    prompts = ["a nude man", "a nude woman"]
    out = generate_noise_prompts(prompts, concept="nude", fixed_noise="X1#aB")
    assert all("X1#aB" in p for p in out)
