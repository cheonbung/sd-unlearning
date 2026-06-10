"""
Noise text generation utilities for FCF.

The paper defines P_noise as: replace the explicit target concept in P_explicit
with random textual noise. The noise must:
  - Include symbols, letters, and numbers
  - Have a fixed length of 5 characters
  - Contain no repetitions (no character appears more than once)
"""

import re
import random
import string


def generate_random_noise_text(length: int = 5, seed: int = None) -> str:
    """
    Generate random noise text as defined in the FCF paper.

    Args:
        length: Number of characters (default 5)
        seed: Optional random seed for reproducibility

    Returns:
        A noise string with no repeated characters, containing
        at least one symbol, one letter, and one digit.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    symbols = "!@#$%^&*+-="
    letters = string.ascii_letters   # a-z, A-Z
    digits  = string.digits          # 0-9
    full_charset = list(symbols + letters + digits)

    for _ in range(1000):
        sample = rng.sample(full_charset, length)
        has_sym = any(c in symbols for c in sample)
        has_let = any(c in string.ascii_letters for c in sample)
        has_dig = any(c in string.digits for c in sample)
        if has_sym and has_let and has_dig:
            return "".join(sample)

    # Fallback: force one of each type
    forced = [
        rng.choice(list(symbols)),
        rng.choice(list(letters)),
        rng.choice(list(digits)),
    ]
    remaining_pool = [c for c in full_charset if c not in forced]
    forced += rng.sample(remaining_pool, length - len(forced))
    rng.shuffle(forced)
    return "".join(forced)


def replace_concept_with_noise(prompt: str, concept: str, noise: str = None) -> str:
    """
    Replace every occurrence of `concept` in `prompt` with `noise`.
    Replacement is case-insensitive.

    Args:
        prompt: Original text prompt
        concept: The concept string to replace (e.g. "Van Gogh")
        noise: Replacement noise string; if None, a fresh noise is generated

    Returns:
        Modified prompt with concept replaced by noise
    """
    if noise is None:
        noise = generate_random_noise_text()
    pattern = re.compile(re.escape(concept), re.IGNORECASE)
    return pattern.sub(noise, prompt)


def generate_noise_prompts(
    explicit_prompts: list,
    concept: str,
    fixed_noise: str = None,
) -> list:
    """
    Build a paired noise-prompt list from explicit prompts.
    Each explicit prompt gets a *unique* noise token unless fixed_noise is given.

    Args:
        explicit_prompts: List of prompts containing the target concept
        concept: The explicit concept to replace
        fixed_noise: If provided, use the same noise for all prompts

    Returns:
        List of noise prompts (same length as explicit_prompts)
    """
    noise_prompts = []
    for prompt in explicit_prompts:
        noise = fixed_noise if fixed_noise else generate_random_noise_text()
        noise_prompts.append(replace_concept_with_noise(prompt, concept, noise))
    return noise_prompts
