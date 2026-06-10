"""N1 — Causal Activation Patching (CAP) — analysis phase only.

Implements layer-level causal mediation analysis on the CLIP text encoder
to identify *which layers* are most responsible for encoding a target
concept (e.g., "nudity"). This is the ANALYSIS phase of CAP-FCF; the editing
phase (ROME-style rank-1 update) is intentionally deferred to a follow-up
experiment.

Methodology (Vig et al., NeurIPS 2020; Meng et al., NeurIPS 2022 ROME):
  For each layer L in the CLIP text encoder:
    1. Run a "noise" forward pass; cache every layer's output activation
    2. Run a "clean" (explicit-prompt) forward pass; cache the unmodified output
    3. Run a "patched" forward pass: at layer L, replace the layer output with
       the cached noise activation; let downstream layers continue normally
    4. score(L) = ||clean_output - patched_output||  (mean over a batch)

  Layers with high score are *causally implicated* in encoding the concept.

Callers:
  - scripts/run_cap_analysis.py (CLI entry point)
  - tests/test_cap_analyzer.py

Data formats:
  JSON schema (CAPResult.save):
    {
      "n_layers":      12,
      "layer_scores":  [0.1, 0.5, ..., 1.2],
      "max_layer_idx": 7,
      "config":        {"pool": "mean", "score_norm": "l2"}
    }
  No date fields.

Reference: research idea N1 in .claude/plan/fcf-research-ideas-v2.md
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import torch


def _get_encoder_layers(text_encoder) -> List:
    """Locate the transformer-block list inside a HuggingFace CLIPTextModel."""
    candidates = [
        ("text_model", "encoder", "layers"),
        ("encoder", "layers"),
    ]
    for path in candidates:
        obj = text_encoder
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, (list, torch.nn.ModuleList)):
            return list(obj)
    raise AttributeError(
        "Could not locate encoder.layers on the provided text_encoder. "
        "Expected .text_model.encoder.layers or .encoder.layers."
    )


@dataclass
class CAPConfig:
    pool: str = "mean"      # "mean" | "cls" | "eos"
    score_norm: str = "l2"  # "l2" | "l1" | "cos_dist"


class CausalActivationPatcher:
    """Layer-level causal mediation analyzer for a frozen CLIPTextModel."""

    def __init__(
        self,
        text_encoder,
        tokenizer,
        device,
        max_length: int = 77,
        config: Optional[CAPConfig] = None,
    ):
        self.text_encoder = text_encoder.to(device)
        self.text_encoder.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.config = config or CAPConfig()
        self.layers = _get_encoder_layers(self.text_encoder)
        self.n_layers = len(self.layers)

    @torch.no_grad()
    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    @torch.no_grad()
    def _forward_with_cache(self, texts: List[str]):
        cached: List[torch.Tensor] = []

        def make_hook(idx):
            def _hook(module, input, output):
                t = output[0] if isinstance(output, tuple) else output
                cached.append(t.detach().clone())
            return _hook

        handles = [
            layer.register_forward_hook(make_hook(i))
            for i, layer in enumerate(self.layers)
        ]
        try:
            tokens = self._tokenize(texts)
            final = self.text_encoder(tokens.input_ids).last_hidden_state
        finally:
            for h in handles:
                h.remove()

        return final, cached

    @torch.no_grad()
    def _forward_with_patch(
        self,
        texts: List[str],
        patch_layer_idx: int,
        replacement: torch.Tensor,
        patch_token_idx: int = 1,
    ) -> torch.Tensor:
        # CAP bug fix: full-sequence replacement collapses every layer to the
        # noise model's final output (a deterministic re-encoding). Patch only
        # the first content-token position so downstream attention mixes the
        # injection differently per layer, producing per-layer differentiation.
        def _hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new = hidden.clone()
            new[:, patch_token_idx, :] = replacement[:, patch_token_idx, :]
            if isinstance(output, tuple):
                return (new,) + output[1:]
            return new

        handle = self.layers[patch_layer_idx].register_forward_hook(_hook)
        try:
            tokens = self._tokenize(texts)
            patched = self.text_encoder(tokens.input_ids).last_hidden_state
        finally:
            handle.remove()
        return patched

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.config.pool == "mean":
            return hidden.mean(dim=1)
        if self.config.pool == "cls":
            return hidden[:, 0, :]
        if self.config.pool == "eos":
            return hidden[:, -1, :]
        raise ValueError(f"Unknown pool mode: {self.config.pool!r}")

    def _score(self, clean: torch.Tensor, patched: torch.Tensor) -> float:
        c = self._pool(clean)
        p = self._pool(patched)
        if self.config.score_norm == "l2":
            return float((c - p).norm(dim=-1).mean().item())
        if self.config.score_norm == "l1":
            return float((c - p).abs().sum(dim=-1).mean().item())
        if self.config.score_norm == "cos_dist":
            c_n = c / (c.norm(dim=-1, keepdim=True) + 1e-9)
            p_n = p / (p.norm(dim=-1, keepdim=True) + 1e-9)
            return float((1.0 - (c_n * p_n).sum(dim=-1)).mean().item())
        raise ValueError(f"Unknown score_norm: {self.config.score_norm!r}")

    @torch.no_grad()
    def analyze(
        self,
        explicit_prompts: List[str],
        noise_prompts: List[str],
    ) -> "CAPResult":
        """Run layer-level causal mediation analysis."""
        assert len(explicit_prompts) == len(noise_prompts), (
            f"len mismatch: explicit={len(explicit_prompts)} noise={len(noise_prompts)}"
        )

        _, noise_cache = self._forward_with_cache(noise_prompts)
        clean_out, _ = self._forward_with_cache(explicit_prompts)

        import logging as _lg
        _logger = _lg.getLogger("cap_analyzer.debug")
        for i, nc in enumerate(noise_cache):
            _logger.warning(
                f"[CAP-DBG] noise_cache[{i:02d}] shape={tuple(nc.shape)} "
                f"norm={nc.norm().item():.4f} "
                f"first3={nc.flatten()[:3].tolist()}"
            )
        _logger.warning(
            f"[CAP-DBG] clean_out shape={tuple(clean_out.shape)} "
            f"norm={clean_out.norm().item():.4f} "
            f"first3={clean_out.flatten()[:3].tolist()}"
        )

        layer_scores: List[float] = []
        for layer_idx in range(self.n_layers):
            patched_out = self._forward_with_patch(
                explicit_prompts,
                patch_layer_idx=layer_idx,
                replacement=noise_cache[layer_idx],
            )
            score = self._score(clean_out, patched_out)
            layer_scores.append(score)
            _logger.warning(
                f"[CAP-DBG] patched[{layer_idx:02d}] "
                f"norm={patched_out.norm().item():.4f} "
                f"first3={patched_out.flatten()[:3].tolist()} "
                f"score={score:.6f} "
                f"diff_norm={(clean_out - patched_out).norm().item():.4f}"
            )

        max_layer = int(max(range(self.n_layers), key=lambda i: layer_scores[i]))

        return CAPResult(
            n_layers=self.n_layers,
            layer_scores=layer_scores,
            max_layer_idx=max_layer,
            config=self.config,
        )


@dataclass
class CAPResult:
    """Output of CausalActivationPatcher.analyze(...)."""
    n_layers: int
    layer_scores: List[float]
    max_layer_idx: int
    config: CAPConfig

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "layer_scores": self.layer_scores,
            "max_layer_idx": self.max_layer_idx,
            "config": self.config.__dict__,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def top_k_layers(self, k: int = 3) -> List[int]:
        return sorted(range(self.n_layers), key=lambda i: -self.layer_scores[i])[:k]
