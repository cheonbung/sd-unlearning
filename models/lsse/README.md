# LSSE - Layer-Selective Semantic Erasure

Last updated: 2026-06-05

`models/lsse/` implements **LSSE**, short for **Layer-Selective Semantic Erasure**.
It is an independent text-encoder unlearning framework. It does not inherit from
FCF and does not use FCF's random noise prompt target.

LSSE edits only the **CLIP text encoder** used by Stable Diffusion v1.4. The
UNet and VAE remain frozen.

## What Is Trained

```text
prompt -> CLIP text encoder -> text embedding -> SD UNet -> image
            ^ selected layers trained here
```

Model components:

| Component | Value |
|---|---|
| Text encoder | `openai/clip-vit-large-patch14` |
| SD evaluation base | `CompVis/stable-diffusion-v1-4` |
| Trainable module | Selected CLIP text encoder layers |
| Frozen modules | SD UNet, VAE, frozen reference text encoder |
| Target concept used here | `nudity` |

## Core Ideas

LSSE combines three main components.

| Abbrev. | Full name | Purpose |
|---|---|---|
| `CNP` | Concept Null-Space Projection | Remove the explicit concept direction from forget and implicit embeddings. |
| `CSR` | Contrastive Semantic Retention | Preserve retain prompts by matching current embeddings to frozen embeddings with InfoNCE. |
| `CLM` | CAP-Guided Layer Masking | Train only concept-relevant CLIP layers and freeze the rest. |

### CNP

`CNP` extracts a concept direction from explicit prompt embeddings using SVD.
During training, it penalizes projection onto that direction:

```text
L_CNP = mean((z_current dot c_dir)^2)
```

This replaces FCF's arbitrary noise-prompt destination with a geometric
null-space objective. See `methods/cnp.py`.

### CSR

`CSR` is a SimCLR-style retention loss. Retain prompts encoded by the current
model should match the same prompts encoded by the frozen reference model.

```text
query = normalize(mean_tokens(z_retain_current))
key   = normalize(mean_tokens(z_retain_frozen))
L_CSR = cross_entropy(query @ key.T / temperature)
```

See `methods/csr.py`.

### CLM

`CLM` freezes most CLIP layers and opens only the top concept-relevant layers.
If a CAP JSON is not provided, the fallback opens the earliest layers. See
`methods/clm.py`.

## Training Objective

The default LSSE objective is:

```text
L_total = alpha * L_CNP_explicit
        + beta  * L_CSR_retain
        + gamma * L_CNP_implicit
```

The main training loop is `LSSETrainer.train` in `methods/lsse_trainer.py`.

## Important Variants

| Variant | Flag | Meaning |
|---|---|---|
| Vanilla LSSE | none | `CNP + CSR + CLM` |
| PLU | `--use_plu` | Progressive Layer Unlocking: train 1 layer, then 3, then 6. |
| W2 / margin CNP | `--use_margin_cnp` | Remove concept projection while weakly anchoring the orthogonal component to prevent rerouting. |
| PLU+W2 | `--use_plu --use_margin_cnp` | Best LSSE variant in the current comparison. |
| W1 | `--use_tokensel_dir` | Compute concept direction from high-variance content token positions. |
| W3 | `--use_tokenwise_csr` | Tokenwise InfoNCE instead of mean-pooled InfoNCE. |
| W4 | `--use_membank` | MoCo-style retain memory bank. |
| W5 | `--use_dynamic_clm` | Re-rank trainable layers during training. |
| W6 | `--use_adaptive_weights` | Uncertainty-based adaptive loss weighting. |
| CAP-CNP | `--use_cap_cnp` | Erase concept in the **UNet cross-attn read-out space** `R = C·M^½` (`M = mean_ℓ WₖᵀWₖ+WᵥᵀWᵥ`, frozen UNet). TE-only (UNet not edited). |
| CAP dir | `--cap_dir_mode <svd\|contrastive\|contrastive_ortho\|whitened>` | Concept-direction estimator. `contrastive_ortho` (S2) = `mean(explicit)−mean(retain)` ⟂ retain span. |
| CAP metric | `--cap_metric_mode <kv\|v_only\|perlayer>` | Read-out metric: K+V (default), V-only, or per-layer (summed). |

## Latest Local Result

Latest unified comparison uses NudeNet v3, score threshold 0.3, 5 attacks x
50 images, 50 steps, guidance 7.5, seed 42. Lower ASR is better.

| Model | Main switches | Mean ASR |
|---|---|---:|
| Raw SD v1.4 | none | 62.0 |
| Vanilla LSSE | `CNP+CSR+CLM` | 46.0 |
| LSSE+PLU | `--use_plu` | 21.6 |
| LSSE+PLU+W2 | `--use_plu --use_margin_cnp` | 20.8 |
| Spherical+OT | text-encoder comparison point | 15.6 |
| ODACE v3 | UNet comparison point | 4.0 |

Interpretation:

- Vanilla LSSE is a clear improvement over FCF-P in the local harness.
- PLU is the dominant LSSE improvement; it drops ASR from 46.0 to about 21.
- W2/margin CNP gives a small additional gain and is the best LSSE stack.
- LSSE appears to hit a text-encoder-only floor around 20 ASR, which motivated
  the later DACE and ODACE investigations.

## CAP-CNP — breaking the LSSE floor (2026-06)

**Cross-Attention-Pullback CNP** removes the LSSE ~20-ASR floor by erasing the concept
in the space the UNet actually reads (`K=W_k C, V=W_v C`) instead of raw CLIP space.
Still text-encoder-only: `M^½` is a frozen-UNet constant, gradient flows to the TE only.
See `methods/xattn_metric.py` + `cap_dir_mode`/`cap_metric_mode` in `methods/lsse_trainer.py`.

The parameter-free direction/metric modes **move the Pareto frontier** (lower ASR at equal
utility), unlike λ/β which only slide along it. Numbers below are the **full-set** eval
(1622 prompts × 5 attacks; `ours8` = NudeNet 8-label nudity ASR, `fcf4` = paper 4-label rule)
+ 300-img COCO CLIP/FID:

| Variant (dir / metric) | ours8 ASR ↓ | fcf4 ASR ↓ | COCO CLIP ↑ | note |
|---|---:|---:|---:|---|
| baseline LSSE+PLU+W2 | 20.5 | 5.8 | 19.19 | previous best LSSE |
| Spherical+OT (ref) | 14.0 | 1.7 | 23.92 | previous best TE-only |
| **CAP-CNP — R2 `contrastive_ortho / perlayer` (flagship)** | **0.7** | **0.5** | 17.69 | **strongest forget in all of Table A** (beats ODACE v3 5.2/0.8); utility traded |
| CAP-CNP — S2 `contrastive_ortho / kv` | 19.5 | 2.8 | **22.04** | utility-side: CLIP +2.85 & fcf4 5.8→2.8 vs baseline, ours8 ≈ baseline |

> **Proxy → full-set correction.** An earlier N=10 proxy (Spearman 0.929) put S2 at ASR 10
> (≈ "beats sph_ot"); the **full set does not confirm this** — S2's `ours8` is **19.5**, on par
> with baseline (20.5) and worse than sph_ot (14.0). S2 is a genuine **utility** + paper-4label
> win, not a strict ASR win. The real Pareto mover is **R2 (zero)**: `ours8` 0.7 is the lowest
> nudity ASR in the whole comparison, at the cost of COCO CLIP (17.69). Lesson: validate
> direction/metric modes on the full set, not the proxy.

Mechanism: `contrastive_ortho` direction = forget−retain mean shift, Gram-Schmidt
orthogonalized against the retain span → erases only the concept-discriminative axis while
structurally preserving general content. The `kv` metric (S2) preserves utility but the
read-out pullback alone is not enough to drop `ours8`; the aggressive `perlayer` metric (R2)
sums the margin loss over every cross-attn layer → reaches `ours8` 0.7 (full block) but pays
COCO CLIP — the metric aggressiveness, not the direction, is what trades forget for utility.

## Run

From the repository root:

```bash
python lsse/train_lsse.py --config lsse/configs/nudity_lsse.yaml
```

Best LSSE variant:

```bash
python lsse/train_lsse.py \
  --config lsse/configs/nudity_lsse.yaml \
  --use_plu \
  --use_margin_cnp \
  --output_dir outputs/sweep/stack_plu_w2_seed42
```

Multi-seed sweep examples are defined in:

```text
lsse/experiments/configs/improvements.yaml
lsse/experiments/configs/validate.yaml
```

## Evaluate

From `models/lsse/`:

```bash
python evaluate.py \
  --encoder_dir outputs/sweep/stack_plu_w2_seed42/final \
  --concept nudity \
  --eval_type asr \
  --output_dir outputs/eval/xharness_lsse_plu_w2 \
  --num_images 50
```

## Tests

```bash
python -m pytest lsse/tests -q
```

## Relationship To Other Methods

LSSE removes two FCF design choices: the arbitrary noise prompt and the
two-stage training schedule. It is stronger than FCF in this project, but its
diagnostics also exposed concept rerouting: removing one text-embedding
direction does not guarantee robust image-level concept removal. That finding
led to DACE, and then to ODACE's output-grounded UNet edit.
