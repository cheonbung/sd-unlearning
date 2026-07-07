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
| CAP metric | `--cap_metric_mode <kv\|v_only\|perlayer\|perlayer_causal\|perlayer_topk>` | Read-out metric: K+V (default), V-only, per-layer (uniform sum), or per-layer weighted by concept-causality `w_ℓ=‖(μ_e−μ_r)@M_ℓ½‖²` (causal=continuous, topk=90%-energy subset). |
| CAP retain | `--cap_retain_anchor` | **A**: pin retain in the read-out metric `mean_ℓ‖(z−z_frozen)@M_ℓ½‖²` (the R2-quality utility fix). |
| CAP loss | `--cap_loss_mode <margin\|project>` | `margin` (proj²+ortho anchor, default) or `project` (exact projection target; rejected — under-erases). |

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
| ODACE v3 | UNet comparison point, ⚠ later found to collapse on Ring-A-Bell (see below) | 4.0 |

Interpretation:

- Vanilla LSSE is a clear improvement over FCF-P in the local harness.
- PLU is the dominant LSSE improvement; it drops ASR from 46.0 to about 21.
- W2/margin CNP gives a small additional gain and is the best LSSE stack.
- LSSE appears to hit a text-encoder-only floor around 20 ASR, which motivated
  the later DACE and ODACE investigations.

## CAP-CNP — breaking the LSSE floor (2026-06) ⚠ excluded from current results

> **2026-07 correction: the R2/R2q-ab/R2q-a numbers below are OOD-collapsed.**
> On Ring-A-Bell adversarial prompts these variants stop generating people
> altogether (CLIP person-presence probability ~0.15 for R2q-ab, ~0.01 for
> R2q-a, vs ≥0.7 for a coherent model) — their low ASR is a broken-output
> artifact, not real erasure. The method description below is still accurate
> (this is genuinely how read-out-space erasure works), but the "dominates
> Sph+OT / beats ODACE" ranking claims do not hold once coherence is checked.
> **The current gallery-surviving LSSE point is `lsse_geo_e2`** (geodesic /
> N5 RG-FCF spherical projection, `configs/nudity_lsse_geo_e2.yaml`) at
> 4-label ASR 2.1, COCO-CLIP 25.66, Ring-A-Bell coherence 0.58 — no collapse.
> This CAP-CNP section is kept for the mechanism explanation and as a
> documented negative result; see "Excluded experiments" at the bottom.

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
| CAP-CNP — R2 `contrastive_ortho / perlayer` | **0.7** | **0.5** | 17.69 | strongest forget, but utility collapses (CLIP 17.69) |
| CAP-CNP — S2 `contrastive_ortho / kv` | 19.5 | 2.8 | 22.04 | utility-side; ours8 ≈ baseline |
| **CAP-CNP — R2q-ab `+retain-anchor +perlayer_causal` (flagship)** | 3.1 | **0.5** | **23.62** | **dominates S2 AND sph_ot on both axes; lower ASR than ODACE v3 (5.2)** |
| CAP-CNP — R2q-a `+retain-anchor` | **1.3** | 1.6 | 20.13 | max-forget; utility recovered above baseline |

> **Proxy → full-set correction.** An earlier N=10 proxy (Spearman 0.929) put S2 at ASR 10
> (≈ "beats sph_ot"); the **full set does not confirm this** — S2's `ours8` is **19.5**, on par
> with baseline (20.5). Lesson: validate direction/metric modes on the full set, not the proxy.
>
> **R2-quality (2026-06-21): R2's utility collapse fixed.** R2 erases in read-out space but retain
> was protected only in raw CLIP space → general content distorted (D1). **A = read-out retain
> anchor** (`cap_retain_anchor`, `mean_ℓ‖(z−z_frozen)@M_ℓ½‖²`) pins retain in the SAME space the
> erasure acts on. Adding **B = causal per-layer weighting** (`perlayer_causal`,
> `w_ℓ=‖(μ_e−μ_r)@M_ℓ½‖²`) gives **R2q-ab (flagship)**: ours8 3.1 / fcf4 0.5 / CLIP 23.62 —
> **dominates both S2 and sph_ot on both axes** and undercuts ODACE v3's ASR (5.2): the strongest
> TE-only point in the project. (C = projection target `cap_loss_mode=project` under-erases →
> ASR 43–50, rejected.) Config: `configs/nudity_lsse_capcnp_r2q.yaml`.

Mechanism: `contrastive_ortho` direction = forget−retain mean shift, Gram-Schmidt
orthogonalized against the retain span → erases only the concept-discriminative axis. The
`perlayer` metric drops `ours8` (R2), but because erasure happens in read-out space, utility
survives only if retain is **anchored in that same space** (R2q): forcing the erasure operator to
be identity on the retain subspace — not raw CLIP — is the decisive utility fix.

**Violence transfer (2026-06-22).** The R2q-ab recipe applied to violence
(`configs/violence_lsse_capcnp_r2q.yaml`, key `lsse_r2q_violence`, eval = Q16) cuts violence ASR
**66.9 → 16.7** (I2P 19.3 / Ring-A-Bell 14.1) — 4th of 25 models, beating every dedicated baseline
(ODACE v3/v15 59.1, ESD-u 62.7, FCF-P 24.4, SLD, Safe-CLIP); the only lower models are nudity-trained
ones with incidental violence suppression. So the read-out-space erasure + retain anchor **transfers**.
Caveat: COCO-CLIP **18.27** (vs nudity 23.62) — violence's read-out projection makes `L_cnp` ~10⁴×
larger than nudity, so the fixed-β retain anchor is under-weighted and utility recovers less. This is a
loss-scale issue, not a method failure.

> **W6 adaptive-weighting did NOT fix it (2026-06-22, `configs/violence_lsse_capcnp_r2q_aw.yaml`, key
> `lsse_r2q_violence_aw`).** `--use_adaptive_weights` (Kendall uncertainty) moved the operating point the
> **wrong way** on utility: Q16 ASR 16.7 → **5.4** (even stronger forget) but COCO-CLIP 18.27 → **14.13**
> (FID 135.8 → 194.1, near broken-model territory). Uncertainty weighting rebalances by loss *magnitude*,
> but the violence retain anchor is itself inflated, so AW suppressed retain harder instead of protecting
> it. The magnitude imbalance ≠ importance imbalance, so adaptive weighting is the wrong tool here. The
> remaining viable fix is to **normalize the read-out projection** (rescale `M_ℓ½` so violence `L_cnp`
> matches the nudity scale), a deterministic loss-scale fix — not run yet. **Non-AW violence R2q
> (16.7 / 18.27) remains the better-balanced violence operating point** and is the documented flagship.

## Excluded experiments (OOD generation collapse)

The following LSSE variants were trained and evaluated but are **excluded from
current results** — their low ASR turned out to be OOD generation collapse on
Ring-A-Bell (CLIP person-presence probability < 0.3), not genuine erasure:

- **CAP-CNP `R2q-ab` / `R2q-a` / `R2 perlayer` / `capcnp_zero`** (nudity) — see
  correction note above.
- **`lsse_r2q_violence` / `lsse_r2q_violence_aw`** (violence transfer) — same
  read-out over-erasure failure mode applied to the violence concept.

The current recommended, non-collapsed LSSE point is **`lsse_geo_e2`**
(`configs/nudity_lsse_geo_e2.yaml`, N5 RG-FCF spherical/geodesic projection) —
see the correction note in the CAP-CNP section above for its numbers.

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
