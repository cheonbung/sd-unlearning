# ODACE - Output-Distribution Adversarial Concept Erasure

Last updated: 2026-06-05

`models/odace/` implements **ODACE**, short for **Output-Distribution Adversarial
Concept Erasure**. In code comments this is also described as an
**output-grounded** concept erasure method.

ODACE is the first method in this project that edits the **Stable Diffusion
UNet** rather than the CLIP text encoder. It targets the cross-attention path
that turns text conditioning into image denoising behavior.

## What Is Trained

```text
prompt -> CLIP text encoder -> text embedding -> SD UNet -> image
                                               ^ trained here
```

Model components:

| Component | ODACE v3 | ODACE v1.5 |
|---|---|---|
| SD base | `CompVis/stable-diffusion-v1-4` | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Text encoder | frozen | frozen |
| Trainable module | UNet cross-attention | UNet cross-attention |
| Default trained projections | full `to_q`, `to_k`, `to_v`, `to_out` (`xattn_full: true`) | same |
| Target concept used here | `nudity` | `nudity` |

Earlier ODACE v2 trained only cross-attention K/V with a weaker recipe. It moved
the weights too little and left ASR near raw SD. The current v3 recipe uses full
cross-attention, higher learning rate, and stronger negative guidance.

## Core Idea

Text-encoder methods optimize a proxy: "does the text embedding look erased?"
ODACE optimizes the quantity that actually drives the generated image:

```text
UNet noise prediction at a diffusion timestep
```

During training, ODACE samples a latent along a frozen-model denoising trajectory
conditioned on a forget prompt. Then the trainable UNet is asked to predict a
noise direction that moves away from the concept.

Current trainer target:

```text
e_0    = frozen UNet prediction with unconditional conditioning
e_p    = frozen UNet prediction with forget prompt conditioning
target = e_0 - eta * (e_p - e_0)

L_forget = MSE(e_forget_current, target)
L_retain = MSE(e_retain_current, e_retain_frozen)
L_total  = alpha * L_forget + beta * L_retain
```

The target is computed in `core/trainer.py`. Trainable cross-attention modules
are selected in `methods/unet_edit.py`.

## Latest Local Result

Full-set comparison (1622 prompts x 5 attacks; NudeNet v3, `fcf4` = paper
4-label rule, `ours8` = project 8-label rule). Report ASR paired with
**Ring-A-Bell coherence** (CLIP person-presence probability; ≥0.7 = the model
still generates people under attack, i.e. no collapse):

| Model | Intervention | ASR 4-lab | ASR 8-lab | Ring-A-Bell coherence | COCO-CLIP |
|---|---|---:|---:|---:|---:|
| **ODACE benign-neg** (`odace_benign_n1`) | UNet cross-attn, redirect-to-benign | **2.1** | 15.7 | 1.00 (coherent) | 25.43 |
| ODACE benign-anchor (`odace_benign`) | UNet cross-attn, redirect-to-benign | 4.4 | 17.4 | 0.99 (coherent) | 25.62 |
| Raw SD v1.4 | none | ~50.4 | 64.9 | 0.97 | 26.48 |
| SD2.1-base | no unlearning, filtered pretraining | — | 54.4 | — | 25.95 |

`odace_benign` / `odace_benign_n1` are the **current gallery-surviving ODACE
points** — they redirect the UNet cross-attn target toward a benign latent
instead of pushing away from the concept, which keeps generation coherent
under adversarial attack. Configs: `configs/nudity_odace_benign.yaml`,
`configs/nudity_odace_benign_n1.yaml`.

### Excluded: ODACE v2 / v3 / v1.5 (OOD generation collapse)

| Model | Base | Intervention | Mean ASR (old 8-lab) | Ring-A-Bell coherence |
|---|---|---|---:|---:|
| ODACE v2 | v1.4 | UNet K/V only, weak recipe | 56.0 | — |
| ODACE v3 | v1.4 | full UNet cross-attn, negative-guidance | 4.0 | **0.12 (collapsed)** |
| ODACE v1.5 | v1.5 | full UNet cross-attn, negative-guidance | 4.0 | not separately probed (same recipe as v3) |

ODACE v3 per-attack ASR (original reading, pre-collapse-discovery):

| Attack | ASR |
|---|---:|
| I2P | 8.0 |
| Ring-A-Bell | 0.0 |
| Ring-A-Bell(Re) | 0.0 |
| P4D | 2.0 |
| UnlearnDiffAtk | 10.0 |

**This table was experimented on but is excluded from current results and
recommendations.** The negative-guidance recipe (`eta * (e_p - e_0)` pushing
*away* from the concept — see Core Idea below) turned out to collapse image
generation entirely on Ring-A-Bell prompts (CLIP person-presence probability
0.12, vs ≥0.7 for a coherent model): the **Ring-A-Bell / Ring-A-Bell(Re) 0.0
ASR above is not real erasure** — the model simply stopped generating people,
so NudeNet had nothing to detect. COCO-FID/CLIP still looked normal because
that eval uses non-adversarial COCO captions, which don't trigger the
collapse. This is why the project moved to the **benign-anchor / benign-neg**
redirect-to-benign variants above. `odace_v2` is unaffected by this finding
(it under-edits rather than collapses) and remains excluded for the separate
reason of being too weak.

## Run

From the repository root, current gallery-surviving (non-collapsed) variants:

```bash
python odace/train_odace.py --config configs/nudity_odace_benign.yaml
python odace/train_odace.py --config configs/nudity_odace_benign_n1.yaml
```

The original negative-guidance recipe (excluded, see above):

```bash
python odace/train_odace.py --config configs/nudity_odace.yaml
```

For the SD v1.5 transfer check (same excluded recipe on the v1.5 base):

```bash
python odace/train_odace.py --config configs/nudity_odace_v15.yaml
```

Main config values for v3:

| Parameter | Value |
|---|---:|
| `learning_rate` | `1.0e-4` |
| `num_steps` | `1500` |
| `eta` | `3.0` |
| `ddim_steps` | `30` |
| `sample_guidance` | `3.0` |
| `xattn_full` | `true` |

Outputs:

| Variant | Output directory |
|---|---|
| ODACE v3 | `models/odace/outputs/odace_nudity/final` |
| ODACE v1.5 | `models/odace/outputs/odace_v15/final` |

## Evaluate

ASR for the v1.4 ODACE checkpoint:

```bash
cd odace
python evaluate_odace.py \
  --unet_dir outputs/odace_nudity/final \
  --output_dir outputs/eval/odace_v3 \
  --num_images 50
```

Retain utility without COCO:

```bash
cd odace
python evaluate_utility.py \
  --unet_dir outputs/odace_nudity/final \
  --output_dir outputs/eval/odace_v3
```

For cross-base comparisons, including ODACE v1.5 and SD2.1-base, use
`eval/xeval.py` and `eval/eval_coco.py`, because `evaluate_odace.py`
currently hardcodes the SD v1.4 base.

```bash
cd xmodel
python xeval.py --models odace_v3,odace_v15,sd21base
python eval_coco.py --models odace_v3,odace_v15,sd21base
```

## Tests

```bash
python -m pytest odace/tests -q
```

## Relationship To Other Methods

ODACE is the answer to the main limitation found in FCF, LSSE, Spherical+OT,
and DACE: text-embedding objectives do not fully determine image-level ASR.
ODACE trains the UNet output behavior directly, at the cross-attention point
where text conditioning becomes denoising behavior. That change is what breaks
the text-encoder ASR floor in the current experiments.
