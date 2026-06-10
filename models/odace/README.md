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

Latest unified comparison uses NudeNet v3, score threshold 0.3, 5 attacks x
50 images, 50 steps, guidance 7.5, seed 42. Lower ASR is better.

| Model | Base | Intervention | Mean ASR | COCO-FID | COCO-CLIP |
|---|---|---|---:|---:|---:|
| Raw SD v1.4 | v1.4 | none | 62.0 | 118.64 | 26.48 |
| ODACE v2 | v1.4 | UNet K/V only, weak recipe | 56.0 | - | - |
| ODACE v3 | v1.4 | full UNet cross-attn | 4.0 | 118.89 | 25.27 |
| ODACE v1.5 | v1.5 | full UNet cross-attn | 4.0 | 117.90 | 25.23 |
| SD2.1-base | v2.1 | no unlearning, filtered pretraining | 54.4 | 118.32 | 25.95 |

ODACE v3 per-attack ASR:

| Attack | ASR |
|---|---:|
| I2P | 8.0 |
| Ring-A-Bell | 0.0 |
| Ring-A-Bell(Re) | 0.0 |
| P4D | 2.0 |
| UnlearnDiffAtk | 10.0 |

Interpretation:

- ODACE v3 and ODACE v1.5 are the strongest models in the project so far.
- COCO-FID/CLIP shows that general non-nudity generation remains close to raw SD.
- SD2.1-base has normal general quality but remains vulnerable to adversarial
  nudity prompts, so filtered pretraining is not a robust unlearning substitute.

## Run

From the repository root:

```bash
python odace/train_odace.py --config configs/nudity_odace.yaml
```

For the SD v1.5 transfer check:

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
