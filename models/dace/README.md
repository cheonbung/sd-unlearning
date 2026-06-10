# DACE - Dynamic Adversarial Concept Erasure

Last updated: 2026-06-05

`dace/` implements **DACE**, short for **Dynamic Adversarial Concept Erasure**.
It is an independent text-encoder unlearning experiment. It does not import
`fcf/`, `models/novel/`, or `lsse/`.

DACE edits only the **CLIP text encoder**. Its purpose in the project is as
important as its raw score: it tests whether dynamically suppressing a
text-embedding concept axis is enough to reduce image-level ASR.

## What Is Trained

```text
prompt -> CLIP text encoder -> text embedding -> SD UNet -> image
            ^ trained here
```

Model components:

| Component | Value |
|---|---|
| Text encoder | `openai/clip-vit-large-patch14` |
| SD evaluation base | `CompVis/stable-diffusion-v1-4` |
| Trainable module | CLIP text encoder |
| Frozen modules | SD UNet, VAE, frozen reference text encoder |
| Target concept used here | `nudity` |

## Motivation

LSSE diagnostics suggested a failure mode called **concept rerouting**:

```text
erase one fixed concept direction -> concept leaks into another direction
```

DACE tries to make rerouting unhelpful by recomputing the live concept subspace
during training and anchoring the non-concept part of the embedding.

## Corrected Concept Axis

Early probing rejected the naive "forget vs retain" axis because it was
lexically saturated. The corrected DACE axis is:

```text
d_i = pool(z_explicit_i) - pool(z_neutral_i)
```

where `z_neutral_i` is the same prompt with concept words stripped by
`core/dataset.py::neutralize`.

The live subspace `U` is the top-k SVD subspace of these concept-shift vectors.
It is recomputed every `adv_every` steps. See `methods/adversary.py`.

## Training Objective

```text
min_theta alpha * L_forget + gamma * L_ortho + beta * L_retain

U        = top-k SVD of live concept-shift vectors
L_forget = || U^T(pool(z_exp) - pool(z_neu)) ||^2
L_ortho  = keep the non-concept component of z_exp close to frozen
L_retain = preserve neutral and retain prompts against the frozen encoder
```

The loss implementation is in `methods/erasure.py`; the training loop is in
`core/trainer.py`.

## Latest Local Result

Latest unified comparison uses NudeNet v3, score threshold 0.3, 5 attacks x
50 images, 50 steps, guidance 7.5, seed 42. Lower ASR is better.

| Model | Intervention | Mean ASR |
|---|---|---:|
| Raw SD v1.4 | none | 62.0 |
| DACE v2 | text encoder, dynamic concept axis | 50.8 |
| DACE+PLU | text encoder, concept axis + progressive layer unlocking | 73.6 |
| LSSE+PLU+W2 | text-encoder comparison point | 20.8 |
| ODACE v3 | UNet comparison point | 4.0 |

Interpretation:

- DACE is a useful **negative result**: improving a text-embedding concept-axis
  proxy did not translate into strong image-level unlearning.
- DACE+PLU is worse than raw SD in the current ASR harness, showing that layer
  dynamics plus this concept-axis loss can amplify the wrong behavior.
- This result is one of the reasons the project moved from text-embedding proxy
  objectives to ODACE's output-grounded UNet objective.

## Run

From the repository root:

```bash
python dace/train_dace.py --config configs/nudity_dace.yaml
```

PLU variant:

```bash
python dace/train_dace.py \
  --config configs/nudity_dace_plu.yaml \
  --output_dir outputs/dace_plu_nudity
```

Outputs:

| Variant | Output directory |
|---|---|
| DACE | `dace/outputs/dace_nudity/final` |
| DACE+PLU | `dace/outputs/dace_plu_nudity/final` if using the command above |

## Evaluate With The Unified Harness

```bash
cd lsse
python evaluate.py \
  --encoder_dir ../dace/outputs/dace_nudity/final \
  --concept nudity \
  --eval_type asr \
  --output_dir outputs/eval/xharness_dace \
  --num_images 50
```

Use `../dace/outputs/dace_plu_nudity/final` for DACE+PLU.

## Diagnostics

```bash
cd dace
python experiments/p0_crossmethod_diag.py
python experiments/p0b_concept_axis.py
```

`P0` rejected the forget-vs-retain separability axis. `P0b` supported the
explicit-vs-neutral concept axis as a better text proxy, but later training
showed that even this proxy underdetermines final image ASR.

## Tests

```bash
python -m pytest dace/tests -q
```

## Relationship To Other Methods

DACE sits between LSSE and ODACE in the research story. It takes LSSE's
"rerouting" diagnosis seriously and adds a dynamic concept-axis adversary, but
still edits only the text encoder. Its weak ASR result supports the conclusion
that image-level concept erasure cannot be reliably inferred from text-embedding
metrics alone.
