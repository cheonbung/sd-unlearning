# FCF - Fortified Concept Forgetting

Last updated: 2026-06-05

`fcf/` is the reproduction baseline for **FCF**, short for **Fortified Concept
Forgetting**. It follows Fan et al., "Fortified Concept Forgetting for
text-to-image generative models by machine unlearning on CLIP".

In this project FCF edits only the **CLIP text encoder** used by Stable
Diffusion v1.4. The UNet and VAE are not trained. Evaluation then plugs the
edited text encoder back into Stable Diffusion and checks whether attack prompts
still generate the target concept.

## What Is Trained

```text
prompt -> CLIP text encoder -> text embedding -> SD UNet -> image
            ^ trained here
```

Model components:

| Component | Value |
|---|---|
| SD base | `CompVis/stable-diffusion-v1-4` |
| Text encoder | `openai/clip-vit-large-patch14` |
| Trainable module | CLIP text encoder |
| Frozen modules | SD UNet, VAE, frozen reference text encoder |
| Target concept used here | `nudity` |

## Method Summary

FCF is a two-stage text-encoder unlearning method.

### Stage 1 - Explicit Concept Forgetting

Explicit prompts containing the target concept are moved toward paired noise
prompts, while retain prompts are kept close to the original frozen encoder.

```text
L_total  = L_retain + eta * L_forget
L_forget = MSE(T_current(prompt_f), T_frozen(prompt_n))
L_retain = MSE(T_current(prompt_r), T_frozen(prompt_r))
```

Here `prompt_f` is a forget prompt, `prompt_n` is a noise prompt, and
`prompt_r` is a retain prompt. See `FCFTrainer.train_explicit` in
`core/trainer.py`.

### Stage 2 - Implicit Concept Forgetting

The second stage handles implicit concepts such as `male`, `boy`, `man`,
`female`, `girl`, `woman`. The project supports two variants.

| Variant | Full name | Core idea |
|---|---|---|
| `FCF-P` | Projection Feature Forgetting | Estimate a concept direction from explicit concept texts and subtract its projection from implicit concept embeddings. |
| `FCF-E` | Empirical Feature Forgetting | Compute an experience vector from explicit-minus-noise embeddings and subtract it from implicit concept embeddings. |

Code entry points:

| Function | Role |
|---|---|
| `FCFTrainer.train_explicit` | Stage 1 explicit forgetting |
| `FCFTrainer.train_projection_implicit` | Stage 2 `FCF-P` |
| `FCFTrainer.train_empirical_implicit` | Stage 2 `FCF-E` |
| `FCFTrainer.compute_experience` | `FCF-E` experience vector |

## Latest Local Result

Latest unified comparison uses the shared 5-attack ASR harness:
NudeNet v3, score threshold 0.3, 5 attacks x 50 images, 50 DDIM steps,
guidance 7.5, seed 42. Lower ASR is better.

| Model | Intervention | Mean ASR |
|---|---|---:|
| Raw SD v1.4 | none | 62.0 |
| FCF-P | text encoder | 52.8 |
| FCF-E | text encoder | 61.2 |

> ⚠️ **Correction (2026-06):** the `FCF-P 52.8 / FCF-E 61.2` above are from our
> **unfaithful `fcf/` reimplementation** (concept word-lists instead of the
> paper's 25 sentence triplets, and a missing `/(1-eta)` projection
> normalization). Re-running the **authors' official code + data** reproduces
> FCF-P to a paper-aligned full-set 4-label ASR of **3.7 (paper 3.43)**, 8-label
> 16.9 — below ESD-u (21.6). FCF reproduces; the 52.8 figure is a reimpl
> artifact, not the method. See `compare/comparison_all_methods.md` section
> 3-correction and `compare/fcf_repro/fullset_eval.json`.

Interpretation:

- `FCF-P` improves over raw SD in this harsher local harness, but remains far
  above later methods such as Spherical+OT, LSSE+PLU+W2, and ODACE.
- `FCF-E` is close to raw SD in the local harness and is weaker than `FCF-P`.
- The original paper reports much lower ASR under its own protocol; do not mix
  those numbers with this project's 5-attack x 50-image harness.

See `compare/comparison_all_methods.md` for the current cross-method table.

## Run

From the repository root:

```bash
python fcf/train.py --config fcf/configs/nudity_fcf_p.yaml
python fcf/train.py --config fcf/configs/nudity_fcf_e.yaml
```

Useful overrides:

```bash
python fcf/train.py \
  --config fcf/configs/nudity_fcf_p.yaml \
  --eta 0.3 \
  --mu_p 0.8 \
  --num_epochs 60 \
  --device cuda
```

Outputs:

| Variant | Output directory |
|---|---|
| FCF-P | `fcf/outputs/fcf_p_nudity/final` |
| FCF-E | `fcf/outputs/fcf_e_nudity/final` |

## Evaluate With The Unified Harness

The latest comparison evaluates FCF checkpoints through the shared LSSE harness
so that FCF, LSSE, DACE, and text-encoder variants use the same image generation
and NudeNet scoring code.

```bash
cd lsse
python evaluate.py \
  --encoder_dir ../fcf/outputs/fcf_p_nudity/final \
  --concept nudity \
  --eval_type asr \
  --output_dir outputs/eval/xharness_fcf_p \
  --num_images 50
```

Use `../fcf/outputs/fcf_e_nudity/final` for FCF-E.

## Tests

```bash
python -m pytest fcf/tests -q
```

## Relationship To Later Methods

FCF is the baseline that later project tracks react to:

- Spherical+OT keeps the FCF-P structure but changes the projection geometry and
  the noise prompt selection.
- LSSE removes the arbitrary noise-prompt target and replaces it with SVD-based
  null-space erasure plus contrastive retention.
- DACE shows that text-embedding proxy metrics can be misleading for ASR.
- ODACE moves the intervention point from the text encoder to UNet
  cross-attention, which is where the current best ASR is obtained.
