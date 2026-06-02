# SD Unlearning

Research code for concept unlearning in Stable Diffusion through CLIP text
encoder fine-tuning. The repository groups three related experiment tracks:

- `fcf/`: baseline Fortified Concept Forgetting (FCF) reproduction.
- `fcf-novel-methods/`: independent FCF extensions for manifold projection,
  optimal-transport noise prompts, and causal activation analysis.
- `lsse/`: Layer-Selective Semantic Erasure (LSSE), a single-loop alternative
  with null-space forgetting, contrastive retention, and layer masking.

The generated checkpoints, images, logs, external model weights, and other
large artifacts are intentionally ignored by Git. Re-run training/evaluation to
recreate them under each subproject's `outputs/` directory.

## Repository Layout

```text
.
|-- fcf/                  # FCF-P / FCF-E baseline implementation
|   |-- configs/          # Nudity, violence, and Van Gogh configs
|   |-- core/             # Dataset, noise, and FCF trainer logic
|   |-- data/             # Training and evaluation prompt files
|   |-- evaluation/       # ASR, LPIPS, FID, CLIP-score evaluators
|   |-- analysis/         # Ablation, sweeps, and training-curve scripts
|   |-- scripts/          # Data preparation and attack helper scripts
|   `-- tests/            # Unit tests for the baseline components
|-- fcf-novel-methods/    # Isolated experimental FCF extension project
|   |-- core/             # Local copy of the FCF base components
|   |-- methods/          # N1 CAP, N5 spherical projection, N6 OT noise
|   |-- configs/          # Novel-method nudity configs
|   |-- scripts/          # CAP and OT-noise CLIs
|   `-- tests/            # Unit tests for the novel methods
`-- lsse/                 # Layer-Selective Semantic Erasure experiments
    |-- core/             # LSSE dataset
    |-- methods/          # CNP, CSR, CLM, and LSSE trainer
    |-- configs/          # LSSE nudity config
    `-- tests/            # Unit tests for LSSE
```

## Methods

### FCF Baseline

`fcf/` implements the two-stage method from:

> Fan et al., "Fortified Concept Forgetting for text-to-image generative models
> by machine unlearning on CLIP", Computer Standards & Interfaces 97 (2026)
> 104142.

The baseline trains only the CLIP text encoder used by Stable Diffusion:

1. Stage 1: explicit concept forgetting by moving explicit prompts toward
   random noise-prompt embeddings.
2. Stage 2: implicit concept forgetting with either:
   - `fcf_p`: projection feature forgetting.
   - `fcf_e`: empirical feature forgetting.

### Novel FCF Extensions

`fcf-novel-methods/` is intentionally independent from `fcf/`. It keeps a local
copy of the base trainer and adds:

- N1 CAP: causal activation patching to score concept-sensitive CLIP layers.
- N5 RG-FCF: spherical/Riemannian geodesic projection instead of Euclidean
  projection.
- N6 OT-FCF: learned 5-character noise prompts selected by Wasserstein distance
  in CLIP embedding space.

`manifold="euclidean"` preserves the baseline projection behavior;
`manifold="spherical"` enables the N5 variant.

### LSSE

`lsse/` implements Layer-Selective Semantic Erasure:

- N7 CNP: Concept Null-Space Projection.
- N8 CSR: Contrastive Semantic Retention.
- N9 CLM: CAP-guided layer masking.

The trainer also exposes experimental switches including DDF, MACD, PLU,
multi-direction CNP, and layer-dependent learning rates.

## Setup

Use a CUDA-capable Python environment for realistic training and image
generation. CPU execution is possible for small tests but impractical for full
Stable Diffusion runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r fcf/requirements.txt
python -m pip install -r fcf-novel-methods/requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r fcf\requirements.txt
python -m pip install -r fcf-novel-methods\requirements.txt
```

Alternatively, create the conda environment described by the repository:

```bash
conda env create -f environment.yml
conda activate fcf
```

The code downloads Hugging Face models on first use:

- Stable Diffusion: `CompVis/stable-diffusion-v1-4`
- CLIP text encoder: `openai/clip-vit-large-patch14`

Depending on your Hugging Face setup, you may need to authenticate before
loading Stable Diffusion weights.

## Quality Gate

Run the source validation gate before committing experiment changes:

```bash
python scripts/quality_gate.py
```

The gate compiles the owned source directories and runs all unit tests:

- `fcf/tests`
- `fcf-novel-methods/tests`
- `lsse/tests`

It also scans source/config files for local absolute paths such as
`C:/Users/...`, which would break reproduction on another machine. Generated
outputs and the local `fcf/Q16/` external checkout are skipped.

## Quick Start

Run commands from the repository root unless noted otherwise.

### Train FCF-P

```bash
python fcf/train.py --config fcf/configs/nudity_fcf_p.yaml
```

### Train FCF-E

```bash
python fcf/train.py --config fcf/configs/nudity_fcf_e.yaml
```

Other baseline configs are available for violence and Van Gogh style:

```bash
python fcf/train.py --config fcf/configs/violence_fcf_p.yaml
python fcf/train.py --config fcf/configs/vangogh_fcf_p.yaml
```

Common FCF overrides:

```bash
python fcf/train.py \
  --config fcf/configs/nudity_fcf_p.yaml \
  --eta 0.3 \
  --mu_p 0.8 \
  --num_epochs 10 \
  --device cuda
```

### Train Novel FCF

Spherical projection:

```bash
python fcf-novel-methods/train.py \
  --config fcf-novel-methods/configs/nudity_v2.yaml \
  --manifold spherical
```

Euclidean compatibility check:

```bash
python fcf-novel-methods/train.py \
  --config fcf-novel-methods/configs/nudity_v2.yaml \
  --manifold euclidean
```

Learn an OT-noise vocabulary, then train with it:

```bash
python fcf-novel-methods/scripts/learn_ot_noise.py \
  --explicit_file fcf-novel-methods/data/prompts/nudity_explicit.txt \
  --output fcf-novel-methods/outputs/ot_noise/nudity_learned.json

python fcf-novel-methods/train.py \
  --config fcf-novel-methods/configs/nudity_v2.yaml \
  --ot_noise_file fcf-novel-methods/outputs/ot_noise/nudity_learned.json
```

Run CAP layer analysis:

```bash
python fcf-novel-methods/scripts/run_cap_analysis.py \
  --concept nudity \
  --explicit_file fcf-novel-methods/data/prompts/nudity_explicit.txt \
  --output fcf-novel-methods/outputs/cap/nudity_heatmap.json
```

### Train LSSE

```bash
python lsse/train_lsse.py --config lsse/configs/nudity_lsse.yaml
```

Useful LSSE switches:

```bash
python lsse/train_lsse.py \
  --config lsse/configs/nudity_lsse.yaml \
  --use_extended_csr \
  --use_ddf \
  --use_plu \
  --clm_top_k 3
```

Use CAP scores with LSSE:

```bash
python lsse/train_lsse.py \
  --config lsse/configs/nudity_lsse.yaml \
  --cap_file lsse/outputs/cap/nudity_heatmap.json
```

## Image Generation

Generate images with a fine-tuned Hugging Face encoder directory:

```bash
python fcf/generate_images.py \
  --encoder_dir fcf/outputs/fcf_p_nudity/final \
  --prompts_file fcf/data/eval/i2p_nudity.txt \
  --output_dir fcf/outputs/images/fcf_p_nudity_i2p
```

Generate a Stable Diffusion baseline with no unlearned encoder:

```bash
python fcf/generate_images.py \
  --prompts_file fcf/data/eval/i2p_nudity.txt \
  --output_dir fcf/outputs/images/sd_baseline_i2p
```

The generator also supports the original CSV-style interface with
`--prompts_path`, `--model_path`, `--save_path`, and `--model_name`.

## Evaluation

ASR evaluation for nudity:

```bash
python fcf/evaluate.py \
  --config fcf/configs/nudity_fcf_p.yaml \
  --encoder_dir fcf/outputs/fcf_p_nudity/final \
  --concept nudity \
  --eval_type asr
```

Quality evaluation:

```bash
python fcf/evaluate.py \
  --config fcf/configs/nudity_fcf_p.yaml \
  --encoder_dir fcf/outputs/fcf_p_nudity/final \
  --concept nudity \
  --eval_type quality
```

Style evaluation for Van Gogh forgetting:

```bash
python fcf/evaluate.py \
  --config fcf/configs/vangogh_fcf_p.yaml \
  --encoder_dir fcf/outputs/fcf_p_vangogh/final \
  --concept vangogh \
  --eval_type style \
  --baseline_dir fcf/outputs/images/sd_baseline_vangogh_style \
  --baseline_other_dir fcf/outputs/images/sd_baseline_vangogh_other
```

`evaluate.py` writes `eval_results.json` under the selected `--output_dir`.

## Tests

Run the focused unit tests from each subproject:

```bash
python -m pytest fcf/tests -q
python -m pytest fcf-novel-methods/tests -q
python -m pytest lsse/tests -q
```

The tests use small mocked components where possible. Full training,
generation, and evaluator runs require the model dependencies and enough GPU
memory for Stable Diffusion.

## Data

The repository includes prompt files under each subproject's `data/` tree:

- `data/prompts/`: explicit concept prompts, implicit concept names, retain
  prompts, and explicit concept aliases.
- `data/eval/`: I2P, Ring-A-Bell, P4D, UnlearnDiffAtk, and style prompt files.

The evaluation README files describe the expected attack prompt names and
sources. Some scripts under `fcf/scripts/` help download or prepare red-team
prompt data.

## Outputs and Version Control

The root `.gitignore` excludes:

- `outputs/` directories.
- model/checkpoint formats such as `.pt`, `.pth`, `.ckpt`, `.safetensors`,
  `.bin`, and `.onnx`.
- Python caches, node dependencies, local tool state, logs, archives, PDFs, and
  pickle files.
- `fcf/Q16/`, which is an embedded external repository used locally for Q16
  classifier experiments.

This keeps the GitHub repository source-focused. Store large training artifacts
separately if they need to be shared.

## Results

Measured experiment logs are kept in:

- `fcf/docs/results.md`
- `fcf-novel-methods/docs/results.md`
- `lsse/docs/results.md`

Those files record ASR, LPIPS, CLIP score, and FID measurements for prior runs.
Several older Korean notes in the repository have mojibake/encoding artifacts;
prefer code, configs, and numeric tables as the source of truth when updating
documentation.

## Notes

- All three tracks fine-tune the CLIP text encoder, not the Stable Diffusion
  UNet or VAE.
- Default training hyperparameters mirror the FCF paper where applicable:
  learning rate `2.5e-5`, `eta=0.25`, `mu_p=0.7`, `mu_e=1.0`, `num_epochs=60`.
- `fcf-novel-methods/` and `lsse/` are independent experiments; they should not
  import from `fcf/` during normal use.
