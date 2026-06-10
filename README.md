# SD Unlearning

Research code for **concept unlearning in Stable Diffusion**, comparing erasure
methods across different *intervention points* — the CLIP text encoder, the SD
UNet cross-attention, and inference-time guidance — under one shared evaluation
protocol. The project grew from a single FCF reproduction into a multi-method
comparison whose headline finding is that **where** you intervene matters more
than how much you change: output-grounded UNet editing (ODACE) erases far more
robustly than text-encoder proxies.

The repository groups several experiment tracks plus a cross-model harness:

- `fcf/`: baseline Fortified Concept Forgetting (FCF) reproduction (text encoder).
- `models/novel/`: independent FCF extensions (manifold projection,
  optimal-transport noise prompts, causal activation analysis).
- `lsse/`: Layer-Selective Semantic Erasure (LSSE), a single-loop text-encoder
  alternative with null-space forgetting, contrastive retention, layer masking.
- `dace/`: Dynamic Adversarial Concept Erasure — a text-encoder concept-axis
  experiment (a deliberate negative-result probe).
- `odace/`: Output-Distribution Adversarial Concept Erasure — the first method
  here that edits the **UNet cross-attention** instead of the text encoder.
- `models/`: same-protocol reproductions of ESD, SLD, and Safe-CLIP.
- `eval/`: cross-model ASR + COCO FID/CLIP evaluation harness.
- `compare/`: the unified comparison report and the official-code FCF reproduction.

Generated checkpoints, images, logs, external model weights, and other large
artifacts are intentionally Git-ignored. Re-run training/evaluation to recreate
them under each subproject's `outputs/` directory.

## Repository Layout

```text
.
|-- fcf/                  # FCF-P / FCF-E baseline (CLIP text encoder)
|   |-- configs/ core/ data/ evaluation/ analysis/ scripts/ tests/
|-- models/novel/    # Isolated FCF extensions (N1 CAP, N5 spherical, N6 OT)
|   |-- core/ methods/ configs/ scripts/ tests/
|-- lsse/                 # Layer-Selective Semantic Erasure (N7-N9)
|   |-- core/ methods/ configs/ evaluation/ tests/
|-- dace/                 # Dynamic Adversarial Concept Erasure (text encoder)
|   |-- methods/ configs/ core/ experiments/ tests/ train_dace.py
|-- odace/                # Output-Distribution Adversarial Concept Erasure (UNet)
|   |-- methods/ configs/ core/ experiments/ tests/ train_odace.py
|   |-- evaluate_odace.py evaluate_utility.py
|-- models/            # Same-protocol reproductions
|   |-- esd/ sld/ safeclip/ run_eval.sh
|-- eval/               # Cross-model eval harness
|   |-- xeval.py          # ASR over 5 attack suites for any registered model
|   |-- eval_coco.py      # COCO FID / CLIP / LPIPS / CLIP-IQA locality
|   `-- build_xgallery.py
|-- compare/              # Unified comparison + official FCF reproduction
|   |-- comparison_all_methods.md   # <-- authoritative results table
|   |-- build_gallery.py
|   `-- fcf_repro/        # Authors'-code FCF reproduction + paper-aligned eval
`-- scripts/              # quality_gate.py and shared helpers
```

## Methods

### FCF Baseline (`fcf/`)

Reproduces the two-stage text-encoder method from:

> Fan et al., "Fortified Concept Forgetting for text-to-image generative models
> by machine unlearning on CLIP", Computer Standards & Interfaces 97 (2026)
> 104142.

1. Stage 1 — explicit forgetting: move explicit prompts toward random
   noise-prompt embeddings.
2. Stage 2 — implicit forgetting: `fcf_p` (projection) or `fcf_e` (empirical).

> Note: the in-repo `fcf/` re-implementation diverged from the paper on subtle
> data/normalization details. The faithful reproduction lives in
> `models/fcf/` (authors' official code + data); see Results below.

### Novel FCF Extensions (`models/novel/`)

Independent from `fcf/`; keeps a local copy of the base trainer and adds:

- N1 CAP: causal activation patching to score concept-sensitive CLIP layers.
- N5 RG-FCF: spherical/Riemannian geodesic projection (`manifold="spherical"`;
  `"euclidean"` preserves baseline behavior).
- N6 OT-FCF: learned noise prompts selected by Wasserstein distance in CLIP space.

### LSSE (`lsse/`)

- N7 CNP: Concept Null-Space Projection.
- N8 CSR: Contrastive Semantic Retention.
- N9 CLM: CAP-guided layer masking.

Also exposes experimental switches (DDF, MACD, PLU, multi-direction CNP,
layer-dependent learning rates).

### DACE (`dace/`)

Dynamic Adversarial Concept Erasure edits only the CLIP text encoder, recomputing
the live concept subspace during training and anchoring the non-concept part. It
is primarily a **negative-result probe**: directly suppressing a text-embedding
concept axis does *not* reliably lower image-level ASR (it can even worsen it),
which motivated moving the intervention into the UNet.

### ODACE (`odace/`)

Output-Distribution Adversarial Concept Erasure — the first method here to edit
the **SD UNet cross-attention** (`to_q/k/v/out`) with the text encoder frozen.
Instead of a text-embedding proxy, it optimizes the UNet noise prediction that
actually drives the image. ODACE is the project's strongest eraser (see Results).

### Reproduced Baselines (`models/`)

Same-protocol reproductions for fair comparison:

- ESD-u (trained UNet, non-cross-attention).
- SLD (training-free inference guidance: Medium / Strong / Max).
- Safe-CLIP (public CLIP text-encoder swap).

## Results

The single source of truth is **`compare/comparison_all_methods.md`** (Table A:
all models, one ASR protocol; plus the FCF paper-alignment sections §③-정정-P1..P5).
Headlines (mean nudity ASR, lower = safer; NudeNet v3, 5 attack suites):

| Method | Intervention | mean ASR↓ | Note |
|---|---|---|---|
| **ODACE v3 / v1.5** | UNet cross-attn | **4.0** | strongest; raw-level COCO FID/CLIP |
| safe_neg / Sph+OT | inference / text enc. | ~15 | |
| **FCF-P** (official code) | text encoder | 17.2 | full-set 4-label **3.7 ≈ paper 3.43** |
| ESD-u | UNet non-cross-attn | 21.6 | strongest reproduced canon baseline |
| Safe-CLIP / SLD | text enc. / inference | 44–62 | broken by Ring-A-Bell attacks |
| DACE | text enc. (concept axis) | 51–74 | negative result |
| raw SD v1.4 | — | 62.0 | reference |

Key takeaways:

- **Intervention point dominates.** Inference < text-encoder < UNet for adversarial
  robustness; output-grounded cross-attention (ODACE) erases deepest, and edit
  *magnitude* is not the driver (ESD edits 95% of the UNet yet trails ODACE 5×).
- **FCF does reproduce** with the authors' official code + data: FCF-P reaches the
  paper's low-ASR regime (full-set 4-label 3.7 vs paper 3.43; rank Spearman ~0.9).
  The earlier in-repo `fcf/` re-implementation was unfaithful, not the method.
- **Text-encoder concept-axis erasure (DACE) underdetermines ASR** — a deliberate
  negative result that motivated ODACE.

> ASR is cross-model comparable (it is the nudity rate of attack images, independent
> of base). FID/CLIP scales differ by protocol — see the report's warnings before
> merging any quality numbers.

## Setup

Use a CUDA-capable Python environment for realistic training and image
generation. CPU works for small tests but is impractical for full SD runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r models/fcf/legacy_reimpl/requirements.txt
python -m pip install -r models/novel/requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r fcf\requirements.txt
python -m pip install -r models/novel/requirements.txt
```

Alternatively, create the conda environment:

```bash
conda env create -f environment.yml
conda activate fcf
```

The cross-model harness (`eval/`, `compare/`) additionally needs `diffusers`,
`nudenet` (v3), and `clean-fid`; the repository's eval scripts were run in a
conda env named `lsse` with those installed.

Models are downloaded from Hugging Face on first use:

- Stable Diffusion: `CompVis/stable-diffusion-v1-4` (and v1-5 / 2-1-base mirrors)
- CLIP text encoder: `openai/clip-vit-large-patch14`

Some Stable Diffusion / SD2.1 weights are gated; authenticate or rely on the
public mirrors wired into `eval/xeval.py`.

## Reproducing From Scratch

For a full from-scratch reproduction on a different machine (re-train every method,
regenerate every image), follow **[REPRODUCE.md](REPRODUCE.md)**. It pins exact
versions (`requirements-lock.txt`), fetches non-vendored external assets
(`scripts/fetch_external.sh`: FCF upstream code + data, Ring-A-Bell vectors), and
gives the train → register → generate → evaluate sequence. Note the two FCF
variants: the in-repo `fcf/` re-implementation (unfaithful, ASR ~52) versus the
authors' official-code reproduction (faithful, FCF-P 3.7 ≈ paper 3.43).

## Quality Gate

```bash
python scripts/quality_gate.py
```

Compiles the owned source directories, runs the unit tests, and scans for local
absolute paths (e.g. `C:/Users/...`) that would break reproduction elsewhere.
Generated outputs and the local `fcf/Q16/` checkout are skipped.

## Quick Start

Run commands from the repository root unless noted otherwise.

### Train (text-encoder methods)

```bash
python models/fcf/legacy_reimpl/train.py --config models/fcf/legacy_reimpl/configs/nudity_fcf_p.yaml          # FCF-P
python models/fcf/legacy_reimpl/train.py --config models/fcf/legacy_reimpl/configs/nudity_fcf_e.yaml          # FCF-E
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse.yaml    # LSSE
python models/dace/train_dace.py --config models/dace/configs/nudity_dace.yaml    # DACE
```

Common FCF overrides:

```bash
python models/fcf/legacy_reimpl/train.py --config models/fcf/legacy_reimpl/configs/nudity_fcf_p.yaml \
  --eta 0.3 --mu_p 0.8 --num_epochs 10 --device cuda
```

Novel FCF (spherical projection / OT noise / CAP) — see the per-flag examples in
`models/novel/` configs and `scripts/`:

```bash
python models/novel/train.py \
  --config models/novel/configs/nudity_v2.yaml --manifold spherical
```

### Train ODACE (UNet cross-attention)

```bash
python models/odace/train_odace.py --config models/odace/configs/nudity_odace.yaml       # v3 recipe (SD v1.4)
python models/odace/train_odace.py --config models/odace/configs/nudity_odace_v15.yaml   # v1.5 base
```

### Cross-Model Evaluation

Evaluate any registered model's ASR over the 5 attack suites, and its COCO
locality (FID / CLIP / LPIPS / CLIP-IQA):

```bash
python eval/xeval.py --models raw_v14,odace_v3,esd_u,fcf_p_official
python eval/eval_coco.py --models raw_v14,odace_v3,fcf_p_official
```

Registered model keys (UNet, text-encoder, inference, and baseline methods) live
in `eval/xeval.py:REGISTRY`. Reproduced baselines also have `models/run_eval.sh`.

The official-code FCF reproduction and its paper-aligned full-set / FID-5K /
violence evaluations are under `models/fcf/` (see that directory and the
report's §③-정정 sections).

## Image Generation

```bash
python models/fcf/legacy_reimpl/generate_images.py \
  --encoder_dir models/fcf/legacy_reimpl/outputs/fcf_p_nudity/final \
  --prompts_file models/fcf/data/eval/i2p_nudity.txt \
  --output_dir models/fcf/legacy_reimpl/outputs/images/fcf_p_nudity_i2p
```

Omit `--encoder_dir` for a raw Stable Diffusion baseline. The generator also
supports the CSV-style interface (`--prompts_path`, `--model_path`, `--save_path`,
`--model_name`).

## Evaluation (FCF subproject)

```bash
python fcf/evaluate.py --config models/fcf/legacy_reimpl/configs/nudity_fcf_p.yaml \
  --encoder_dir models/fcf/legacy_reimpl/outputs/fcf_p_nudity/final --concept nudity --eval_type asr
```

`--eval_type` also accepts `quality` and `style` (Van Gogh). Results are written
to `eval_results.json` under the selected `--output_dir`.

## Tests

```bash
python -m pytest models/fcf/legacy_reimpl/tests -q
python -m pytest models/novel/tests -q
python -m pytest models/lsse/tests -q
python -m pytest models/dace/tests -q
python -m pytest models/odace/tests -q
```

Tests use small mocked components where possible; full training/generation/eval
needs the model dependencies and enough GPU memory for Stable Diffusion.

## Data

Prompt files live under each subproject's `data/` tree:

- `data/prompts/`: explicit concept prompts, implicit names, retain prompts, aliases.
- `data/eval/`: I2P, Ring-A-Bell, Ring-A-Bell(Re), P4D, UnlearnDiffAtk (nudity +
  violence) and style prompts.

The per-subproject evaluation READMEs describe attack prompt names and sources.

## Outputs and Version Control

The root `.gitignore` excludes `outputs/` directories; model/checkpoint formats
(`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`, `.onnx`); Python caches, node
dependencies, local tool state, logs, archives, PDFs, and pickle files; and
`fcf/Q16/` (an embedded external repo for Q16 experiments). Store large training
artifacts separately if they need to be shared.

## Results Files

Per-track logs:

- `fcf/docs/results.md`, `models/novel/docs/results.md`, `lsse/docs/results.md`
- `odace/README.md`, `dace/README.md` (method-level write-ups)
- **`compare/comparison_all_methods.md`** (unified, authoritative)

Some older Korean notes have mojibake/encoding artifacts; prefer code, configs,
and the numeric tables in `compare/` as the source of truth.

## Notes

- Intervention points differ by track: `fcf/`, `models/novel/`, `lsse/`, and
  `dace/` fine-tune the **CLIP text encoder**; `odace/` edits the **SD UNet**
  cross-attention; `models/` covers UNet (ESD), text-encoder (Safe-CLIP), and
  inference-time (SLD) interventions.
- FCF default hyperparameters mirror the paper: learning rate `2.5e-5`,
  `eta=0.25`, `mu_p=0.7`, `mu_e=1.0`, `num_epochs=60`.
- `models/novel/`, `lsse/`, `dace/`, and `odace/` are independent experiments
  and must not import from `fcf/` during normal use.
