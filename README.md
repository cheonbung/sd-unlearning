# SD Unlearning

Research code for **concept unlearning in Stable Diffusion**, comparing erasure
methods across different *intervention points* — the CLIP text encoder, the SD
UNet cross-attention, and inference-time guidance — under one shared evaluation
protocol. The project grew from a single FCF reproduction into a multi-method
comparison whose headline finding is that **where** you intervene matters more
than how much you change: output-grounded UNet editing (ODACE) erases far more
robustly than text-encoder proxies.

The repository groups several experiment tracks plus a cross-model harness:

- `models/fcf/`: Fortified Concept Forgetting (FCF) — the authors'-code
  reproduction (`official_fcf_{p,e}/`, canonical) plus the archived in-repo
  re-implementation (`legacy_reimpl/`).
- `models/novel/`: independent FCF extensions (manifold projection,
  optimal-transport noise prompts incl. SLERP-OT, causal activation analysis).
- `models/lsse/`: Layer-Selective Semantic Erasure (LSSE), a single-loop text-encoder
  alternative with null-space forgetting, contrastive retention, layer masking.
- `models/dace/`: Dynamic Adversarial Concept Erasure — a text-encoder concept-axis
  experiment (a deliberate negative-result probe).
- `models/odace/`: Output-Distribution Adversarial Concept Erasure — the first method
  here that edits the **UNet cross-attention** instead of the text encoder.
- `models/{esd,sld,safeclip}/`: same-protocol reproductions of ESD, SLD, and Safe-CLIP.
- `eval/`: cross-model ASR + COCO FID/CLIP evaluation harness.
- `compare/`: the unified comparison report, the live gallery
  (`build_live_gallery.py` → image grid + quantitative table with condition
  toggles), and the official-code FCF reproduction.

Generated checkpoints, images, logs, external model weights, and other large
artifacts are intentionally Git-ignored. Re-run training/evaluation to recreate
them under each subproject's `outputs/` directory.

## Repository Layout

```text
.
|-- models/                   # One directory per method
|   |-- fcf/                  # FCF-P / FCF-E (CLIP text encoder)
|   |   |-- official_fcf_{p,e}/   # authors'-code reproduction (canonical, ASR ~3.7)
|   |   `-- legacy_reimpl/        # earlier in-repo reimpl (archived, unfaithful)
|   |-- novel/                # FCF extensions (CAP, spherical, OT, SLERP-OT)
|   |-- lsse/                 # Layer-Selective Semantic Erasure (text encoder)
|   |-- dace/                 # Dynamic Adversarial Concept Erasure (text encoder)
|   |-- odace/                # Output-Distribution Adversarial Concept Erasure (UNet)
|   |-- esd/ sld/ safeclip/   # Same-protocol baselines
|   `-- run_eval.sh
|-- eval/                     # Cross-model eval harness
|   |-- xeval.py              # ASR over 5 attack suites for any registered model
|   `-- eval_coco.py          # COCO FID / CLIP / LPIPS / CLIP-IQA locality
|-- compare/                  # Unified comparison + official FCF reproduction
|   |-- comparison_all_methods.md   # <-- authoritative results table
|   |-- build_live_gallery.py       # live image gallery + quantitative table (HTML)
|   `-- fcf_repro/                  # Authors'-code FCF reproduction + paper-aligned eval
`-- scripts/                  # quality_gate.py and shared helpers
```

## Methods

### FCF Baseline (`models/fcf/`)

Reproduces the two-stage text-encoder method from:

> Fan et al., "Fortified Concept Forgetting for text-to-image generative models
> by machine unlearning on CLIP", Computer Standards & Interfaces 97 (2026)
> 104142.

1. Stage 1 — explicit forgetting: move explicit prompts toward random
   noise-prompt embeddings.
2. Stage 2 — implicit forgetting: `fcf_p` (projection) or `fcf_e` (empirical).

> Note: the in-repo `models/fcf/legacy_reimpl/` re-implementation diverged from the paper on subtle
> data/normalization details. The faithful reproduction lives in
> `models/fcf/` (authors' official code + data); see Results below.

### Novel FCF Extensions (`models/novel/`)

Independent from `models/fcf/`; keeps a local copy of the base trainer and adds:

- N1 CAP: causal activation patching to score concept-sensitive CLIP layers.
- N5 RG-FCF: spherical/Riemannian geodesic projection (`manifold="spherical"`;
  `"euclidean"` preserves baseline behavior).
- N6 OT-FCF: learned noise prompts selected by Wasserstein distance in CLIP space.

### LSSE (`models/lsse/`)

- N7 CNP: Concept Null-Space Projection.
- N8 CSR: Contrastive Semantic Retention.
- N9 CLM: CAP-guided layer masking.

Also exposes experimental switches (DDF, MACD, PLU, multi-direction CNP,
layer-dependent learning rates).

### DACE (`models/dace/`)

Dynamic Adversarial Concept Erasure edits only the CLIP text encoder, recomputing
the live concept subspace during training and anchoring the non-concept part. It
is primarily a **negative-result probe**: directly suppressing a text-embedding
concept axis does *not* reliably lower image-level ASR (it can even worsen it),
which motivated moving the intervention into the UNet.

### ODACE (`models/odace/`)

Output-Distribution Adversarial Concept Erasure — the first method here to edit
the **SD UNet cross-attention** (`to_q/k/v/out`) with the text encoder frozen.
Instead of a text-embedding proxy, it optimizes the UNet noise prediction that
actually drives the image. The original negative-guidance variant (`odace_v3`)
pushes generation away from the concept and was later found to collapse
generation on OOD Ring-A-Bell attacks (fake-low ASR from broken images, not real
erasure). The **benign-anchor / benign-neg** variants redirect toward a benign
target instead of pushing away, stay coherent under attack, and are the
project's actual current-best UNet points (see Results).

### Reproduced Baselines (`models/{esd,sld,safeclip}/`)

Same-protocol reproductions for fair comparison:

- ESD-u (trained UNet, non-cross-attention).
- SLD (training-free inference guidance: Medium / Strong / Max).
- Safe-CLIP (public CLIP text-encoder swap).

## Results

The single source of truth is **`compare/comparison_all_methods.md`** (Table A:
all models, one ASR protocol; plus the FCF paper-alignment sections §③-정정-P1..P5).

> **⚠ Coherence-aware headline.** ASR alone is not sufficient: some low-ASR points
> turn out to be **OOD generation collapse** on Ring-A-Bell adversarial prompts
> (the model stops generating people at all, so nothing is left to detect as
> nudity — see "Excluded experiments" below). All rows below are the
> **non-collapsed** methods, reported as **4-lab ASR** (FCF exposed-only
> convention) paired with **Ring-A-Bell coherence** (CLIP person-presence
> probability; ≥0.7 = coherent, the model still generates people under attack).

| Method | Intervention | ASR 4-lab↓ | Ring-A-Bell coherence | COCO-CLIP↑ | Note |
|---|---|---|---|---|---|
| **SLERP-OT (Sph+OT)** | inference / text enc. | **0.7** | 0.79 (coherent) | 23.65 | honest OOD-coherent winner; RPG-RT worst-case gap 0 |
| **ODACE benign-neg** | UNet cross-attn (redirect) | **2.1** | 1.00 (coherent) | 25.43 | strongest coherent point overall |
| LSSE geodesic | text enc. | 2.1 | 0.58 | 25.66 | coherent TE alternative to Sph+OT |
| ODACE benign-anchor | UNet cross-attn (redirect) | 4.4 | 0.99 (coherent) | 25.62 | most adaptation-proof under DPO red-team |
| FCF-P (official code) | text encoder | 3.7 ≈ paper 3.43 | 0.79 (coherent) | ~24.4 | reproduces paper (Spearman 0.9) |
| ESD-u | UNet non-cross-attn | 12.0 | 0.85 (coherent) | 25.0 | strongest reproduced canon baseline |
| Safe-CLIP | text enc. | 32.2 | 0.85 (coherent) | 25.5 | bypassed by Ring-A-Bell semantically, not via collapse |
| SLD-Max | inference | 18.9 | 0.92 (coherent) | 24.1 | bypassed by Ring-A-Bell semantically, not via collapse |
| DACE | text enc. (concept axis) | 51–74 (8-lab) | — | — | negative result |
| raw SD v1.4 | — | 50.4 | 0.97 (coherent, unerased) | — | reference |

See `models/lsse/README.md` and `models/odace/README.md` for the full per-variant
frontier behind the LSSE geodesic / ODACE benign rows above.

Key takeaways:

- **Redirect-to-benign beats push-away once coherence is checked.** Methods that
  push generation *away* from the concept (ODACE negative-guidance, LSSE
  CAP-CNP's most aggressive variants) can collapse image generation on OOD
  attacks and win on ASR for the wrong reason. Methods that redirect *toward* a
  benign target (SLERP-OT, ODACE benign-anchor/benign-neg, LSSE geodesic) stay
  coherent and are the honest current winners.
- **FCF does reproduce** with the authors' official code + data: FCF-P reaches the
  paper's low-ASR regime (4-label 3.7 vs paper 3.43; rank Spearman ~0.9). The
  earlier in-repo `models/fcf/legacy_reimpl/` re-implementation was unfaithful,
  not the method.
- **Text-encoder concept-axis erasure (DACE) underdetermines ASR** — a deliberate
  negative result that motivated moving the intervention into the UNet/read-out space.

### Adaptive robustness (RPG-RT), multi-concept, and training cost

Three further axes beyond static-attack ASR (full tables in
`compare/comparison_all_methods.md` §⑥–⑧):

- **Adaptive red-team (RPG-RT, vicuna-7b prompt-rewrite attacker), asr_query↓ =
  worst-case bypass rate:** Sph+OT **2.5** < ODACE benign-neg 5.0 < LSSE geodesic
  7.0 < ODACE benign-anchor 9.0 ≈ FCF-P 9.0 < ESD-u 10.5 ≈ ODACE-MC 10.5 <
  SLD-Max 22.0 < Safe-CLIP 34.0 < raw 64.5. Sph+OT is the most attack-resistant
  text-encoder method; the ODACE benign/redirect variants and LSSE geodesic hold
  up nearly as well, while shallow interventions fall hard — ESD rewritten
  through at asr_prompt 65, and SLD-Max / Safe-CLIP at 80 / 90 (static ASR
  collapses under adaptive rewriting).
- **DPO-fine-tuned adaptive attacker (4 iters/target), worst-case asr_query
  iter0→best:** Sph+OT **3.0→3.0** (gap 0 — DPO never beats iter0) < ODACE
  benign-neg 2.0→4.0 < LSSE geodesic 3.5→4.5 < ODACE benign-anchor 8.5→8.5
  (gap 0) < FCF-P 2.5→7.5 < ESD 6.5→10.5 ≪ raw **52.5→75.0**. Sph+OT and ODACE
  benign-anchor are structurally unmovable (DPO gains nothing over the frozen
  attacker); raw explodes, confirming the attacker works. (The original
  negative-guidance ODACE v3 posted a lower-looking 0.0→1.5 here too, but that
  reflects OOD collapse, not robustness — see "Excluded experiments" below.)
- **Multi-concept (nudity + violence + Van Gogh, one model):** only **ODACE-MC**
  erases all three while keeping utility (COCO-CLIP **24.8** ≈ raw 26.5). Text-encoder
  multi-concept (LSSE-MC, Sph+OT-MC) collapses the model (COCO-CLIP ~10–12, FID
  183–302) — their low nudity/violence ASR is a **broken-output artifact**, not real
  erasure (Sph+OT-MC does not even remove the Van Gogh style).
- **Training cost (RTX 4070, GPU-h):** text-encoder methods are 50–150× cheaper
  (Sph+OT 0.026, DACE 0.007–0.009) than UNet methods (ESD 1.33, ODACE-MC 1.23), but
  only UNet scales to robust multi-concept erasure — a clear cost ↔ capability trade-off.

> ASR is cross-model comparable (it is the nudity rate of attack images, independent
> of base). FID/CLIP scales differ by protocol — see the report's warnings before
> merging any quality numbers.

### Excluded experiments (OOD generation collapse)

The following variants were trained and evaluated but are **excluded from the
results above** because their low ASR turned out to be a side effect of the
model collapsing image generation on Ring-A-Bell adversarial prompts (CLIP
person-presence probability < 0.3, vs. ≥0.7 for a coherent model) rather than
genuine concept erasure. They remain in the live gallery's diagnostic sections
as evidence of the collapse phenomenon itself, but are not comparison
candidates:

- **ODACE v3 / v1.5** (negative-guidance UNet cross-attn) — static ASR 4.0
  looked strongest, but Ring-A-Bell coherence 0.12.
- **LSSE CAP-CNP `r2q_ab` / `r2q_a` / `capcnp_zero`** (text-encoder read-out
  erasure, aggressive variants) — `r2q_ab` ASR 3.1 / coherence 0.15; `r2q_a`
  ASR 1.3 / coherence ~0.01.
- **LSSE R2q (violence-trained)** — same read-out over-erasure failure mode on
  the violence concept.

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
variants: the in-repo `models/fcf/legacy_reimpl/` re-implementation (unfaithful, ASR ~52) versus the
authors' official-code reproduction (faithful, FCF-P 3.7 ≈ paper 3.43).

## Quality Gate

```bash
python scripts/quality_gate.py
```

Compiles the owned source directories, runs the unit tests, and scans for local
absolute paths (e.g. `C:/Users/...`) that would break reproduction elsewhere.
Generated outputs and the local `models/fcf/Q16/` checkout are skipped.

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
`models/fcf/Q16/` (an embedded external repo for Q16 experiments). Store large training
artifacts separately if they need to be shared.

## Results Files

Per-track logs:

- `models/fcf/docs/results.md`, `models/novel/docs/results.md`, `models/lsse/docs/results.md`
- `models/odace/README.md`, `models/dace/README.md` (method-level write-ups)
- **`compare/comparison_all_methods.md`** (unified, authoritative)

Some older Korean notes have mojibake/encoding artifacts; prefer code, configs,
and the numeric tables in `compare/` as the source of truth.

## Notes

- Intervention points differ by track: `models/fcf/`, `models/novel/`, `models/lsse/`, and
  `models/dace/` fine-tune the **CLIP text encoder**; `models/odace/` edits the **SD UNet**
  cross-attention; `models/{esd,sld,safeclip}/` covers UNet (ESD), text-encoder (Safe-CLIP), and
  inference-time (SLD) interventions.
- FCF default hyperparameters mirror the paper: learning rate `2.5e-5`,
  `eta=0.25`, `mu_p=0.7`, `mu_e=1.0`, `num_epochs=60`.
- `models/novel/`, `models/lsse/`, `models/dace/`, and `models/odace/` are independent experiments
  and must not import from `models/fcf/` during normal use.
