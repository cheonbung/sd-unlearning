# FCF Implementation Usage Guide

Implementation of:
> "Fortified Concept Forgetting for text-to-image generative models by machine unlearning on CLIP"
> Fan et al., Computer Standards & Interfaces 97 (2026) 104142

---

## Installation

```bash
pip install -r requirements.txt
```

**GPU Requirement**: NVIDIA GPU with ≥8.4 GB VRAM (paper used A6000).
Training on CPU is possible but very slow.

---

## Quick Start

### 1. Train FCF-P (Nudity Concept Forgetting)

```bash
python train.py --config configs/nudity_fcf_p.yaml
```

This runs two stages:
- **Stage 1** (Algorithm 1): Explicit concept forgetting — 500 steps
- **Stage 2** (Algorithm 2): Projection feature forgetting — 500 steps

Training time: ~16 min on A6000 (paper result).

### 2. Train FCF-E (Empirical variant)

```bash
python train.py --config configs/nudity_fcf_e.yaml
```

### 3. Train Van Gogh Style Forgetting

```bash
python train.py --config configs/vangogh_fcf_p.yaml
```

### 4. Train Violence Concept Forgetting

```bash
python train.py --config configs/violence_fcf_p.yaml
```

---

## Evaluation

### Generate images with fine-tuned model

```bash
# Generate from I2P prompts
python generate_images.py \
    --encoder_dir outputs/fcf_p_nudity/final \
    --prompts_file data/eval/i2p_nudity.txt \
    --output_dir outputs/images/fcf_p_nudity_i2p

# Generate SD baseline (no forgetting)
python generate_images.py \
    --prompts_file data/eval/i2p_nudity.txt \
    --output_dir outputs/images/sd_baseline_i2p
```

### Compute ASR (Table 1)

```bash
python evaluate.py \
    --encoder_dir outputs/fcf_p_nudity/final \
    --concept nudity \
    --eval_type asr
```

### Compute LPIPS for style forgetting (Table 2)

```bash
python evaluate.py \
    --encoder_dir outputs/fcf_p_vangogh/final \
    --concept vangogh \
    --eval_type style \
    --baseline_dir outputs/images/sd_baseline_vangogh
```

### Compute FID + CLIP score (Table 3)

```bash
python evaluate.py \
    --encoder_dir outputs/fcf_p_nudity/final \
    --concept nudity \
    --eval_type quality
```

---

## Ablation Study (Table 6)

Tests 4 configurations: ECFP×ICFP combinations.

```bash
python analysis/ablation_study.py \
    --config configs/nudity_fcf_p.yaml \
    --output_dir outputs/ablation_nudity
```

---

## Hyperparameter Sweep

Reproduces the η and μ_p sensitivity analysis from the paper.

```bash
python analysis/hyperparameter_sweep.py \
    --config configs/nudity_fcf_p.yaml \
    --output_dir outputs/sweep_nudity \
    --eta_values 0.01 0.1 0.25 0.6 \
    --mu_p_values 0.3 0.5 0.7 0.85
```

---

## Plot Training Curves

```bash
python analysis/plot_training.py \
    --explicit_log outputs/fcf_p_nudity/history_explicit.json \
    --implicit_log outputs/fcf_p_nudity/history_implicit.json \
    --output outputs/fcf_p_nudity/training_curves.png
```

---

## Project Structure

```
SD_unlearning/
├── fcf/
│   ├── trainer.py         # Core algorithms (Alg. 1, 2, 3)
│   ├── noise_utils.py     # Random noise text generation
│   └── dataset.py         # Prompt set management
├── evaluation/
│   ├── asr_evaluator.py   # NudeNet / Q16 ASR computation
│   ├── lpips_evaluator.py # LPIPS_f, LPIPS_m, LPIPS_d
│   └── fid_clip_evaluator.py  # FID and CLIP score
├── analysis/
│   ├── ablation_study.py      # Table 6: ECFP×ICFP ablation
│   ├── hyperparameter_sweep.py# η and μ_p sensitivity
│   └── plot_training.py       # Loss curve visualization
├── configs/
│   ├── nudity_fcf_p.yaml     # FCF-P config for nudity
│   ├── nudity_fcf_e.yaml     # FCF-E config for nudity
│   ├── vangogh_fcf_p.yaml    # FCF-P config for Van Gogh
│   └── violence_fcf_p.yaml   # FCF-P config for violence
├── data/
│   ├── prompts/               # Training prompt files
│   └── eval/                  # Evaluation prompt files
├── train.py               # Main training entry point
├── generate_images.py         # SD image generation
└── evaluate.py                # Full evaluation pipeline
```

---

## Key Hyperparameters (Paper Defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 2.5×10⁻⁵ | Adam optimizer LR |
| `η (eta)` | 0.25 | Forgetting weight in L_total |
| `μ_p (mu_p)` | 0.7 | Projection forgetting strength |
| `μ_e (mu_e)` | 1.0 | Empirical forgetting strength |
| `explicit_steps` | 500 | Stage 1 training steps |
| `implicit_steps` | 500 | Stage 2 training steps |

Adjust via CLI: `python train.py --config ... --eta 0.3 --mu_p 0.8`

---

## Evaluation Setup

### NudeNet (nudity detection)
```bash
pip install nudenet
```

### Q16 (violence detection)
The `evaluation/q16_classifier.py` includes a CLIP-based zero-shot fallback
if the original Q16 model is unavailable. For full accuracy, use the
original Q16 model from: https://github.com/ml-research/Q16

### Red-teaming tools for adversarial prompts
- **Ring-A-Bell**: https://github.com/chiayi-hsu/Ring-A-Bell
- **P4D**: https://github.com/joycenerd/P4D  
- **UnlearnDiffAtk**: https://github.com/OPTML-Group/Unlearn-Diff

---

## Multi-concept Forgetting (Table 4)

To forget multiple concepts from the same category simultaneously,
create a combined explicit/implicit prompt file and train:

```bash
# Create combined nudity+violence prompts
cat data/prompts/nudity_explicit.txt data/prompts/violence_explicit.txt \
    > data/prompts/nudity_violence_explicit.txt
# similarly for implicit and maintain

python train.py --config configs/nudity_fcf_p.yaml \
    # override prompt files as needed
```
