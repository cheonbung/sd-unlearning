# Reproducing SD Unlearning From a Fresh Clone

This is the from-scratch runbook for a **different machine**. It assumes you will
**re-train every method and regenerate every image** (no checkpoints or images are
committed — they are Git-ignored by design).

## What "reproducible" means here

- **Method + numbers reproducible:** following these steps re-creates every model and
  re-derives the comparison in `compare/comparison_all_methods.md` to within run-to-run noise.
- **NOT bit-identical:** generation/training are seeded (`seed=42`), but different GPU / CUDA /
  driver / PyTorch builds change floating-point results, so images and ASR/FID shift by a small
  margin. Expect *statistically equivalent*, not identical, numbers.
- **Two FCF reproductions exist** (do not confuse them):
  - `models/fcf/legacy_reimpl/` — our from-scratch re-implementation. Re-trainable from this repo. Gives the
    **unfaithful** result (FCF-P mean ASR ~52) because it uses `target - η·proj` *without* the
    `/(1-η)` normalization and word-list (not sentence-triplet) data. Documented in
    `comparison_all_methods.md` §③.
  - **official** — the **authors' code** (fetched, see Step 2), gives the **faithful** result
    (FCF-P full-set 4-label **3.7 ≈ paper 3.43**). This is the headline FCF result. §③-정정.

## 0. Prerequisites

- A CUDA GPU (results were produced on an RTX 4070, ~12 GB). CPU is impractical for full SD runs.
- `git`, `conda` (or `python -m venv`), ~50 GB free for downloaded weights + generated images.

## 1. Clone + environment (pinned)

```bash
git clone <this-repo-url> SD_unlearning
cd SD_unlearning

conda create -n lsse python=3.10 -y
conda activate lsse
# CUDA 12.1 torch build used for our results:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins the **exact** versions that produced the results (notably
`transformers==5.9.0`, `nudenet==3.4.2`, `diffusers==0.38.0`, `clean-fid==0.1.35`). Do **not**
float these: e.g. `transformers` 5.9 changed CLIP `get_image_features`, and `nudenet` v3 uses
different labels than the paper's v2 (see `comparison_all_methods.md` §③-정정-P2).

## 2. Fetch external assets (not vendored)

```bash
PYTHON=$(which python) bash scripts/fetch_external.sh
```

This fetches, into the working tree (Git-ignored):

| Asset | Source | License | Why |
|---|---|---|---|
| FCF upstream → `models/fcf/upstream/` | `github.com/f-c-forgetting/FCF` @ `e65e96a4` | **none** → fetch, not vendored | faithful FCF-P/E training |
| Ring-A-Bell vectors `*_vector.npy` | `chiayi-hsu/Ring-A-Bell` | per upstream | regenerate RaB attacks |
| Q16 `prompts.p` | `ml-research/Q16` | MIT | **vendored in-repo** (violence eval) |
| COCO val2017 | cocodataset.org | — | auto-downloaded by `eval_coco*.py` |
| SD 1.4/1.5/2.1, CLIP, Safe-CLIP, NudeNet | Hugging Face | per model | auto-downloaded; gated SD2.1/SD1.5 fall back to mirrors in `eval/xeval.py:REGISTRY` |

## 3. Train every method

All text-encoder/UNet trainers run in the `lsse` env. Outputs land in each track's `outputs/`
(Git-ignored). SLD and Safe-CLIP are **training-free** (skip).

```bash
# --- our text-encoder tracks ---
python models/fcf/legacy_reimpl/train.py  --config models/fcf/legacy_reimpl/configs/nudity_fcf_p.yaml          # FCF-P reimpl (~52)
python models/fcf/legacy_reimpl/train.py  --config models/fcf/legacy_reimpl/configs/nudity_fcf_e.yaml          # FCF-E reimpl
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse.yaml     # LSSE (+ --use_plu etc.)
python models/lsse/train_lsse.py --config models/lsse/configs/nudity_lsse_capcnp.yaml  # LSSE+CAP-CNP S2 (read-out-space erase; --cap_dir_mode/--cap_metric_mode for S1-S5/R variants)
python models/dace/train_dace.py --config models/dace/configs/nudity_dace.yaml     # DACE (negative result)
python models/novel/train.py --config models/novel/configs/nudity_v2.yaml --manifold spherical

# --- UNet tracks ---
python models/odace/train_odace.py --config models/odace/configs/nudity_odace.yaml      # ODACE v3 (champion)
python models/odace/train_odace.py --config models/odace/configs/nudity_odace_v15.yaml  # ODACE v1.5
python models/esd/train_esd.py --config models/esd/configs/nudity_esd_u.yaml   # ESD-u

# --- FAITHFUL FCF (authors' code; uses only torch+transformers+pandas, no LDM) ---
cd models/fcf/upstream
python concept_forgetting_train.py --input_prompts data/train/nudity.csv --save_path ../official_explicit.pt
python features_forgetting_P.py --model_path ../official_explicit.pt   # -> FCF-P (eta_clean=0.7)
python features_forgetting_E.py --model_path ../official_explicit.pt --experience_path ../experience.pth  # FCF-E
cd ../../..
# Convert each trained CLIPTextModel state into an HF dir the harness loads via te_swap:
#   load CLIPTextModel, load_state_dict(<trained>.pt), .save_pretrained("models/fcf/official_fcf_p/final")
#   (likewise official_fcf_e/final).  Hyperparams: lr 2.5e-5, eta 0.25, mu_p/eta_clean 0.7, mu_e 1.0.
```

## 4. Register checkpoints for the harness

`eval/xeval.py:REGISTRY` maps each model key to where its checkpoint must live. After training,
ensure the trained model sits at the registered path, e.g.:

| Key | Expected path (REGISTRY) |
|---|---|
| `odace_v3` / `odace_v15` | `models/odace/outputs/odace_v3/final` · `models/odace/outputs/odace_v15/final` (UNet) |
| `esd_u` | `models/esd/outputs/esd_u/final` (UNet) |
| `fcf_p` / `fcf_e` (reimpl) | `models/fcf/legacy_reimpl/outputs/fcf_{p,e}_nudity/final` (CLIPTextModel) |
| `fcf_p_official` / `fcf_e_official` | `models/fcf/official_fcf_{p,e}/final` (CLIPTextModel) |
| `lsse_plu`, `dace_v2`, `sph_ot`, … | see `te_dir` in REGISTRY |
| `sld_*`, `safeclip`, `safe_neg`, `raw_*`, `sd21base` | no training (config/HF id only) |

## 5. Generate + evaluate

```bash
# Efficacy (ASR) + COCO locality for any registered models:
python eval/xeval.py     --models raw_v14,odace_v3,esd_u,fcf_p_official,fcf_e_official,sld_medium,safeclip
python eval/eval_coco.py --models raw_v14,odace_v3,fcf_p_official        # COCO-300 FID/CLIP/LPIPS/IQ

# Paper-aligned FCF reproduction (full prompt sets, 4-label):
python models/fcf/eval_fullset.py            # raw + FCF-P/E full-set (1 img/prompt)
python models/fcf/rescore_fullset_paperrule.py   # paper presence-rule headline
python models/fcf/eval_fullset_all.py        # full-set for all Table-A methods
python models/fcf/eval_coco_fid5k.py         # COCO FID-5K (paper ~15 scale)
python models/fcf/eval_violence_q16.py       # violence Q16 locality (all models)
```

Results land as JSON next to each script and in each model's `outputs/.../metrics.json`. The unified
tables live in `compare/comparison_all_methods.md` (regenerate the live gallery + quantitative table via `compare/build_live_gallery.py`).

## 6. Caveats / gotchas (verified)

- **Run the documented commands above, not the `*.sh` wrappers.** Several `*/experiments/*.sh`,
  `models/run_eval.sh`, `eval/run_BC.sh`, and `models/novel/scripts/dispatch/*.sh` contain
  **machine-specific absolute paths** (e.g. `/mnt/d/...`, `/home/user/miniconda3/...`, and a legacy
  `/mnt/c/Users/vip/...` from an older machine). They are run logs of how *we* invoked things, not
  portable entrypoints. The Python entrypoints are repo-relative and portable; set `PYTHON=` or run
  inside the activated env.
- **NudeNet version:** we score with v3 (`3.4.2`); the paper used v2 labels. ASR is compared by
  band/rank/%-reduction, not absolute value (§③-정정-P2).
- **HF model drift:** `runwayml/stable-diffusion-v1-5` was removed and SD2.1-base is gated; mirrors
  are wired into `eval/xeval.py:REGISTRY` and fall back automatically.
- **Q16 under transformers 5.9:** `models/lsse/evaluation/q16_classifier.py` breaks (returns a ModelOutput);
  `models/fcf/eval_violence_q16.py` ships a self-contained, version-robust Q16 scorer.
- **FCF violence data exists** (`models/fcf/upstream/data/train/violence.csv`) if you want to
  reproduce the paper's violence-forgotten models (our shipped FCF checkpoints forgot nudity only).
