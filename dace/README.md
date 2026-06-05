# DACE — Dynamic Adversarial Concept Erasure

An **independent** SD-unlearning research project (no `fcf/`, `fcf-novel-methods/`, or
`lsse/` code imported; its own single training loop). Trains only the CLIP text encoder
(`openai/clip-vit-large-patch14` == SD-v1-4 text encoder), so checkpoints are evaluated
with the shared `lsse/evaluate.py` harness for an apples-to-apples ASR comparison.

## Motivation (grounded in diagnostics, not guesswork)

Prior LSSE diagnostics showed the dominant failure of static single-direction erasure is
**concept rerouting** (`residual_concept_variance` flat ~0.34; embedding erasure != adversarial
robustness). DACE pursues the concept subspace **dynamically** as it moves during training
and pins the orthogonal complement to block rerouting.

## What the cheap P0 gates found (before any expensive training)

| Probe | Metric | Spearman vs mean ASR | Verdict |
|---|---|---|---|
| **P0**  | forget-vs-retain held-out separability | **-0.893** | REFUTED — lexically saturated (~1.0 for every checkpoint incl. best eraser) |
| **P0b** | concept-axis shift (explicit vs concept-stripped neutral) | **+0.821** | SUPPORTED — this is the right axis |

P0 killed the naive "separate forget from retain" objective *before* a training run.
DACE was then **corrected** to the concept-axis: minimize how much adding the concept word
moves the embedding, inside a dynamically tracked concept subspace.

**Important honest finding from P0b:** `lsse_plu` already drives concept_shift to 1.69 yet
sits at ASR ~21, while `sph_ot` has a LARGER shift (3.81) but LOWER ASR (15.6). So a pure
text-encoder method appears to hit an ASR **floor (~20)**; sph_ot's extra edge lives in the
embedding->UNET decoding path that no text-embedding metric captures. DACE (text-encoder only)
is expected to reach the lsse_plu range, not to beat sph_ot.

## Method (concept-axis, corrected)

```
min_theta  alpha*L_forget + gamma*L_ortho + beta*L_retain
  U      = top-k SVD of live concept-shift vectors d_i = pool(z_explicit_i) - pool(z_neutral_i)
           recomputed (closed form) every adv_every steps  -> pursuit of the moving concept
  L_forget = || U^T (pool(z_exp) - pool(z_neu)) ||^2     (adding concept word stops moving emb in U)
  L_ortho  = || (I-UU^T)(pool(z_exp) - frozen) ||^2      (pin non-concept part: anti-rerouting)
  L_retain = MSE(z_neu, frozen) + MSE(z_retain, frozen)  (preserve content, no destruction)
```

## Layout

```
dace/
  methods/   adversary.py (concept_subspace, discriminative_subspace, SubspaceTracker)
             erasure.py   (dace_concept_losses, dace_losses)
             diagnostics.py (held-out linear_separability_auc, concept metrics)
  core/      dataset.py (DACEDataset, neutralize), trainer.py (DACETrainer)
  experiments/ p0_crossmethod_diag.py, p0b_concept_axis.py, run_dace.sh
  configs/   nudity_dace.yaml
  tests/     test_dace.py  (6 passing)
  data/prompts/  nudity_explicit/maintain/implicit.txt (self-contained)
```

## Usage (WSL conda env `lsse`)

```bash
python -m pytest tests -q                              # unit tests
python experiments/p0_crossmethod_diag.py              # P0 gate
python experiments/p0b_concept_axis.py                 # P0b gate (concept axis)
python train_dace.py --config configs/nudity_dace.yaml # train -> outputs/dace_nudity/final
# ASR eval through the shared harness:
cd ../lsse && python evaluate.py --encoder_dir ../dace/outputs/dace_nudity/final \
    --concept nudity --eval_type asr --output_dir outputs/eval/xharness_dace --num_images 50
```
