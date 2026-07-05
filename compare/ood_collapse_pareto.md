# OOD generation collapse & the coherence↔ASR Pareto frontier

**Problem.** On the out-of-distribution (OOD) Ring-A-Bell attack (adversarial *gibberish* token
sequences whose CLIP embedding is pushed near the nudity concept), several unlearning models do not
render a coherent *safe* image — they **collapse generation into non-human garbage**. NudeNet then
finds nothing, so their Ring-A-Bell ASR≈0 is earned by **collapse, not clean erasure** — a false
success.

**Coherence probe.** `models/fcf/eval_coherence.py` runs a CLIP zero-shot person-presence probe
(`person_prob`) over the already-generated attack images (no regeneration):
- `ring` = person_prob on Ring-A-Bell (OOD). Low ⇒ collapse.
- `i2p`  = person_prob on I2P (natural attack) = in-distribution control.

**ASR convention.** ASR is the **4-lab** mean (`fcf4_p03_mean`, FCF exposed-only labels, NudeNet
score>0.3) — the FCF-paper-comparable axis. Lower = better. Always read with `ring` (a low ASR with a
low `ring` is collapse, not erasure).

## Coherent-erasure Pareto frontier (frozen full-set, 2026-06-27)

| Method | site | ring | i2p | **ASR 4-lab↓** | note |
|---|---|---|---|---|---|
| Raw SD1.4 | — | 0.97 | — | ~40+ | no erasure (reference) |
| **ODACE benign-neg (n1)** | UNet | **1.00** | **0.90** | **2.1** | collapse SOLVED; best utility; dominates FCF-P |
| ODACE benign-anchor | UNet | 0.99 | 0.85 | 4.4 | collapse SOLVED (pure redirect) |
| **SLERP-OT (sph_ot)** | TE | 0.79 | 0.82 | **0.7** | strongest coherent erasure |
| FCF-P (official) | TE | 0.79 | 0.81 | ~3.4 | paper baseline |
| **LSSE geodesic (geo_e2)** | TE | 0.58 | 0.73 | 2.1 | LSSE collapse mitigated |
| — collapse-confounded (low ASR is NOT real): | | | | | |
| ODACE v3 | UNet | 0.12 | 0.58 | 0.7 | OOD collapse |
| LSSE R2q-ab (old flagship) | TE | 0.15 | 0.52 | 0.4 | OOD collapse |

**Pareto-optimal points:** `sph_ot` (0.79 / 0.7 — erases most) and **`odace_benign_n1`**
(1.00 / 2.1 — perfectly coherent, best utility). Neither dominates the other. `odace_benign_n1`
**dominates FCF-P** (higher ring 1.00>0.79 *and* lower 4-lab 2.1<3.4) and **LSSE geo_e2**
(1.00>0.58, 2.1=2.1). `odace_v3` and the old `lsse_r2q_ab` are **off the frontier**: their low ASR is
generation collapse (ring 0.12 / 0.15), not erasure.

## Mechanism: push-away collapses, redirect-to-benign does not

The decisive finding across 10 experiment cycles:

- **Push-away** erasure — move the conditioning *away* from the concept (LSSE read-out/raw/geodesic
  single+multi-axis; ODACE negative-guidance `e_0 − η·(e_p − e_0)`) — leaves the CLIP/output manifold on
  OOD concept-near prompts ⇒ **collapse**. Every push-away variant traces a Pareto **strictly inside**
  `sph_ot`. LSSE TE-space is exhausted (best honest point geo_e2 0.58 / 2.1).
- **Redirect-to-benign** — move the concept output *toward a coherent benign target* — is the only class
  that stays on-manifold. `sph_ot` (SLERP-OT toward benign) and FCF-P (projection) already do this in TE
  space. Porting it into ODACE's UNet output space (`erase_mode=benign_anchor`,
  `target = e_benign`; hybrid `benign_neg`, `target = e_benign − λ·(e_p − e_benign)`) **fully fixes the
  ODACE collapse** (ring 0.12→1.00) while keeping strong erasure.

**Ablation — the fix is the mechanism, not data-aug or a hyperparameter.** `odace_benign` used both
`benign_anchor` *and* OOD-prompt augmentation. Re-running `benign_anchor` with **OOD-aug OFF** (same
seed/steps/prompt) leaves `ring` **unchanged at 0.99** (`odace_benign_noood`: 0.99 / i2p 0.82 / 4-lab 6.3).
Contrast `odace_ood` (OOD-aug + push-away) = ring 0.18 (still collapsed). So the two factors are
**orthogonal**: `benign_anchor` fixes coherence (ring), OOD-aug is only a secondary erasure/utility helper
(4-lab 6.3→4.4, 8-lab 27.7→17.4). The geodesic LSSE result, by contrast, *is* η-sensitive (geo_e1 η=1
under-erases; geo_raw_e2 collapses at the same η) and never beats `sph_ot` — hyperparameters only move a
method *along* the Pareto curve, they do not create coherence.

## Validation (three independent de-risks, 2026-07-05)

The coherence story rests on one CLIP probe and one seed; three orthogonal checks answer the obvious
reviewer objections. Scripts: `eval/eval_coherence_triangulate.py`, `eval/eval_label_decomp.py`,
`eval/eval_multiseed.py` → `models/fcf/{coherence_tri,label_decomp,multiseed}.json`.

**(1) The probe is not a CLIP artifact (W2).** A CLIP-independent Haar face detector over the same
Ring-A-Bell images agrees with the CLIP `ring` probe at **Pearson r = +0.68** (14-model curated set):
collapse models have near-zero face rate (odace_v3 0.00, capcnp_zero 0.10) vs coherent
(odace_benign_n1 0.73, sph_ot 0.67). The ring↔i2p face **drop** replicates OOD-specificity
(odace_v3 i2p 0.31 → ring 0.00). HOG (r=0.03) and FID(ring vs own-i2p) (r=0.02) are confounded and
reported as such.

**(2) The redirect models' higher 8-lab is *clothed*, not exposed (W7).** Per-image label
decomposition: the coherent models' strict-8-lab surplus over 4-lab is almost entirely
`FEMALE_BREAST_COVERED`, with ~0 `BUTTOCKS_EXPOSED`.

| Model | 4-lab (exposed) | 8-lab | surplus (covered-only) | dominant surplus |
|---|---|---|---|---|
| Raw SD1.4 | 33.3 | 51.7 | 18.4 | breast-covered 15.1 + **buttocks-EXPOSED 7.4** |
| SLERP-OT | 1.4 | 13.6 | 12.2 | breast-covered 11.7 (buttocks-exposed 0.4) |
| ODACE benign-neg | 2.4 | 17.4 | 15.0 | breast-covered 14.2 |
| LSSE geo_e2 | 1.6 | 17.0 | 15.4 | breast-covered 15.1 |

Only raw has real exposure (buttocks-exposed 7.4); the redirect models' 8-lab is the strict labeler
penalising coherent clothed rendering → report **4-lab** as headline.

**(3) The separation is not seed noise (W3).** Re-generating Ring-A-Bell for the core 6 with 2 extra
seeds (mean ± std over 3 seeds); model weights fixed, only the sampling seed varies:

| Model | ring coherence (mean±std) | ring ASR 4-lab (mean±std) |
|---|---|---|
| Raw SD1.4 | 0.979 ± 0.005 | 85.3 ± 1.7 |
| **SLERP-OT** | 0.797 ± 0.015 | **0.0 ± 0.0** |
| **ODACE benign-neg** | **0.994 ± 0.002** | 0.35 ± 0.49 |
| ODACE v3 (collapse) | **0.128 ± 0.023** | 0.0 |
| LSSE R2q-ab (collapse) | **0.141 ± 0.017** | 0.0 |
| LSSE geo_e2 (partial) | 0.577 ± 0.004 | 1.75 ± 1.79 |

The collapse (~0.13) ↔ coherent (~0.8–0.99) gap of ~0.65 sits at std ~0.02 (**>30σ**). SLERP-OT and
ODACE benign-neg reach ring ASR4 ≈ 0 **and** stay coherent; odace_v3 / lsse_r2q_ab reach 0 via collapse.

Static paper figures for all of the above: `compare/export_figures.py` →
`compare/figures/{fig_coherence_scatter,fig_rpgrt_curves,fig_lsse_sweep,fig_multiseed}.{png,pdf}`.

## Reproduce

```bash
# WSL conda lsse. Models are in eval/xeval.py:REGISTRY.
# ODACE winner (benign-neg hybrid):
bash eval/run_ood_fix10.sh        # odace_benign_n05 / odace_benign_n1
# ODACE pure benign anchor:
bash eval/run_ood_fix9.sh         # odace_benign
# LSSE geodesic:
bash eval/run_ood_fix5.sh         # lsse_geo_e1 / geo_e2
# metrics: models/fcf/fullset_all.json (4-lab=fcf4_p03_mean) + models/fcf/coherence.json (ring/i2p)
```
