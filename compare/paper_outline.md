# Paper outline — "The Coherence Illusion in Concept Erasure"

Spine **A** (diagnostic + mechanism first). Chosen because the pure-SOTA spine is undermined by our
own finding: the lowest-ASR models partly achieve it via OOD generation collapse. Framing the
collapse as the *discovery* turns that weakness into the paper's edge.

> Standing numbers live in `models/fcf/*.json`; the live gallery (`compare/build_live_gallery.py`)
> renders every figure/table below. Report nudity ASR as **4-lab** (FCF exposed-only); always pair
> with the OOD coherence axis. Verify `file:line` / JSON before quoting.

---

## Title candidates
- The Coherence Illusion in Concept Erasure: Why Low Attack Success Can Mean Generation Collapse
- Erase or Break? A Coherence-Aware Evaluation of Diffusion Concept Unlearning

## Abstract (sketch)
Concept-erasure methods for diffusion models are ranked by attack success rate (ASR). We show ASR is
**insufficient**: on out-of-distribution adversarial prompts (Ring-A-Bell), aggressively-erased models
reach near-zero ASR by rendering **incoherent non-human images**, not by safely rendering the concept
away. We introduce a **coherence probe** (validated by a CLIP-independent detector, r=+0.68) that
separates genuine erasure from collapse, and show the collapse is caused by **push-away** objectives
leaving the generative manifold. A **redirect-to-benign** objective, when **grounded in the output
space** (ODACE) or applied as embedding interpolation (SLERP-OT), stays on-manifold and achieves
honest low 4-lab ASR (0.7–2.1) with high coherence and adaptation-proof robustness (RPG-RT gap ≈ 0).
Read-out-space text-encoder erasure (LSSE) cannot get both, which we use to prove output-grounding is
the key.

## Contributions
- **C1 (diagnostic).** ASR alone cannot certify erasure; a coherence probe is needed. The
  coherence–erasure tradeoff: aggressive low-ASR optimization falls off the coherence cliff, while
  prior baselines sit at the weak-but-coherent end (they do NOT collapse — ring 79–92). *Not* "prior
  SOTA cheated"; the diagnostic is what's new.
- **C2 (mechanism).** Collapse is caused by push-away objectives; redirect-to-benign fixes it. Causal
  evidence: the no-OOD-aug ablation (benign stays coherent without data aug) + the LSSE strength sweep
  (ASR↓ tracks ring↓ monotonically at the extremes).
- **C3 (why output-grounding).** Redirect works only when grounded in the *output*: ODACE benign
  (ring 100, 4-lab 2.1) succeeds; LSSE read-out-space redirect (slerp/redirect) stays coherent but
  fails to erase (ring 88–96, ASR 42–52). The anchor must constrain the generated image.
- **C4 (concept-locality + robustness).** Output-grounded editing is concept-local (ODACE nudity
  erase leaves violence Q16 ≈ raw); read-out-space suppresses a broad "unsafe" direction. Redirect
  defenses are adaptation-proof under RPG-RT (gap ≈ 0).

## Section structure
1. **Intro** — ASR is the field's yardstick; motivate with a collapse image strip (raw vs push-away
   garbage vs redirect safe). State C1–C4.
2. **Related work** — erasure families (ESD/UCE/RECE/MACE/FCF/SLD/Safe-CLIP); adversarial prompts
   (Ring-A-Bell/P4D/UnlearnDiff); adaptive RPG-RT. Gap: none evaluate OOD coherence.
3. **The coherence probe** — CLIP person_prob on ring vs i2p; **validation** by CLIP-independent Haar
   face detector (r=+0.68) + 8-lab-vs-4-lab decomposition (surplus = clothed).
4. **Mechanism** — push-away vs redirect-to-benign; TE vs UNet taxonomy; the output-grounding argument
   with LSSE as the contrast case; no-OOD-aug ablation.
5. **Results** — coherence-aware Pareto (4-lab ASR × coherence × CLIP); honest winners SLERP-OT &
   ODACE-benign; adaptive robustness (RPG-RT curves).
6. **Analysis** — concept-locality (cross-concept transfer); the LSSE strength sweep.
7. **Limitations & conclusion.**

## Figures / tables → existing artifacts
| Fig/Tab | Content | Source (gallery section / JSON) |
|---|---|---|
| Fig 1 | Collapse image strip (ring vs i2p) | `ood_image_strip` · `_fs/*/*.png` |
| Fig 2 | Coherence × ASR scatter w/ collapse zone | `drawCollapse` · coherence.json + fullset_all.json |
| Fig 3 | Method taxonomy 2×2 | `taxonomy_section` |
| Fig 4 | RPG-RT adaptation curves | `rpgrt_curve_section` · rpgrt_dpo.json |
| Tab 1 | Main coherence-aware table (4/8-lab, CLIP, ring, Q16) | `fullset_table` |
| Tab 2 | Probe validation (CLIP vs face r=+0.68) | `validation_section` · coherence_tri.json |
| Tab 3 | 8-lab surplus decomposition (covered vs exposed) | `validation_section` · label_decomp.json |
| Tab 4 | Cross-concept transfer | `transfer_heatmap_section` · violence_q16.json |
| Tab 5 | LSSE strength sweep (ASR↔ring) | fullset_all.json + coherence.json |

**TODO (no-GPU):** export Figs 1–4 as static PNG/PDF (gallery charts are JS-rendered).

## Honest framing / threats to validity (must state)
- Report 4-lab as headline; 8-lab surplus is clothed (Tab 3), so it is not leakage but strict-labeler
  penalty on coherent rendering.
- The LSSE "flagship" (R2q-ab 3.1 8-lab) dominates SLERP-OT on the 2D (ASR,CLIP) plane but **loses
  once coherence is the 3rd axis** — do not headline it as SOTA.
- Coherence probe is one proxy; corroborated by an independent detector but a fuller quality metric
  (attack-FID vs a fixed real reference) is future work.

## Remaining experiments (deferred GPU queue)
- **Multi-seed** error bars on the core 6 (needs a seed param in `xeval.generate`).
- **SD2.1 / SDXL** generalization of push=collapse / redirect=coherent (mechanism generality).
- **Recent baselines** (UCE / RECE / MACE) under the coherence lens.
