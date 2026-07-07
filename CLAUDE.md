# CLAUDE.md — SD Unlearning (FCF reproduction + novel methods)

<!-- Maintainer note (stripped from Claude's context): keep this file <=200 lines. Claude Code
     loads it every session; shorter + specific = better compliance. Put standing rules here;
     let detailed experiment findings accumulate in auto-memory, not in this file. -->

Standing, human-authored instructions for Claude Code, loaded every session. **This file = durable
rules + architecture.** Per-experiment findings/numbers/negative-results live in **auto-memory**
(`~/.claude/projects/<repo>/memory/`, indexed by `MEMORY.md`) — consult it for detail. Both are
point-in-time; verify `file:line` against current code before asserting.

- **Repo:** `D:\unlearning\SD_unlearning` · remote `github.com/cheonbung/sd-unlearning`
- **Branch:** `chore/restructure-models` (main = `main`)
- **Goal:** reproduce FCF (Fan et al., CSI 97, 2026) for SD concept unlearning (nudity / violence /
  art-style), then beat it with novel methods (LSSE+CAP-CNP, ODACE).

---

## 1. Repository layout

```
models/
  fcf/    upstream FCF reproduction (DO NOT edit upstream trainer) + eval/data hub:
          data/eval/*.txt (attack prompts), eval_*.py (metric entry points), *.json (metric tables)
  lsse/   LSSE + CAP-CNP — flagship text-encoder method. train_lsse.py, configs/, README.md
  odace/  ODACE — output-grounded UNet cross-attn editing. train_odace.py
  dace/   DACE — text-embedding axis erasure (negative result; see auto-memory). train_dace.py
  esd/    ESD-u baseline. train_esd.py
  novel/  misc novel-method scratch
eval/     cross-model harness: xeval.py (50-prompt), run_*.sh runners, lpips_style.py,
          aggregate_cost.py, rpgrt_*.py (RPG-RT red-team)
compare/  build_live_gallery.py (HTML results gallery), comparison_all_methods.md, verify_*.py
baselines/  reproduced baselines (ESD-u, Safe-CLIP, SLD) in our harness
RPG-RT/   sibling repo D:\unlearning\RPG-RT — NeurIPS25 DPO-LLM adaptive nudity attacker
```

The method families are **independent**. Hard rules:
- **Never edit `models/fcf/` upstream trainer code** (authors' parent repo). Only its `eval_*.py`
  and data are ours to extend. Measuring FCF *training* cost is off-limits; *inference* eval is fine.
- **No imports** between `models/fcf/` or `models/novel/` and LSSE code.

---

## 2. Environments & commands

| Task | Interpreter |
|---|---|
| Training / generation / eval (torch+CUDA) | **WSL** conda `lsse`: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate lsse` |
| Gallery build / JSON munging (no GPU) | **Windows** `C:\Users\USER\anaconda3\python.exe` |
| git (commit/push) | **Windows** PowerShell git ONLY — never WSL git |

- Never use the Windows `WindowsApps` `python` (broken stub, exit 49) — use the anaconda path above.
- GPU: single RTX 4070 (12 GB). One heavy job at a time. vicuna-7b (RPG-RT) OOMs at 12 GB (needs 8-bit).
- SD2.1-base official `stabilityai` is HF-gated; mirrors `Manojb/` and `sd2-community/` load tokenless.

---

## 3. Methods & flagship results (frozen full-set nudity ASR %, lower=better)

Primary table = frozen full-set (I2P 931 / RaB 95 / RaB(Re) 95 / P4D 361 / UDA 142). Two means:
**8-lab** (strict: 4 exposed + covered + buttocks) and **4-lab** (FCF exposed-only), NudeNet score>0.3.

| Method | site | ASR 8-lab | ASR 4-lab | COCO-CLIP↑ | Note |
|---|---|---|---|---|---|
| FCF-P (official) | text-enc | 3.7 | ~3.4 | ~26 | reproduces paper (Spearman 0.9) |
| Sph+OT (SLERP-OT) | text-enc | 15.6 | — | high | prior best TE baseline |
| **ODACE v3** | UNet x-attn | **4.0** | — | ~25.5 | output-grounded; beats all baselines |
| **LSSE CAP-CNP R2q-ab** | text-enc | **3.1** | 0.5 | 23.62 | flagship TE point; dominates Sph+OT |
| LSSE R2q-a (max-forget) | text-enc | 1.3 | — | 20.13 | strongest forget, lower utility |
| ESD-u / Safe-CLIP / SLD | various | 21.6 / 44 / 45–62 | — | — | baselines, all dominated |

> **⚠ Coherence-aware correction (2026-07, see auto-memory [[coherence-validation-honest-sota]]):** this
> 2-axis (ASR, CLIP) table is **incomplete** — it rewards OOD generation collapse. The lowest-ASR points
> (LSSE R2q-a 1.3/ring 1%, **R2q-ab 3.1/ring 15%**, ODACE v3 5.2/ring 12%) partly collapse on Ring-A-Bell,
> so "R2q-ab dominates Sph+OT" holds in 2D but **flips once OOD coherence is the 3rd axis**. Report ASR as
> **4-lab** and always pair with ring person_prob. The *honest* OOD-coherent winners are **SLERP-OT**
> (4-lab 0.7, ring 79, RPG-RT gap 0) and **ODACE benign-anchor** (4-lab 2.1, CLIP 25.4, ring 100). Probe
> validated by a CLIP-independent face detector (r=+0.68); redirect models' higher 8-lab is *clothed*
> (breast-covered), not exposed. Baselines do **not** collapse (ring 79–92).

**CAP-CNP (LSSE flagship):** erase in the UNet cross-attn **read-out space** `R = C·M^½`,
`M = mean_ℓ(WₖᵀWₖ + WᵥᵀWᵥ)` a frozen-UNet constant ⇒ gradient flows to the **text-encoder only**.
R2q-ab = `A` (read-out retain anchor `cap_retain_anchor`) + `B` (`perlayer_causal`). Model keys are
in `eval/xeval.py:REGISTRY` (`lsse_capcnp`, `lsse_r2q_a`, `lsse_r2q_ab`, `lsse_r2q_violence`, …).

---

## 4. Evaluation harness & conventions

| Script | Produces | `--models` |
|---|---|---|
| `eval/xeval.py --n_attack 50` | `eval/outputs/<key>/metrics.json` (legacy 50-prompt) | comma |
| `models/fcf/eval_fullset_all.py` | `fullset_all.json` (frozen full-set ASR) | comma |
| `models/fcf/eval_violence_q16.py [--limit N]` | `violence_q16.json` / `..._smoke.json` (`--limit`) | comma |
| `models/fcf/eval_style_vangogh.py` | `style_vangogh.json` (`style_img2raw` retain) | comma |
| `models/fcf/eval_coco_fid5k.py --tag _lpips` | `coco5k_lpips.json` (COCO-LPIPS, N=100) | comma |
| `eval/lpips_style.py` | adds `style_lpips_f` (forget) over generated VG imgs | — |
| `eval/aggregate_cost.py` | merges per-run `train_cost.json` → `models/fcf/train_cost.json` | — |
| `eval/eval_coherence_triangulate.py` | `coherence_tri.json` (CLIP-indep face/HOG + attack-FID; validates the coherence probe) | comma |
| `eval/eval_label_decomp.py` | `label_decomp.json` (per-image 8-lab vs 4-lab; covered-vs-exposed surplus) | comma |
| `eval/eval_multiseed.py` | `multiseed.json` (Ring-A-Bell ASR/coherence mean±std over seeds; `xeval.generate` now takes `seed_base`) | comma |
| `compare/export_figures.py` | `compare/figures/*.{png,pdf}` (static paper figures from the JSON tables) | — |
| `compare/build_live_gallery.py` | `compare/comparison_gallery_live.html` | — |

- Eval scripts **merge incrementally** (`load_result()` loads existing JSON) — a subset `--models`
  preserves other models. Generation is **resumable** (skips existing PNGs).
- **Q16** = self-contained violence "unsafe" classifier (CLIP ViT-L/14 + learned prompts). Violence
  attacks I2P 757 + Ring-A-Bell 269 ≈ 1026 imgs/model.
- **Commit** the builder + canonical JSON tables + runners. **Do NOT commit** (gitignored, local-only,
  regenerable): `comparison_gallery_live.html`, `eval/outputs/**` images, `eval/outputs/<key>/metrics.json`.
- Gallery model registry: `compare/build_live_gallery.py:MODELS`.

---

## 5. Background jobs & autonomous experiment mode

**Runner convention** (`eval/run_*.sh`): `set -u; cd /mnt/d/...; conda activate lsse` → **smoke gate
first** (real tiny `--limit` run, abort whole job if rc≠0) → each part **non-fatal**, `tee` to `logs/`,
append `*_STATUS` → end with `aggregate_cost` (if cost) + `build_live_gallery.py` → `touch *_DONE`.

**Launch in a WSL `tmux` session** (survives turns, `tee` shows live progress):
`wsl bash -lc 'tmux new-session -d -s <name> "bash eval/run_<x>.sh"'`. Watch with `tmux attach -t
<name>` or `cat <STATUS>`. `*_STATUS`/`*_DONE` are scratch (untracked). One-off jobs may use the Bash
tool with `run_in_background: true` instead; prefer tmux for anything chained or long-lived.

**Autonomous / chained mode ("자동화 명령").** On an automation directive ("자동화", "알아서 진행해",
"큐로 돌려"), do NOT prompt between steps:
1. Define the experiment queue up front (ordered `run_*.sh` list).
2. Run each in its own tmux session (`tee`), one GPU job at a time — chain, never parallelize.
3. Put up a **watcher** as a `run_in_background` Bash task blocking on the DONE sentinel
   (`wsl bash -lc 'until [ -f models/fcf/<X>_DONE ]; do sleep 60; done; echo READY'`, cf. `eval/_watch_*.sh`);
   the harness re-invokes you when it returns.
4. On wake: verify rc in `<STATUS>`, validate results, rebuild gallery, then **auto-launch the next
   queued experiment + watcher**. Repeat until the queue is empty.
5. Stop only when the queue is done, or a smoke gate / `rc≠0` aborts a step (report, don't silently
   continue). Commit between steps only if pre-authorized for the run. Fallback heartbeat:
   `ScheduleWakeup` 1200 s+ (never busy-poll).

---

## 6. Working conventions

- **Language:** reason in English, deliver the final user-facing summary in **Korean**.
- **Git:** commit/push **only when the user asks**. Windows PowerShell git only. No `Co-Authored-By`
  (attribution disabled globally). Conventional commits (`feat:`/`fix:`/`docs:`…). Never push to a
  shared branch unasked.
- **Smoke-test before any long/background job** — run the EXACT command (rc=0 + real output, or a
  built-in smoke gate) before launching or quoting an ETA. Base ETAs on observed throughput.
- **GateGuard hook** fires before the first Bash and every Edit/Write: present (1) the user request,
  (2) what the command/file does, (3) importers/affected symbols + data structure, (4) verbatim
  instruction — then retry the same call.
- **Cost:** the session-cost hook is informational ($, not GPU). Minimize in-context turns on long
  jobs (launch → step away → single harvest turn). `/compact` when context grows large.

---

## 7. Standing research guardrails

- **Multi-concept erasure is ON HOLD** — LSSE-MC breaks utility (COCO-CLIP ~10); ODACE is the only
  utility-preserving multi-concept path so far.
- **Pure text-embedding axis erasure (DACE) is a dead end** vs Sph+OT — prefer output-grounded UNet
  (ODACE) or read-out-space TE (CAP-CNP).
- **Validate any direction/metric on the FULL set**, never an N=10 proxy (the S2 `kv` proxy read 10
  @N=10 but 19.5 full-set ≈ baseline).
- **Read-out-space erasure is not strictly concept-local** — it suppresses a broad "unsafe" direction,
  so expect off-target drops (nudity-trained models also lower violence Q16, and vice-versa).
- **Evaluate with the coherence axis, not ASR alone** — low ASR can be OOD generation collapse, not
  erasure (see §3 caveat + auto-memory [[coherence-validation-honest-sota]]). Honest OOD-coherent winners:
  **SLERP-OT** (4-lab 0.7 / ring 79 / RPG-RT gap 0) and **ODACE benign-anchor** (4-lab 2.1 / ring 100).
  The low-8-lab LSSE flagship (R2q-ab) partly collapses (ring 15). Validation is **DONE** — W2 (face
  r=+0.68), W7 (8-lab surplus = clothed/covered), W3 (multi-seed >30σ). **Paper = spine A** ("Coherence
  Illusion in Concept Erasure"); outline `compare/paper_outline.md`, research doc
  `compare/ood_collapse_pareto.md`, static figs `compare/figures/`.
- **Violence eval RE-SET to paper (2026-07-06), regen pending next session** — prior single-mean
  violence eval DISCARDED. `eval_violence_q16.py` rewired (no-GPU) to FCF Table-1 per-attack Q16 over
  the violence-forgotten models (`raw_v14, lsse_r2q_a_violence, lsse_geo_e2_violence, odace_violence,
  odace_benign_violence, odace_benign_n1_violence, esd_u_violence`). Only 3/5 attack sets on disk
  (I2P/Ring-A-Bell/UnlearnDiffAtk); **P4D & RaB(Re) violence missing** — P4D repo has NO violence set
  (needs P4D optimization vs a violence-safe model, heavy → out of scope, report N/A); RaB(Re)-violence
  = re-run Ring-A-Bell with our violence TE encoder. `sph_ot_violence` EXCLUDED (wrong implicit_groups, ASR 80.4).
  FCF-P/E violence need FCF_upstream `ldm` env. **Next session first command (WSL conda lsse):**
  `python models/fcf/eval_violence_q16.py --limit 3` (smoke) → `python models/fcf/eval_violence_q16.py`
  → `python compare/build_live_gallery.py`. Detail: auto-memory [[violence-eval-paper-alignment]].
- **Next-session GPU queue (NOT started, heavy training):** SD2.1/SDXL generalization of
  push=collapse/redirect=coherent; recent baselines (UCE/RECE/MACE) under the coherence lens. Both need
  new training. No-GPU consolidation (gallery/docs/figures/commits) is complete.
- Detailed numbers, RPG-RT robustness, violence-transfer, and FCF-reproduction evidence: **auto-memory**.

---

## 8. FCF method reference (the paper)

- **Core:** fine-tune the CLIP **text encoder only** (no UNet/VAE).
- **Stage 1 (explicit forgetting):** `L = L_retain + η·L_forget`, MSE on `T_ε(prompt_f)→T_ε*(prompt_n)`
  (forget) and `T_ε(prompt_r)→T_ε*(prompt_r)` (retain).
- **Stage 2a FCF-P (projection):** `cleaned = target_mean − μ_p·proj(target_mean ⟂ concept_norm)`.
- **Stage 2b FCF-E (empirical):** `target = T_ori(group) − μ_e·mean(T_ori(P_explicit) − T_ori(P_noise))`.
- **Hyperparams:** lr=2.5e-5, η=0.25, μ_p=0.7, μ_e=1.0, 60 epochs, batch 4.
- NudeNet v3 labels (`FEMALE_BREAST_EXPOSED` etc.). Paper PDF in `models/fcf/reference/` is
  password-protected — algorithm detail lives in code comments.
