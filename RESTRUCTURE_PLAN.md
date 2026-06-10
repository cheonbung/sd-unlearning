# RESTRUCTURE RUNBOOK (models/ reorg) — execution spec & recovery doc

Branch: `chore/restructure-models`. Background jobs STOPPED (tmux fcfrepro+phase6 killed).
Goal: `models/{core,comparison}/`, rename `xmodel→eval`, `fcf-novel-methods→novel`,
official FCF = canonical `models/core/fcf`, reimpl archived to `legacy_reimpl/`.

## Directory moves (filesystem `mv`; preserves gitignored outputs/ + checkpoints)
| from | to |
|---|---|
| lsse | models/core/lsse |
| dace | models/core/dace |
| odace | models/comparison/odace |
| fcf-novel-methods | models/comparison/novel |
| baselines/esd | models/comparison/esd |
| baselines/sld | models/comparison/sld |
| baselines/safeclip | models/comparison/safeclip |
| baselines/run_eval.sh | models/comparison/run_eval.sh (then rmdir baselines) |
| xmodel | eval |
| compare/fcf_repro/* | models/core/fcf/ (official scripts, official_fcf_{p,e}/, upstream/, *.json/*.log/*.sh/sentinels) |
| fcf/data, fcf/reference, fcf/Q16, fcf/evaluation | models/core/fcf/ (SHARED — official scripts read data/eval, q16_weights) |
| fcf/{core,train.py,evaluate.py,generate_images.py,configs,docs,analysis,scripts,tests,logs,outputs,*.md,requirements*,inception*.pt,setup_node.sh} | models/core/fcf/legacy_reimpl/ |

NOTE: legacy_reimpl is ARCHIVE-only; its internal imports may not run (acceptable).

## String/path edits (anchored)
- `xmodel/` → `eval/`  (52 refs: docs, scripts). Incl output dir `xmodel/outputs/<m>_fs` → `eval/outputs/<m>_fs`.
- `compare/fcf_repro/` → `models/core/fcf/`  (46 refs).
- `fcf-novel-methods/` → `models/comparison/novel/`.
- `baselines/esd/`→`models/comparison/esd/`, `baselines/sld/`→`models/comparison/sld/`, `baselines/safeclip/`→`models/comparison/safeclip/`.
- REGISTRY te_dir: `lsse/outputs`→`models/core/lsse/outputs`; `dace/outputs`→`models/core/dace/outputs`;
  `odace/outputs`→`models/comparison/odace/outputs`; `fcf-novel-methods/outputs`→`models/comparison/novel/outputs`;
  `baselines/esd/outputs`→`models/comparison/esd/outputs`; raw_v14 attack_dir `lsse/outputs/...` likewise.
- REGISTRY fcf canonical swap: `fcf_p`/`fcf_e` te_dir → official (`models/core/fcf/official_fcf_p/final` / `_e`).
  reimpl te_dir (was `fcf/outputs/fcf_{p,e}_nudity/final`) → `models/core/fcf/legacy_reimpl/outputs/...`
  (or drop fcf_p/fcf_e reimpl entries; keep fcf_p_official/e_official → `models/core/fcf/official_fcf_{p,e}/final`).

## ANCHOR files needing manual edits (depth/import/constants — NOT just sed)
1. `eval/xeval.py` (was xmodel/xeval.py): `LSSE = REPO/"lsse"` → `REPO/"models"/"core"/"lsse"`;
   imports `from baselines.sld.sld_pipeline import ...` + `from baselines.safeclip.safeclip_loader import ...`
   → add `sys.path.insert(0, str(REPO/"models"/"comparison"))` and import `from sld...`, `from safeclip...`;
   REGISTRY paths per above; prompt dir `LSSE/"data/eval"` stays (lsse moved, LSSE const updated).
2. Moved fcf scripts (eval_fullset_all.py, eval_fullset.py, eval_coco_fid5k.py, eval_violence_q16.py,
   rescore_fullset_paperrule.py): `REPO = HERE.parent.parent` (depth2) → `HERE.parents[2]` (depth3 now);
   `sys.path.insert(0,str(REPO/"xmodel"))` → `REPO/"eval"`; `EVAL_DIR=REPO/"fcf"/"data"/"eval"` → `HERE/"data"/"eval"`;
   output `xmodel/outputs/<m>_fs` string → `eval/outputs/<m>_fs`.
3. `compare/build_gallery.py`: ATTACKS prompt files `lsse/data/eval/...` → `models/core/lsse/data/eval/...`;
   MODELS eval_dir (`lsse/outputs/...`,`odace/outputs/...`,`xmodel/outputs/...`) → new paths.
4. `eval/eval_coco.py`, `eval/build_xgallery.py`: REPO/HERE + REGISTRY-derived paths.
5. `models/core/fcf/run_pipeline.sh` + `run_phase6_violence_all.sh`: `R=compare/fcf_repro` → `R=models/core/fcf`;
   `cd /mnt/d/unlearning/SD_unlearning` keep (WSL abs path) ; python entrypoints `$R/eval_*.py`.
6. `scripts/fetch_external.sh`: FCF_DEST `compare/fcf_repro/upstream` → `models/core/fcf/upstream`;
   Q16 prompt path `lsse/evaluation/...` → `models/core/lsse/evaluation/...`.
7. `.gitignore`: `compare/fcf_repro/upstream/` → `models/core/fcf/upstream/`; `xmodel/data/coco/` → `eval/data/coco/`.
8. Docs: REPRODUCE.md, README.md, UNLEARNING_METHODS.md.

## RESUME (after validate) — do NOT re-run Phase4/5 (already rc=0)
- Delete possibly-partial last PNG: `eval/outputs/esd_u_fs/unlearndiffatk/` highest-numbered png.
- SMOKE: `python models/core/fcf/eval_fullset_all.py --models esd_u --limit 1` → must SKIP existing + score OK.
- Resume P3-ext only, then chain phase6, in WSL tmux:
  `tmux new -d -s fcfrepro 'cd /mnt/d/unlearning/SD_unlearning && source ~/miniconda3/etc/profile.d/conda.sh && conda activate lsse && python models/core/fcf/eval_fullset_all.py 2>&1 | tee models/core/fcf/fullset_all.log && touch models/core/fcf/PIPELINE_DONE'`
  `tmux new -d -s phase6 'bash models/core/fcf/run_phase6_violence_all.sh'`

## VALIDATE before resume
- `python -c "import sys;sys.path.insert(0,'eval');import xeval;..."` REGISTRY parent dirs exist.
- Resume image dirs present at `eval/outputs/*_fs`.
- grep zero stale: `(^|["'/ ])(xmodel|compare/fcf_repro|fcf-novel-methods|baselines)/` outside legacy/docs.
