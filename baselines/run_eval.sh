#!/usr/bin/env bash
# Full reproduced-baseline evaluation: ASR (xeval) then COCO (eval_coco) for the 5 reproduced
# reference baselines. Run inside tmux with a live tee pane:
#   tmux new-session -d -s baseline_eval 'bash baselines/run_eval.sh'
#   tmux attach -t baseline_eval        # watch live
# generate() skips existing images, so this safely resumes a partially-done run. The final
# marker below lands IN the tee'd log so a watcher can detect completion from the log file.
set -uo pipefail
cd /mnt/d/unlearning/SD_unlearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/miniconda3/envs/lsse/bin/python
MODELS=esd_u,sld_medium,sld_strong,sld_max,safeclip
{
  "$PY" xmodel/xeval.py --models "$MODELS" --n_attack 50 \
  && "$PY" xmodel/eval_coco.py --models "$MODELS" \
  && echo "ALL_BASELINE_EVAL_DONE_OK" \
  || echo "BASELINE_EVAL_FAILED rc=$?"
} 2>&1 | tee xmodel/outputs/eval_baselines.log
