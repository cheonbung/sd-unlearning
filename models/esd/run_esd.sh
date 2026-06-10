#!/usr/bin/env bash
# Launch canonical ESD-u nudity training (SD v1.4). Run inside tmux with tee for a live pane:
#   tmux new-session -d -s esd_train 'bash models/esd/run_esd.sh'
# Writes a standalone UNET -> models/esd/outputs/esd_u/final (xeval kind="esd").
set -euo pipefail
cd /mnt/d/unlearning/SD_unlearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on the 12GB card
PY=/home/user/miniconda3/envs/lsse/bin/python
mkdir -p models/esd/outputs
"$PY" models/esd/train_esd.py \
    --config models/esd/configs/nudity_esd_u.yaml \
    2>&1 | tee models/esd/outputs/esd_u_train.log
echo "ESD_TRAIN_DONE_MARKER exit=${PIPESTATUS[0]}"
