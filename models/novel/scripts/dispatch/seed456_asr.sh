#!/bin/bash
set -uo pipefail
source /root/anaconda3/etc/profile.d/conda.sh
conda activate fcf-novel
cd '/mnt/c/Users/vip/Desktop/byeongcheon(server_pc)/SD_unlearning'
mkdir -p models/novel/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="models/novel/logs/eval_seed456_asr_${TS}.log"
echo "[DISPATCH] starting seed=456 ASR eval, log=${LOG}"
python evaluate.py   --encoder_dir models/novel/outputs/fcf_p_v2_nudity_sph_ot_seed456/final   --concept nudity   --eval_type asr   --output_dir models/novel/outputs/eval/sph_ot_seed456_asr   --num_images 50   --seed 42   2>&1 | tee "${LOG}"
ECODE=${PIPESTATUS[0]}
echo "=== DONE_SEED456_ASR ts=${TS} exit=${ECODE} ==="
