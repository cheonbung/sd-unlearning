#!/bin/bash
set -uo pipefail
source /root/anaconda3/etc/profile.d/conda.sh
conda activate fcf-novel
cd '/mnt/c/Users/vip/Desktop/byeongcheon(server_pc)/SD_unlearning'
mkdir -p models/comparison/novel/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="models/comparison/novel/logs/eval_seed123_asr_${TS}.log"
echo "[DISPATCH] starting seed=123 ASR eval, log=${LOG}"
python evaluate.py   --encoder_dir models/comparison/novel/outputs/fcf_p_v2_nudity_sph_ot_seed123/final   --concept nudity   --eval_type asr   --output_dir models/comparison/novel/outputs/eval/sph_ot_seed123_asr   --num_images 50   --seed 42   2>&1 | tee "${LOG}"
ECODE=${PIPESTATUS[0]}
echo "=== DONE_SEED123_ASR ts=${TS} exit=${ECODE} ==="
