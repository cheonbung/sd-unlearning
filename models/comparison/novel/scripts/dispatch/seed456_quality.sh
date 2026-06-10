#!/bin/bash
set -uo pipefail
source /root/anaconda3/etc/profile.d/conda.sh
conda activate fcf-novel
cd '/mnt/c/Users/vip/Desktop/byeongcheon(server_pc)/SD_unlearning'
mkdir -p models/comparison/novel/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="models/comparison/novel/logs/eval_seed456_quality_${TS}.log"
echo "[DISPATCH] starting seed=456 Quality eval, log=${LOG}"
python evaluate.py   --config models/comparison/novel/configs/nudity_v2_wsl.yaml   --encoder_dir models/comparison/novel/outputs/fcf_p_v2_nudity_sph_ot_seed456/final   --concept nudity   --eval_type quality   --output_dir models/comparison/novel/outputs/eval/sph_ot_seed456_quality   --num_images 30   --seed 42   2>&1 | tee "${LOG}"
ECODE=${PIPESTATUS[0]}
echo "=== DONE_SEED456_QUALITY ts=${TS} exit=${ECODE} ==="
