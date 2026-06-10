#!/bin/bash
set -uo pipefail
source /root/anaconda3/etc/profile.d/conda.sh
conda activate fcf-novel
cd '/mnt/c/Users/vip/Desktop/byeongcheon(server_pc)/SD_unlearning'
mkdir -p models/novel/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="models/novel/logs/eval_seed123_quality_${TS}.log"
echo "[DISPATCH] starting seed=123 Quality eval, log=${LOG}"
python evaluate.py   --config models/novel/configs/nudity_v2_wsl.yaml   --encoder_dir models/novel/outputs/fcf_p_v2_nudity_sph_ot_seed123/final   --concept nudity   --eval_type quality   --output_dir models/novel/outputs/eval/sph_ot_seed123_quality   --num_images 30   --seed 42   2>&1 | tee "${LOG}"
ECODE=${PIPESTATUS[0]}
echo "=== DONE_SEED123_QUALITY ts=${TS} exit=${ECODE} ==="
