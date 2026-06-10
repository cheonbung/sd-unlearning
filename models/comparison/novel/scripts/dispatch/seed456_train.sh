#!/bin/bash
set -uo pipefail
source /root/anaconda3/etc/profile.d/conda.sh
conda activate fcf-novel
cd '/mnt/c/Users/vip/Desktop/byeongcheon(server_pc)/SD_unlearning'
mkdir -p models/comparison/novel/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="models/comparison/novel/logs/train_seed456_sph_ot_${TS}.log"
echo "[DISPATCH] starting seed=456 N5+N6 training, log=${LOG}"

# Pre-flight: ensure no leftover fcf_p_v2_nudity dir (would mix with this run)
if [ -d 'models/comparison/novel/outputs/fcf_p_v2_nudity' ]; then
  echo "!!! fcf_p_v2_nudity already exists — aborting to prevent overwrite. Rename it first."
  exit 2
fi

python models/comparison/novel/train.py   --config models/comparison/novel/configs/nudity_v2.yaml   --seed 456   --manifold spherical   --ot_noise_file outputs/ot_noise/nudity_learned.json   2>&1 | tee "${LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}

if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "=== TRAIN_FAILED seed=456 exit=${TRAIN_EXIT} ==="
  exit ${TRAIN_EXIT}
fi

# Rename via PowerShell (Windows-native, bypasses WSL ACL issue from prior session)
powershell.exe -Command "Rename-Item -Path 'c:\Users\vip\Desktop\byeongcheon(server_pc)\SD_unlearning\fcf-novel-methods\outputs\fcf_p_v2_nudity' -NewName 'fcf_p_v2_nudity_sph_ot_seed456' -ErrorAction Stop; Write-Host RENAME_OK"
RENAME_EXIT=$?

echo "=== DONE_TRAIN_SEED456 ts=${TS} train_exit=${TRAIN_EXIT} rename_exit=${RENAME_EXIT} ==="
