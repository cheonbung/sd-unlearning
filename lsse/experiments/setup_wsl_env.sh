#!/usr/bin/env bash
# Idempotent WSL conda env setup for LSSE experiments.
# Logs to setup.log; writes setup.DONE or setup.FAIL marker at the end.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/setup.log"
rm -f "$HERE/setup.DONE" "$HERE/setup.FAIL"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

{
  log "=== LSSE WSL env setup start ==="

  # 1. Miniconda
  if [ ! -d "$HOME/miniconda3" ]; then
    log "Installing miniconda3 ..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
      || { log "FAIL: miniconda download"; touch "$HERE/setup.FAIL"; exit 1; }
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3" \
      || { log "FAIL: miniconda install"; touch "$HERE/setup.FAIL"; exit 1; }
  else
    log "miniconda3 already present"
  fi

  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  # 기본 채널 ToS 회피: conda-forge만 사용
  conda config --system --set channel_priority flexible >>"$LOG" 2>&1 || true

  # 2. Env (conda-forge override로 defaults 채널 ToS 우회)
  if ! conda env list | grep -q "/envs/lsse"; then
    log "Creating conda env 'lsse' (python 3.10, conda-forge) ..."
    conda create -y -n lsse python=3.10 -c conda-forge --override-channels >>"$LOG" 2>&1 \
      || { log "FAIL: conda create"; touch "$HERE/setup.FAIL"; exit 1; }
  else
    log "env 'lsse' already exists"
  fi
  # 활성화 대신 env 절대경로 python 사용 (set -u + conda activate 충돌 회피)
  PY="$HOME/miniconda3/envs/lsse/bin/python"
  log "Bootstrapping pip (ensurepip) ..."
  "$PY" -m ensurepip --upgrade >>"$LOG" 2>&1 \
    || conda install -y -n lsse -c conda-forge --override-channels pip >>"$LOG" 2>&1 \
    || { log "FAIL: pip bootstrap"; touch "$HERE/setup.FAIL"; exit 1; }
  "$PY" -m pip install --quiet --upgrade pip >>"$LOG" 2>&1 || true

  # 3. Torch (CUDA 12.1 build for RTX 4070)
  if ! "$PY" -c "import torch" 2>/dev/null; then
    log "Installing torch+cu121 ..."
    "$PY" -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121 >>"$LOG" 2>&1 \
      || { log "FAIL: torch install"; touch "$HERE/setup.FAIL"; exit 1; }
  else
    log "torch already installed"
  fi

  # 4. Rest of deps
  log "Installing remaining deps ..."
  "$PY" -m pip install --quiet \
    "diffusers>=0.21.0" "transformers>=4.31.0" "accelerate>=0.21.0" \
    "safetensors>=0.3.1" "Pillow>=9.0.0" "numpy>=1.24.0" "tqdm>=4.65.0" \
    "PyYAML>=6.0" "pandas>=1.5.0" "lpips>=0.1.4" "clean-fid>=0.1.35" \
    "torchmetrics>=0.11.0" "nudenet>=3.4.2" "huggingface_hub>=0.16.0" \
    "pytest>=7.0.0" "pytest-cov>=4.0.0" >>"$LOG" 2>&1 \
    || { log "FAIL: deps install"; touch "$HERE/setup.FAIL"; exit 1; }

  # 5. Verify
  log "Verifying torch CUDA ..."
  "$PY" - <<'PY' >>"$LOG" 2>&1
import torch, transformers, diffusers
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("transformers", transformers.__version__, "diffusers", diffusers.__version__)
PY
  CUDA_OK=$("$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
  if [ "$CUDA_OK" = "True" ]; then
    log "=== SETUP DONE (CUDA available) ==="
    touch "$HERE/setup.DONE"
  else
    log "FAIL: CUDA not available after install"
    touch "$HERE/setup.FAIL"
    exit 1
  fi
} 2>&1 | tee -a "$LOG"
