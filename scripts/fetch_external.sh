#!/usr/bin/env bash
# Fetch external code/data that the repo does NOT vendor, for full from-scratch reproduction.
# Portable: derives the repo root from this script's location (no hardcoded machine paths).
#
#   1. FCF upstream  -> compare/fcf_repro/upstream/   (authors' official code + nudity/violence CSV)
#      Cloned at a PINNED commit. Upstream has NO LICENSE, so we do not redistribute it inside this
#      repo; each user fetches it directly from the authors. Needed only to reproduce the FAITHFUL
#      FCF-P/E checkpoints (FCF-P full-set 4-label 3.7 ~= paper 3.43). The in-repo fcf/ is a separate
#      from-scratch re-implementation (~52, documented as the unfaithful variant).
#   2. Ring-A-Bell concept vectors (Nudity_vector.npy / VanGogh_vector.npy) used to (re)generate the
#      Ring-A-Bell attack prompts -> via the existing fcf/scripts/download_*.py.
#   3. Q16 prompts.p is VENDORED in-repo (MIT) -> only re-fetched here if missing.
#   COCO (eval_coco*) and Hugging Face model weights download automatically on first eval/train.
#
# Usage:  bash scripts/fetch_external.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python}"

FCF_UPSTREAM_URL="https://github.com/f-c-forgetting/FCF"
FCF_UPSTREAM_COMMIT="e65e96a42f8562833c86d87cb4c8d3d6e35b378d"   # 2024-10-28, used for our results
FCF_DEST="compare/fcf_repro/upstream"

echo "== [1/3] FCF upstream (official authors' code) =="
if [ -d "$FCF_DEST/.git" ]; then
  echo "   already present at $FCF_DEST (skip)"
else
  git clone "$FCF_UPSTREAM_URL" "$FCF_DEST"
  git -C "$FCF_DEST" checkout "$FCF_UPSTREAM_COMMIT"
  echo "   cloned $FCF_UPSTREAM_URL @ $FCF_UPSTREAM_COMMIT -> $FCF_DEST"
fi
echo "   training data: $FCF_DEST/data/train/{nudity,violence}.csv (sentence triplets prompt_f/n/r)"

echo "== [2/3] Ring-A-Bell concept vectors (for RaB attack generation) =="
if [ -f "fcf/data/eval/Nudity_vector.npy" ]; then
  echo "   Nudity_vector.npy already present (skip)"
else
  "$PY" fcf/scripts/download_nudity_vector.py || \
    echo "   WARN: download failed; see fcf/scripts/download_rab_files.py (source: chiayi-hsu/Ring-A-Bell)"
fi

echo "== [3/3] Q16 prompts.p (MIT, vendored in-repo) =="
if [ -f "lsse/evaluation/q16_weights/prompts.p" ]; then
  echo "   present (vendored). OK."
else
  echo "   MISSING. Fetch from ml-research/Q16 (MIT):"
  echo "     git clone --depth=1 https://github.com/ml-research/Q16 /tmp/Q16"
  echo "     cp /tmp/Q16/data/ViT-L-14/prompts.p lsse/evaluation/q16_weights/prompts.p"
fi

echo "Done. COCO + Hugging Face weights download on first eval/train run."
echo "Gated/moved HF ids (SD2.1-base, SD1.5) fall back to public mirrors in xmodel/xeval.py:REGISTRY."
