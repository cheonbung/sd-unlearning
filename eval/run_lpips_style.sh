#!/usr/bin/env bash
# (b) Paper-aligned art-style LPIPS_f, queued BEHIND the fillcells job (shares the single GPU).
# Waits for models/fcf/FILLCELLS_DONE, then runs the post-hoc LPIPS pass over the already-generated
# Van Gogh images (NO SD regeneration) and rebuilds the gallery so the VanGogh LPIPS_f column + the
# (a) COCO-LPIPS column both render. tmux: tmux new-session -d -s lpipsstyle 'bash eval/run_lpips_style.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs

ST=models/fcf/LPIPSSTYLE_STATUS
rm -f models/fcf/LPIPSSTYLE_DONE
echo "=== LPIPS-STYLE START $(date) ===" | tee "$ST"

# queue behind the heavy fillcells job (MC full-set generation)
if tmux has-session -t fillcells 2>/dev/null && [ ! -f models/fcf/FILLCELLS_DONE ]; then
  echo "=== waiting for fillcells (models/fcf/FILLCELLS_DONE) $(date) ===" | tee -a "$ST"
  while tmux has-session -t fillcells 2>/dev/null && [ ! -f models/fcf/FILLCELLS_DONE ]; do sleep 120; done
  echo "=== fillcells cleared $(date) ===" | tee -a "$ST"
fi

echo "=== LPIPS_f over retained Van Gogh images $(date) ===" | tee -a "$ST"
python eval/lpips_style.py 2>&1 | tee logs/lpips_style.log
RC=${PIPESTATUS[0]}
echo "=== lpips_style.py END $(date) (rc=$RC) ===" | tee -a "$ST"

echo "=== rebuild gallery (adds COCO-LPIPS + VanGogh LPIPS_f columns) $(date) ===" | tee -a "$ST"
python compare/build_live_gallery.py 2>&1 | tee logs/lpips_style_gallery.log

echo "=== LPIPS-STYLE DONE $(date) ===" | tee -a "$ST"
echo "NEXT (WINDOWS git): git add eval/lpips_style.py eval/run_lpips_style.sh compare/build_live_gallery.py models/fcf/style_vangogh.json ; commit ; push" | tee -a "$ST"
touch models/fcf/LPIPSSTYLE_DONE
