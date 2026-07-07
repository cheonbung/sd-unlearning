#!/usr/bin/env bash
# Block until the proxy-calibration job finishes (PROXY_CALIB_DONE sentinel) or its tmux session
# ends, then dump the status + the calibration verdict. Launched via:
# wsl bash eval/_watch_costfill.sh (run_in_background). (Generic watcher, re-pointed per phase.)
cd /mnt/d/unlearning/SD_unlearning
i=0
while [ "$i" -lt 300 ]; do
  i=$((i + 1))
  if [ -f models/fcf/DACE_A_DONE ]; then echo "WATCH_RESULT=DONE iter=$i"; break; fi
  if ! tmux has-session -t daceA 2>/dev/null; then echo "WATCH_RESULT=TMUX_GONE iter=$i"; break; fi
  sleep 30
done
echo "=== final DACE_A_STATUS ==="
tail -30 models/fcf/DACE_A_STATUS 2>/dev/null
echo "=== DACE_A.json verdict ==="
cat models/fcf/DACE_A.json 2>/dev/null
