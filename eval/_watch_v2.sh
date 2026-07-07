#!/usr/bin/env bash
# Throwaway completion watcher for the improvev2 tmux job. Blocks until the IMPROVE_V2_DONE sentinel
# appears (or a ~19h safety cap), then prints the final status tail. Run in background via:
#   wsl bash /mnt/d/unlearning/SD_unlearning/eval/_watch_v2.sh
cd /mnt/d/unlearning/SD_unlearning || exit 1
i=0
while [ ! -f models/fcf/IMPROVE_V2_DONE ] && [ "$i" -lt 230 ]; do
  sleep 300
  i=$((i + 1))
done
if [ -f models/fcf/IMPROVE_V2_DONE ]; then
  echo DONE_DETECTED
else
  echo WATCHER_TIMEOUT
fi
tail -60 models/fcf/IMPROVE_V2_STATUS
