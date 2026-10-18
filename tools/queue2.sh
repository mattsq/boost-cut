#!/bin/bash
# Runs queued jobs one at a time, in order. Each line of queue.txt: name|ngpu|args|script. ngpu "py" runs a
# one-GPU analysis script (modal_app.py::py); a number runs torchrun on that many GPUs (modal_app.py::train).
cd "$(dirname "$0")"
while true; do
  line=$(head -n 1 queue.txt 2>/dev/null)
  if [ -z "$line" ]; then sleep 30; continue; fi
  tail -n +2 queue.txt > queue.tmp && mv queue.tmp queue.txt
  name=${line%%|*}; rest=${line#*|}; ngpu=${rest%%|*}; rest=${rest#*|}
  args=${rest%%|*}; script=${rest#*|}
  echo "$(date '+%H:%M:%S') start $name" >> queue.log
  if [ "$ngpu" == "cpu" ]; then
    modal run --detach modal_app.py::cpu --name "$name" --script "$script" --args "$args" > "localruns/$name.out" 2>&1
  elif [ "$ngpu" == "py" ]; then
    modal run --detach modal_app.py::py --name "$name" --script "$script" --args "$args" > "localruns/$name.out" 2>&1
  else
    modal run --detach modal_app.py::train --name "$name" --ngpu "$ngpu" --script "$script" --args "$args" > "localruns/$name.out" 2>&1
  fi
  echo "$(date '+%H:%M:%S') end $name exit=$?" >> queue.log
done
