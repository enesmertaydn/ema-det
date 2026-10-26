#!/bin/bash
# centre-head variant: one seed per process (seed 42 was run as the pilot); a crashed process is retried
cd /c/Users/enesm/Desktop/emadet/emadetforspringer
PY=/c/Users/enesm/venvs/emadet_gpu/Scripts/python.exe
touch results/emadet_final_centerhead_seed42.DONE
for s in 7 123 2024 31415; do
  for try in 1 2 3; do
    [ -f results/emadet_final_centerhead_seed${s}.DONE ] && break
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 $PY src/train.py --only final --center_head --seeds $s > results/seedrun_center_${s}.log 2>&1
    if grep -q "Test mAP@.5:" results/seedrun_center_${s}.log; then touch results/emadet_final_centerhead_seed${s}.DONE; break; fi
  done
done
touch results/CENTER_SEEDRUNS.DONE
