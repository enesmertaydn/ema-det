#!/bin/bash
# centre-head variant on the Anti-UAV410 thermal data (never used in the design of the head); one seed per process, retried on a crash
cd /c/Users/enesm/Desktop/emadet/emadetforspringer
PY=/c/Users/enesm/venvs/emadet_gpu/Scripts/python.exe
TH=C:/Users/enesm/Desktop/emadet/emadetforspringer/data/antiuav410_clean
OUT=C:/Users/enesm/Desktop/emadet/emadetforspringer/results/thermal
for s in 42 7 123 2024 31415; do
  for try in 1 2 3; do
    [ -f results/thermal/emadet_final_centerhead_seed${s}.DONE ] && break
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 $PY src/train.py --only final --center_head --dataset_root $TH --save_dir $OUT --seeds $s > results/thermal/seedrun_center_${s}.log 2>&1
    if grep -q "Test mAP@.5:" results/thermal/seedrun_center_${s}.log; then touch results/thermal/emadet_final_centerhead_seed${s}.DONE; break; fi
  done
done
touch results/thermal/CENTER_THERMAL.DONE
