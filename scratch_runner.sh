#!/bin/bash
# one GPU job at a time; each seed in its own process, retried once if it crashes (CUDA illegal-access faults)
cd /c/Users/enesm/Desktop/emadet/emadetforspringer
PY=/c/Users/enesm/venvs/emadet_gpu/Scripts/python.exe
run() { # name tag flags...
  name=$1; tag=$2; shift 2
  for s in 42 7 123 2024 31415; do
    for try in 1 2 3; do
      if [ -f results/emadet_${tag}_seed${s}.DONE ]; then break; fi
      PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 $PY src/train.py --only final "$@" --seeds $s > results/seedrun_${name}_${s}.log 2>&1
      if grep -q "Test mAP@.5:" results/seedrun_${name}_${s}.log; then touch results/emadet_${tag}_seed${s}.DONE; break; fi
    done
  done
}
run dsdeep final_dsdeep --ds_deep
run fusion final_fusionbbox --fusion_bbox
run c6 final_c6128 --c6 128
touch results/ALL_SEEDRUNS.DONE
