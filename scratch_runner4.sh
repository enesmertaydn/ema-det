#!/bin/bash
# usage: scratch_runner4.sh <name> <tag> <train.py flags...>   (EO data; one seed per process; a crashed process is retried)
cd /c/Users/enesm/Desktop/emadet/emadetforspringer
PY=/c/Users/enesm/venvs/emadet_gpu/Scripts/python.exe
name=$1; tag=$2; shift 2
for s in 42 7 123 2024 31415; do
  for try in 1 2 3; do
    [ -f results/emadet_${tag}_seed${s}.DONE ] && break
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 $PY src/train.py --only final "$@" --seeds $s > results/seedrun_${name}_${s}.log 2>&1
    if grep -q "Test mAP@.5:" results/seedrun_${name}_${s}.log; then touch results/emadet_${tag}_seed${s}.DONE; break; fi
  done
done
touch results/RUN_${name}.DONE
