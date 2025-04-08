#!/bin/bash

source /envs/fast/bin/activate
if [ "$H2O" = "1" ]; then
  source /envs/h2o/bin/activate
fi
which python
if [ "$NO_INFER" != "1" ]; then
  python benchmark/long_bench/pred.py --model my_model --cfg "$1" --ws "$2" --task_start_id "$3"
fi
python benchmark/long_bench/eval.py --model my_model --cfg "$1"
