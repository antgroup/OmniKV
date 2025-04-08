#!/bin/bash

if [ "$NO_INFER" != "1" ]; then
  python benchmark/long_bench/pred.py --model my_model --cfg "$1" --ws "$2"
fi
python benchmark/long_bench/eval.py --model my_model --cfg "$1"
