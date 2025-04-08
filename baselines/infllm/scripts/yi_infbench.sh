#!/bin/bash

config=baselines/infllm/config/yi-9b-inf.yaml

conda deactivate
source /envs/fast/bin/activate
mkdir baselines/infllm/benchmark/infinite-bench-result

python baselines/infllm/benchmark/pred.py \
--config_path ${config} \
--output_dir_path baselines/infllm/benchmark/yi-infinite-bench-result \
--datasets "$1" \
--verbose
#--datasets kv_retrieval,passkey,number_string,code_debug,math_find,longbook_choice_eng

python baselines/infllm/benchmark/eval.py --dir_path baselines/infllm/benchmark/infinite-bench-result