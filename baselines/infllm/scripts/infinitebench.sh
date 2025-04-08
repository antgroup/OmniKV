#!/bin/bash

config=baselines/infllm/config/llama-3-inf-llm-pro.yaml
model=""
if [ "$2" != "" ]; then
  config="$2"
  model="yi"
fi
echo "config is $config"

conda deactivate
source /envs/fast/bin/activate
mkdir "baselines/infllm/benchmark/infinite-bench-result$model"

python baselines/infllm/benchmark/pred.py \
--config_path ${config} \
--output_dir_path baselines/infllm/benchmark/infinite-bench-result \
--datasets "$1" \
--verbose
#--datasets kv_retrieval,passkey,number_string,code_debug,math_find,longbook_choice_eng

python baselines/infllm/benchmark/eval.py --dir_path baselines/infllm/benchmark/infinite-bench-result