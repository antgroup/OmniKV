#!/bin/bash

# 激活虚拟环境
source /envs/fast/bin/activate
conda deactivate
if [ "$H2O" = "1" ]; then
  source /envs/h2o/bin/activate
fi
which python
which accelerate


tasks=(
  "longbook_qa_eng"
  "longbook_choice_eng"
  "longdialogue_qa_eng"
  "longbook_sum_eng"
  "longbook_qa_chn"
  "code_debug"
  "math_find"
  "passkey"
  "number_string"
  "kv_retrieval"
)
if [ "$2" != "" ]; then
  tasks=("$2")
fi
echo "Tasks to be test == ${tasks[@]}"

for task in "${tasks[@]}"; do
    if [ "$NO_INFER" != "1" ]; then
      python benchmark/infinite_bench/src/eval_any_my_model.py \
        --task "$task" \
        --data_dir "/input/fangshuai.fs/huggingface/repo/InfiniteBench" \
        --output_dir "benchmark/infinite_bench/results/eff_test/$1" \
        --config_path "$1" \
        --stop_idx 20
    fi
done

python benchmark/infinite_bench/src/compute_scores.py \
  --task "all" \
  --data_dir "/input/fangshuai.fs/huggingface/repo/InfiniteBench" \
  --output_dir "benchmark/infinite_bench/results/eff_test/$1" \
  --config_path "$1"


