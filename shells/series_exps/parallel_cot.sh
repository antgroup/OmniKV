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
"2wikimqa" "hotpotqa" "2stage"
)

echo "Tasks to be test == ${tasks[@]}"

# 获取GPU编号列表，并确定一共有多少个GPU
IFS=',' read -r -a gpu_list <<< "$2"
num_gpus=${#gpu_list[@]}

# 任务计数器，用来标识使用哪个 GPU
task_counter=0
# 用来存储每个子任务的 PID
pids=()

for task in "${tasks[@]}"; do
    # 设置任务使用的 GPU
    gpu_id=${gpu_list[$((task_counter % num_gpus))]}
    if [ "$NO_INFER" != "1" ]; then
      CUDA_VISIBLE_DEVICES=$gpu_id python benchmark/long_bench/cot_pred.py \
        --task "$task" \
        --cfg_path "$1" \
        --verbose 1 &

      # 获取进程的 PID 并保存
      pids+=($!)
    fi
    # 每个任务计数器加一
    task_counter=$((task_counter + 1))
    # 如果任务计数器超过 num_gpus，则等待相应的前一个任务完成
    if [ $task_counter -ge $num_gpus ]; then
      wait ${pids[$((task_counter - num_gpus))]}
    fi
    sleep 1
done

# 等待剩余的所有后台任务完成
for pid in "${pids[@]}"; do
  wait $pid
done
