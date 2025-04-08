import os
import subprocess
from multiprocessing import Pool, Semaphore, current_process
import time
from tiny_tools.log import logger

# 配置文件列表
configs = [
    # "0824.llama-70B.raw.for_long.json",
    # "0825.llama-70B.raw.cot.json",
    # "0826.llama70b.h2o.cot.json",
    # "0904.llama70b.h2o.cot.30.json",
    # "0826.llama70b.h2o.cot.40.json",
    "0908.llama70b.sink.cot.20.json",
    "0908.llama70b.sink.cot.30.json",
    "0908.llama70b.sink.cot.40.json",
    # "0827.llama70b.omnikv.cot.json",
    # "0904.llama70b.omnikv.cot.30.json",
    # "0904.llama70b.omnikv.cot.40.json",
    # "0912.70b.omnikv.cot.20.uni.json",
    # "0912.70b.omnikv.cot.30.uni.json",
    # "0912.70b.omnikv.cot.40.uni.json",
    # "0912.70b.omnikv.cot.20.exp.json",
    # "0912.70b.omnikv.cot.30.exp.json",
    # "0912.70b.omnikv.cot.40.exp.json",
    "0826.yi.raw.cot.json",
    "0826.yi.omnikv.cot.json",
    "0826.yi.omnikv.cot.40.json",
    "0826.yi.omnikv.cot.50.json",
    "0904.yi.h2o.cot.30.json",
    "0904.yi.h2o.cot.40.json",
    "0904.yi.h2o.cot.50.json",
]

# 获取GPU数量
gpu_count = 4

# 初始化信号量
semaphores = [Semaphore(1) for _ in range(gpu_count)]


def run_task(cfg):
    gpu_id = None
    for i in range(len(semaphores)):
        if semaphores[i].acquire(False):
            gpu_id = i
            break
    if gpu_id is None:
        logger.info(f"No free GPU available for {cfg}, waiting...")
        gpu_id = semaphores.index(1)
        semaphores[gpu_id].acquire()

    logger.info(f"Process {current_process().name} using GPU {gpu_id}")
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python benchmark/long_bench/cot_pred.py --cfg_path configs/tests/{cfg} --task 2stage --verbose 1"
    if "h2o" in cfg:
        cmd = "source /envs/h2o/bin/activate; which python; " + cmd
    else:
        cmd = "source /envs/fast/bin/activate;  which python; " + cmd
    subprocess.run(cmd, shell=True)
    print(f"{cfg} is done ------")
    logger.info(f"------ {cfg} is done ------")

    # Release the semaphore
    semaphores[gpu_id].release()


if __name__ == "__main__":
    pool = Pool(processes=gpu_count)
    pool.map(run_task, configs)
    pool.close()
    pool.join()
    print("All tasks completed.")
