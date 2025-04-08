import os
import time

from infer import get_any_chat_api
from tiny_tools.read_json import read_config
from tiny_tools.tensor_tools import read_idx_tracer, idx_tracer
from tiny_tools.log import create_logger
from fire import Fire
import pickle
import json
from transformers import GenerationConfig
import matplotlib.pyplot as plt
import torch


def read_data(path='benchmark/long_bench/data/2wikimqa.jsonl', model='llama', sort=None):
    if model == 'llama':
        std_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Evidence (step by step) then answer: 1."""

    elif model == 'yi':
        std_template = """{system_prompt} But you will give your final answer in less than 80 words.
{user_message}"""

    elif model == 'yi34b':
        std_template = """
{user_message}
Chain-of-Reasoning answer to ({input}): 
Step 1:"""

    elif model == 'llama70b':
        std_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Evidence (step by step) then answer: 1."""
    else:
        raise NotImplementedError

    sys_template = 'You are a helpful assistant who thinks step by step.'
    prompt_template = """Context:\n{context}\n
Question: {input}\n
This is a multi-hop question, so answer it step by step.\n"""

    data = []
    with open(path, 'r', encoding='utf-8') as _in:
        for line in _in:
            d = json.loads(line)
            user_msg = prompt_template.format(**d)
            prompt = std_template.format(system_prompt=sys_template, user_message=user_msg, **d)
            # print(prompt[-100:])
            data.append({"prompt": prompt, 'd': d})
            data[-1].update({"len": len(data[-1]['prompt'])})
    if sort:
        data = sorted(data, key=lambda x: x['len'], reverse=True)
    return data


def main(cfg_path, sample_num=10, topk=2048, sort=None, task=None):
    logger = create_logger('record', f'debug_logs/{os.environ["SAVE_SELECTED_IDX"]}.log')

    chat, tok, max_len, o_dict = get_any_chat_api(cfg_path)
    model_name = 'llama'
    if 'yi' in cfg_path:
        model_name = 'yi'
        if '34' in cfg_path:
            model_name = 'yi34b'
    elif '70' in cfg_path:
        model_name = 'llama70b'
    kwargs = {}
    if task is not None:
        kwargs['path'] = f"benchmark/long_bench/data/{task}.jsonl"
    data = read_data(model=model_name, sort=sort, **kwargs)
    sample_num = min(sample_num, len(data))
    data = data[:sample_num]

    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=200)
    if model_name == 'yi':
        gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=50)
    acc = 0
    for dd in data:
        p, dt = dd['prompt'], dd['d']
        answers = dt['answers']
        out = chat(p, gen_cfg)
        print('----', dt['input'], '|||', answers, '|||\n', out)
        logger.info(f'Question: {dt["input"]}\nStd Answer: {answers}\nLLM out: {out}')
        if answers[0].lower() in out.lower():
            acc += 1
        idx_tracer.save_idx()

    acc /= len(data)
    print("ACC(in)", acc)


if __name__ == '__main__':
    Fire(main)
