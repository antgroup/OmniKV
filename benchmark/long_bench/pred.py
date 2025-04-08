import os
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from tiny_tools.read_json import read_config
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, GenerationConfig
import torch.distributed as dist
from infer import get_any_chat_api
from tiny_tools.tensor_tools import idx_tracer


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k",
                                 "vicuna-v1.5-7b-16k",
                                 "my_model"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--ws", default=2, type=int, help='world size')
    parser.add_argument("--task_start_id", default=0, type=int)
    parser.add_argument("--task", default=None, type=str)
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name or 'llama-2' in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path,
             out_path, args):
    seed_everything(42)
    d_cfg = read_config(args.cfg)
    device = 0
    # x = torch.rand(100_000, 100_000, device=0)
    # device_num = torch.cuda.device_count()
    # if d_cfg.get('use_multi_gpus', False):
    #     device = rank % device_num
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    #     device = 0
    #     assert torch.cuda.device_count() == 1

    model, tokenizer, model_max_length = load_model_and_tokenizer(model2path[model_name], model_name, device, args.cfg)
    if model_max_length is not None:
        max_length = model_max_length
        print(f"max_length is set to {max_length}")

    for json_obj in tqdm(data, desc=f'{dataset}'):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle,
        # since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = (tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                      tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                           "repobench-p"]:  # chat models are better off without build prompts on these tasks
            if 'my_model' not in model_name:
                prompt = build_chat(tokenizer, prompt, model_name)
            else:
                prompt = build_chat(tokenizer, prompt, d_cfg['model_name'])

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            if "my_model" not in model_name:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model(
                    prompt, generation_config=None,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id,
                                  tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    skip_special_tokens=True)
        else:
            if "my_model" not in model_name:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            else:
                output = model(
                    prompt, generation_config=None,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    skip_special_tokens=True
                )
        if not isinstance(output, str):
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            pred = output

        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                       "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

        # 用来分析模型性质的code:::
        if os.environ.get('SAVE_SELECTED_IDX', False):
            idx_tracer.save_idx()
            # 提前终止
            if idx_tracer.num_samples > 20:
                return


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, cfg):
    max_length = None
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
            device)
    if 'my_model' in model_name:
        model, tokenizer, max_length, other_kwargs = get_any_chat_api(cfg)
        tokenizer.eos_token_id = other_kwargs['eos_token_id']
        print("EOS is", tokenizer.eos_token_id)
    elif "llama2" in model_name:
        # replace_llama_attn_with_flash_attn()
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    # model = model.eval_dir()
    return model, tokenizer, max_length


def load_dataset(path, mode='r'):
    data = [json.loads(line) for line in open(path, mode, encoding="utf-8")]
    return data


class CustomProcess(Process):
    def __init__(self, env_var_key, env_var_value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_var_key = env_var_key
        self.env_var_value = env_var_value
        os.environ[self.env_var_key] = self.env_var_value

    def run(self):
        # 在子进程中设置环境变量
        os.environ[self.env_var_key] = self.env_var_value
        # 调用原始的进程执行目标函数
        super().run()


if __name__ == '__main__':
    args = parse_args()
    world_size = args.ws
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("benchmark/long_bench/config/model2path.json", "r"))
    model2maxlen = json.load(open("benchmark/long_bench/config/model2maxlen.json", "r"))
    model_name = args.model
    # define your model
    max_length = model2maxlen.get(model_name, -1)
    d_cfg = read_config(args.cfg)
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
            "dureader", "gov_report", "qmsum",
            # "multi_news", # 找不到
            "vcsum", "trec", "triviaqa",
            # "samsum", # 找不到
            "lsht",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
        ]
    datasets = datasets[args.task_start_id:]
    if args.task is not None:
        datasets = args.task.split(',')
        # print("for debug", datasets)
    # we design specific prompt format and max generation length for each task,
    # feel free to modify them to optimize model output
    dataset2prompt = json.load(open("benchmark/long_bench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/long_bench/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    # if not os.path.exists("pred"):
    #     os.makedirs("pred")
    # if not os.path.exists("pred_e"):
    #     os.makedirs("pred_e")
    base_path = ''
    if os.environ.get("NO_NAS", False):
        base_path = '/jitai/'
    for dataset in datasets:
        if args.e:
            data = load_dataset(f'benchmark/long_bench/data/{dataset}_e.jsonl', 'r')
            if not os.path.exists(f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}"):
                os.makedirs(f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}", exist_ok=True)
            out_path = f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}/{dataset}.jsonl"
        else:
            data = load_dataset(f'benchmark/long_bench/data/{dataset}.jsonl', 'r')
            if not os.path.exists(f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}"):
                os.makedirs(f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}", exist_ok=True)
            out_path = f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}/{dataset}.jsonl"
        with open(out_path, 'w') as _in:
            pass  # 清空里面的内容
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        # data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        torch.cuda.empty_cache()
        get_pred(0, world_size, data_all, max_length,
                 max_gen, prompt_format, dataset, None, model_name, model2path,
                 out_path, args)
        # 去掉多进程
        # for rank in range(world_size):
        #     # if d_cfg.get('use_multi_gpus', False):
        #     #     p = CustomProcess(target=get_pred,
        #     #                       args=(rank, world_size, data_subsets[rank], max_length,
        #     #                             max_gen, prompt_format, dataset, None, model_name, model2path,
        #     #                             out_path, args),
        #     #                       env_var_key="CUDA_VISIBLE_DEVICES",
        #     #                       env_var_value=f"{rank % torch.cuda.device_count()}")
        #     # else:
        #     #     p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length,
        #     #                                           max_gen, prompt_format, dataset, None, model_name, model2path,
        #     #                                           out_path, args))
        #
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
