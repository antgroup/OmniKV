import os
import re
import yaml
from infer import get_any_chat_api
from fire import Fire
from transformers import GenerationConfig
import json
from tiny_tools.log import logger_file_path
from tqdm import trange

last_line = -1


def gen_temp_cfg(**kwargs):
    with open("configs/temp_latency.json", "w", encoding="utf-8") as _out:
        json.dump(kwargs, _out, indent=4)

    return "configs/temp_latency.json"


def run(
    cpu_num_threads=12,
    context_len=128,
    offload=True,
    model_cls="multi",
    load_in_4bit=False,
    num_sel_tokens=0.016,
    h2o_ratio=0.3,
    h2o_seg_sz="auto",
    auto=False,
    model_name=None,
    cache_cls="multi",
    use_multi_gpus=False,
    cfg_path=None,
):
    if auto:
        print("this is a latency test...")
        if auto == "70B":
            for ctl in [150, 170, 200]:
                run(
                    context_len=ctl,
                    model_cls="multi",
                    num_sel_tokens=2048,
                    model_name="70B",
                )
        if auto == "trade_off":
            for num in [128, 256, 512, 1024, 2048, 4096, 6400]:
                run(context_len=128, model_cls="multi", num_sel_tokens=num)
        if auto == "wo_pack" or auto == "all":
            for ctl in [4, 8, 16, 32, 64, 128, 256, 400, 450]:
                run(
                    context_len=ctl,
                    model_cls="multi",
                    num_sel_tokens=2048,
                    cache_cls="without_pack",
                )
        if auto == "omnikv_wo_offload" or auto == "all":
            for ctl in [4, 8, 16, 32, 64, 128]:
                run(
                    context_len=ctl,
                    model_cls="multi",
                    num_sel_tokens=2048,
                    offload=False,
                )
        if auto == "omnikv_offload" or auto == "all":
            for ctl in [4, 8, 16, 32, 64, 128, 256, 400, 450]:
                run(context_len=ctl, model_cls="multi", num_sel_tokens=2048)
        if auto == "full" or auto == "all":
            for ctl in [4, 8, 16, 32, 64, 128]:
                run(context_len=ctl, model_cls="raw_llama")
        if auto == "full_2gpus" or auto == "all":
            for ctl in [256, 450]:
                run(context_len=ctl, model_cls="raw_llama", use_multi_gpus=True)
        if auto == "h2o":
            for ctl in [4, 8, 16, 32, 64, 128, 256, 400]:
                run(context_len=ctl, model_cls="h2o", h2o_ratio=0.3)
        if auto == "brutal" or auto == "all":
            for ctl in [4, 8, 16, 32, 64, 128, 256, 400, 450]:
                run(context_len=ctl, model_cls="brutal_offload")
        if auto == "infllm" or auto == "all":
            for ctl in reversed([4, 8, 16, 32, 64, 128, 256, 400, 450]):
                run(context_len=ctl, model_cls="infllm")
        return

    global last_line
    context_len *= 1000
    if model_name is None:
        model_name = "/input/jitai/huggingface/hub/Lourdle/Llama-3-8B-Instruct-262k"
    if model_name == "70B":
        model_name = "/input/ocr_data/huggingface/models/Meta-Llama-3___1-70B-Instruct/"
        load_in_4bit = True
    infllm_cfg_path = None
    if model_cls == "infllm":
        # infllm的百分比为，
        # r_topk/block_size+(n_local+n_init)/len+topk*blocksize/len+max_cached*blocksize/len
        # 0.03  +  0  + 0.1*len/blocksize + 0.2*len / blocksize
        # infllm_config = {
        #     "model": {
        #         "type": "inf-llm",
        #         "path": "/input/jitai/huggingface/hub/Lourdle/Llama-3-8B-Instruct-262k",
        #         "block_size": 128,
        #         "fattn": False,
        #         "n_init": 128,
        #         "n_local": 1024,
        #         "topk": int(0.1 * context_len / 128),
        #         "repr_topk": 4,
        #         "max_cached_block": int(0.2 * context_len / 128),
        #         "exc_block_size": 512,
        #         "base": 283461213,
        #         "distance_scale": 1.0
        #     },
        #     "max_len": 128000000,
        #     "chunk_size": 8192,
        #     "conv_type": "llama-3-inst",
        #     "truncation": "middle"
        # }
        # infllm_cfg_path = 'configs/temp_infllm_latency.yaml'
        # with open('configs/temp_infllm_latency.yaml', 'w') as _out:
        #     yaml.dump(infllm_config, _out, default_flow_style=False)
        infllm_cfg_path = "baselines/infllm/config/llama-3-inf-final.yaml"

    std_json_template = {
        "batch_size": 1,
        "model_name": model_name,
        "model_cls": model_cls,
        "max_context_len": context_len,
        "rope_factor": -1,
        # omnikv hype parameters
        "do_select_layers": "2,8,18",
        "num_wait_load_layers": 1,
        "num_of_selected_tokens": num_sel_tokens,
        "dense_more": True,
        "cache_cls": cache_cls,
        "real_offload": offload,
        "selector_cls": "last",
        # h2o hype parameters
        "hh_ratio": h2o_ratio / 2,
        "recent_ratio": h2o_ratio / 2,
        "seg_size": h2o_seg_sz,
        "skip_prefill": True,  # 如果是False，就也会在prefill阶段进行丢弃
        # brutal offload hyper parameters
        "decoder_start_layer_idx": 6,
        "offload_sid": 10,
        # infllm hyper parameter
        "infllm_cfg_path": infllm_cfg_path,
        # full
        "use_multi_gpus": use_multi_gpus,
        "load_in_4bit": load_in_4bit,
        "use_flash_attn": ("h2o" not in model_cls),
        "use_fixed_prompt": True,
        "use_chat_template": False,
        "cpu_num_threads": cpu_num_threads,
    }
    if cfg_path is None:
        cfg_path = gen_temp_cfg(**std_json_template)
    chat, tkn, _, __ = get_any_chat_api(cfg_path)
    prompt = ["hi" for i in range(context_len)]
    prompt = " ".join(prompt)
    print("tokenized prompt len = ", len(tkn.encode(prompt)))
    # 只是为了能持续输出
    gen_cfg = GenerationConfig(
        temperature=100.0, do_sample=True, top_p=0.8, max_new_tokens=50
    )
    for i in trange(5):
        out = chat(prompt, gen_cfg)

    t_prefill, t_decode, t_chat = [], [], []
    thats_me = False
    prefill_re = r"prefill time ([0-9.]+)s?"
    decode_re = r"decoding time ([0-9.]+)s?"
    if model_cls == "infllm":
        prefill_re = "infllm " + prefill_re
        decode_re = "infllm " + decode_re
    with open(logger_file_path, "r", encoding="utf-8") as _in:
        for i, line in enumerate(_in):
            if i <= last_line:
                continue
            last_line = i
            if oo := re.search(prefill_re, line):
                t_prefill += [float(oo.group(1))]
                thats_me = True
            elif oo := re.search(decode_re, line):
                if thats_me:
                    thats_me = False
                    t_prefill += [float(oo.group(1))]
                else:
                    t_decode += [float(oo.group(1))]
            elif oo := re.search(r"chat time: ([0-9.]+)s?", line):
                t_chat += [float(oo.group(1))]

    print("=" * 10)
    print(
        f"context_len={context_len}; model_cls={model_cls}; num_sel_tokens={num_sel_tokens}"
    )
    print(
        f"cpu_num_threads={cpu_num_threads}; offload={offload}; cache_cls={cache_cls}"
    )
    if "70" in model_name:
        print("70B")
    # print(t_prefill)
    # print(t_decode)
    t_prefill = t_prefill[2:]  # 去掉加载模型的影响
    print("prefill time", sum(t_prefill) * 2 / len(t_prefill))
    print("decode time", sum(t_decode) / len(t_decode))
    print("chat time", sum(t_chat) / len(t_chat))
    print("=" * 10)


if __name__ == "__main__":
    Fire(run)
