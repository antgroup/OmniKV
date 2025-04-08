import argparse
import json
import os.path
import sys
import tqdm
import torch
import copy
from copy import deepcopy
import dataclasses
from xopen import xopen
import math
import matplotlib.pyplot as plt

from rouge import Rouge
import logging
import numpy as np
import time
from torch.cuda.amp import autocast

sys.path.append("baselines/raw_h2o/")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from utils_real_drop.modify_llama import H2OLlamaForCausalLM, H2OLlamaAttention
from tiny_tools.read_json import read_config
from tiny_tools.log import logger
from transformers import set_seed, BitsAndBytesConfig
from configs.template_for_chat import get_chat_template
from modeling.omnikv_config import LlamaCompressorConfig

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
old_prompt = "You are a pirate chatbot who always responds in pirate speak!"
fixed_prompt = "You are a helpful assistant."


def h2o_infer_1(
    prompt,
    tkn,
    model,
    generation_config=None,
    use_chat_template=False,
    use_cot=False,
    model_name=None,
    **kwargs,
):
    with torch.no_grad():
        with autocast():
            terminators = [] + [tkn.eos_token_id] if tkn.eos_token_id else []
            for eos_token in ["<|eot_id|>", "<|endoftext|>", "<|im_end|>"]:
                if eos_token in tkn.vocab:
                    terminators += [tkn.convert_tokens_to_ids(eos_token)]
            if use_chat_template:
                template = get_chat_template(model_name, use_cot)
                prompt = template.format(
                    user_message=prompt, system_prompt="You are a helpful assistant."
                )
                input_ids = tkn(prompt, return_tensors="pt")["input_ids"]
            else:
                input_ids = tkn(fixed_prompt + prompt, return_tensors="pt")["input_ids"]
            n = input_ids.shape[1]
            for name, m in model.named_modules():
                if isinstance(m, H2OLlamaAttention):
                    m.upd_token_budget_based_on_seq_len(n)
            return model.generate(
                inputs=input_ids.cuda(),
                generation_config=generation_config,
                eos_token_id=terminators,
                **kwargs,
            )[:, n:]


def get_chat_api(cfg_path):
    config_j = read_config(cfg_path)
    device = 0
    n_gpu = config_j.get("n_gpu", 1)
    seed = config_j.get("seed", 42)
    set_seed(seed)

    model_name = config_j["model_name"]
    use_cot = config_j.get("cot", False)
    use_chat_template = config_j.get("cot", False)
    if use_cot:
        assert use_chat_template

    config = LlamaCompressorConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Enabling H2O KV cache")
    config.hh_ratio = config_j["hh_ratio"]
    config.recent_ratio = config_j["recent_ratio"]
    config.seg_size = config_j.get("seg_size", 500)
    load_in_8bit = config_j.get("load_in_8bit", False)
    load_in_4bit = config_j.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = H2OLlamaForCausalLM.from_pretrained(
        model_name, config=config, quantization_config=quant_config
    )
    if quant_config is None:
        model = model.half().eval().cuda()

    def chat(prompt, generation_config=None, skip_special_tokens=True, **kwargs):
        st = time.time()
        with torch.no_grad():
            out_ids = h2o_infer_1(
                prompt,
                tokenizer,
                model,
                generation_config,
                use_cot=use_cot,
                use_chat_template=use_chat_template,
                model_name=model_name,
                **kwargs,
            )
            for name, m in model.named_modules():
                if isinstance(m, H2OLlamaAttention):
                    m._clean_cache()

            res = tokenizer.batch_decode(
                out_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )[0]
        logger.info(f"--- chat time: {round(time.time() - st, 3)}s")
        return res

    return (
        chat,
        tokenizer,
        config_j["max_context_len"],
        {"eos_token_id": tokenizer.eos_token_id},
    )
